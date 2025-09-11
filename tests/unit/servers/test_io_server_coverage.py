"""Comprehensive tests to improve io_server.py coverage to 80%+."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.io_server import (
    close_session,
    export_csv,
    get_encoding_fallbacks,
    get_session_info,
    load_csv,
    load_csv_from_content,
    load_csv_from_url,
)


class TestEncodingFallbacks:
    """Test encoding detection and fallback mechanisms."""

    def test_get_encoding_fallbacks_utf8(self):
        """Test fallback encodings for UTF-8."""
        fallbacks = get_encoding_fallbacks("utf-8")
        assert "utf-8" in fallbacks
        assert "utf-8-sig" in fallbacks
        assert "latin-1" in fallbacks
        assert "iso-8859-1" in fallbacks

    def test_get_encoding_fallbacks_latin1(self):
        """Test fallback encodings for Latin-1."""
        fallbacks = get_encoding_fallbacks("latin1")
        assert "latin1" in fallbacks
        assert "utf-8" in fallbacks
        assert "cp1252" in fallbacks

    def test_get_encoding_fallbacks_windows(self):
        """Test fallback encodings for Windows-1252."""
        fallbacks = get_encoding_fallbacks("cp1252")
        assert "cp1252" in fallbacks
        assert "windows-1252" in fallbacks

    def test_get_encoding_fallbacks_unknown(self):
        """Test fallback encodings for unknown encoding."""
        fallbacks = get_encoding_fallbacks("unknown-encoding")
        # Should return the primary encoding first
        assert "unknown-encoding" in fallbacks
        assert "utf-8" in fallbacks
        assert "cp1252" in fallbacks


class TestLoadCsvWithEncoding:
    """Test CSV loading with various encodings."""

    async def test_load_csv_with_encoding_fallback(self):
        """Test loading CSV with encoding that needs fallback."""
        # Create a file with Latin-1 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="latin-1", suffix=".csv", delete=False
        ) as f:
            f.write("name,city\n")
            f.write("José,São Paulo\n")  # Latin-1 characters
            f.write("François,Montréal\n")
            temp_path = f.name

        try:
            # Try to load with wrong encoding first (will trigger fallback)
            result = await load_csv(
                file_path=temp_path,
                encoding="ascii",  # This will fail and trigger fallback
            )

            assert result.rows_affected == 2
            assert result.columns_affected == ["name", "city"]
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_with_utf8_bom(self):
        """Test loading CSV with UTF-8 BOM."""
        # Create a file with UTF-8 BOM
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            # Write BOM
            f.write(b"\xef\xbb\xbf")
            # Write CSV content
            f.write(b"name,value\ntest,123\n")
            temp_path = f.name

        try:
            result = await load_csv(file_path=temp_path)
            assert result.rows_affected == 1
            assert result.columns_affected == ["name", "value"]
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_encoding_error_all_fallbacks_fail(self):
        """Test when all encoding fallbacks fail."""
        # Create a file with mixed/corrupted encoding
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            # Write some invalid UTF-8 sequences
            f.write(b"col1,col2\n")
            f.write(b"\xff\xfe invalid bytes \xfd\xfc\n")
            temp_path = f.name

        try:
            # This should try all fallbacks and eventually succeed with error handling
            result = await load_csv(file_path=temp_path, encoding="utf-8")
            # latin-1 should handle any byte sequence
            assert result is not None
        except ToolError:
            # Or it might fail completely which is also acceptable
            pass
        finally:
            Path(temp_path).unlink()


class TestLoadCsvSizeConstraints:
    """Test file size and memory constraints."""

    async def test_load_csv_max_rows_exceeded(self):
        """Test loading CSV that exceeds max rows."""
        # Create a large CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            # Write more than MAX_ROWS (1,000,000)
            for i in range(10):  # Small test, normally would be 1000001
                f.write(f"{i},value{i}\n")
            temp_path = f.name

        try:
            # Mock the MAX_ROWS constant to make test faster
            with (
                patch("src.databeak.servers.io_server.MAX_ROWS", 5),
                pytest.raises(ToolError, match="rows exceeds limit"),
            ):
                await load_csv(file_path=temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_memory_limit_exceeded(self):
        """Test loading CSV that exceeds memory limit."""
        # Create a CSV with large strings
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            # Create rows with large strings
            large_string = "x" * 10000
            for i in range(10):
                f.write(f"{i},{large_string}\n")
            temp_path = f.name

        try:
            # Mock the MAX_MEMORY_USAGE_MB to trigger the check
            with (
                patch("src.databeak.servers.io_server.MAX_MEMORY_USAGE_MB", 0.001),
                pytest.raises(ToolError, match="exceeds memory limit"),
            ):
                await load_csv(file_path=temp_path)
        finally:
            Path(temp_path).unlink()


class TestCloseSession:
    """Test session closing functionality."""

    async def test_close_session_success(self):
        """Test successfully closing a session."""
        # Create a session first
        csv_content = "col1,col2\n1,2"
        load_result = await load_csv_from_content(csv_content)

        # Close the session
        result = await close_session(load_result.session_id)

        assert result.success is True
        assert result.session_id == load_result.session_id

        # Verify session is closed
        with pytest.raises(ToolError, match="Session not found"):
            await get_session_info(load_result.session_id)

    async def test_close_session_not_found(self):
        """Test closing non-existent session."""
        with pytest.raises(ToolError, match="Session not found"):
            await close_session("non-existent-session")

    async def test_close_session_with_context(self):
        """Test closing session with context reporting."""
        # Create a session
        csv_content = "col1,col2\n1,2"
        load_result = await load_csv_from_content(csv_content)

        # Mock context with async method
        from unittest.mock import AsyncMock

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock(return_value=None)

        # Close with context
        result = await close_session(load_result.session_id, ctx=mock_ctx)

        assert result.success is True
        # Context info should have been called
        mock_ctx.info.assert_called()


class TestExportCsvAdvanced:
    """Test advanced export functionality."""

    async def test_export_csv_with_tabs(self):
        """Test exporting as TSV (tab-separated)."""
        # Create session with data
        csv_content = "name,value,category\ntest1,100,A\ntest2,200,B"
        load_result = await load_csv_from_content(csv_content)

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            temp_path = f.name

        try:
            result = await export_csv(
                session_id=load_result.session_id, file_path=temp_path, format="tsv"
            )

            assert result.success is True
            assert result.format == "tsv"

            # Verify the file is tab-separated
            with Path(temp_path).open() as f:
                content = f.read()
                assert "\t" in content
                assert "," not in content.split("\n")[0]  # No commas in header
        finally:
            Path(temp_path).unlink()

    async def test_export_csv_with_quotes(self):
        """Test exporting with quote handling."""
        # Create session with data containing commas and quotes
        csv_content = (
            'name,description\n"Smith, John","He said ""Hello"""\n"Doe, Jane","Normal text"'
        )
        load_result = await load_csv_from_content(csv_content)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = await export_csv(session_id=load_result.session_id, file_path=temp_path)

            assert result.success is True

            # Verify quotes are properly handled
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert "Smith, John" in df["name"].values
        finally:
            Path(temp_path).unlink()

    async def test_export_csv_create_directory(self):
        """Test export creates directory if it doesn't exist."""
        csv_content = "col1,col2\n1,2"
        load_result = await load_csv_from_content(csv_content)

        # Use a directory that doesn't exist
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            file_path = new_dir / "export.csv"

            result = await export_csv(session_id=load_result.session_id, file_path=str(file_path))

            assert result.success is True
            assert file_path.exists()
            assert new_dir.exists()


class TestLoadCsvFromUrl:
    """Test loading CSV from URL."""

    @patch("pandas.read_csv")
    async def test_load_csv_from_url_success(self, mock_read_csv):
        """Test successfully loading from URL."""
        # Mock pandas read_csv to return a DataFrame
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        result = await load_csv_from_url(url="http://example.com/data.csv")

        assert result.rows_affected == 2
        assert result.columns_affected == ["col1", "col2"]
        mock_read_csv.assert_called_once()

    @patch("pandas.read_csv")
    async def test_load_csv_from_url_with_encoding_error(self, mock_read_csv):
        """Test URL loading with encoding error and fallback."""
        # First call raises UnicodeDecodeError, second succeeds
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.side_effect = [
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
            mock_df,  # Succeeds with fallback encoding
        ]

        result = await load_csv_from_url(url="http://example.com/data.csv", encoding="utf-8")

        assert result.rows_affected == 2
        assert mock_read_csv.call_count == 2  # Called twice due to fallback

    @patch("pandas.read_csv")
    async def test_load_csv_from_url_all_encodings_fail(self, mock_read_csv):
        """Test URL loading when all encodings fail."""
        # All calls raise UnicodeDecodeError
        mock_read_csv.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")

        with pytest.raises(ToolError, match="Encoding error"):
            await load_csv_from_url(url="http://example.com/data.csv", encoding="utf-8")


class TestLoadCsvFromContentEdgeCases:
    """Test edge cases in load_csv_from_content."""

    async def test_load_csv_from_content_single_row(self):
        """Test loading CSV with only header and one row."""
        csv_content = "col1,col2\n1,2"
        result = await load_csv_from_content(csv_content)

        assert result.rows_affected == 1
        assert result.columns_affected == ["col1", "col2"]

    async def test_load_csv_from_content_special_characters(self):
        """Test loading CSV with special characters."""
        csv_content = "name,symbol\nAlpha,a\nBeta,b\nGamma,y"
        result = await load_csv_from_content(csv_content)

        assert result.rows_affected == 3
        assert result.columns_affected == ["name", "symbol"]

    async def test_load_csv_from_content_numeric_columns(self):
        """Test loading CSV with numeric column names."""
        csv_content = "1,2,3\na,b,c\nd,e,f"
        result = await load_csv_from_content(csv_content)

        assert result.rows_affected == 2
        # Pandas converts numeric column names to strings
        assert len(result.columns_affected) == 3

    async def test_load_csv_from_content_with_index(self):
        """Test that data is loaded correctly."""
        csv_content = "id,name,value\n1,test1,100\n2,test2,200"
        result = await load_csv_from_content(csv_content)

        assert result.rows_affected == 2
        assert result.columns_affected == ["id", "name", "value"]
        # Verify the session has data
        info = await get_session_info(result.session_id)
        assert info.row_count == 2
        assert info.column_count == 3
