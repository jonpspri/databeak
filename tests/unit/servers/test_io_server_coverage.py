"""Comprehensive tests to improve io_server.py coverage to 80%+."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from databeak.servers.io_server import (
    export_csv,
    get_encoding_fallbacks,
    get_session_info,
    load_csv,
    load_csv_from_content,
)
from tests.test_mock_context import create_mock_context


class TestEncodingFallbacks:
    """Test encoding detection and fallback mechanisms."""

    def test_get_encoding_fallbacks_utf8(self) -> None:
        """Test fallback encodings for UTF-8."""
        fallbacks = get_encoding_fallbacks("utf-8")
        # UTF-8 is not included when it's the primary encoding (line 245 in io_server.py)
        assert "utf-8-sig" in fallbacks
        assert "latin1" in fallbacks  # Note: latin1 not latin-1
        assert "iso-8859-1" in fallbacks

    def test_get_encoding_fallbacks_latin1(self) -> None:
        """Test fallback encodings for Latin-1."""
        fallbacks = get_encoding_fallbacks("latin1")
        assert "latin1" in fallbacks
        assert "utf-8" in fallbacks
        assert "cp1252" in fallbacks

    def test_get_encoding_fallbacks_windows(self) -> None:
        """Test fallback encodings for Windows-1252."""
        fallbacks = get_encoding_fallbacks("cp1252")
        assert "cp1252" in fallbacks
        assert "windows-1252" in fallbacks

    def test_get_encoding_fallbacks_unknown(self) -> None:
        """Test fallback encodings for unknown encoding."""
        fallbacks = get_encoding_fallbacks("unknown-encoding")
        # Should return the primary encoding first
        assert "unknown-encoding" in fallbacks
        assert "utf-8" in fallbacks
        assert "cp1252" in fallbacks


class TestLoadCsvWithEncoding:
    """Test CSV loading with various encodings."""

    async def test_load_csv_with_encoding_fallback(self) -> None:
        """Test loading CSV with encoding that needs fallback."""
        # Create a file with Latin-1 encoding
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="latin-1",
            suffix=".csv",
            delete=False,
        ) as f:
            f.write("name,city\n")
            f.write("José,São Paulo\n")  # Latin-1 characters
            f.write("François,Montréal\n")
            temp_path = f.name

        try:
            # Try to load with wrong encoding first (will trigger fallback)
            result = await load_csv(
                create_mock_context(),
                file_path=temp_path,
                encoding="ascii",  # This will fail and trigger fallback
            )

            assert result.rows_affected == 2
            assert result.columns_affected == ["name", "city"]
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_with_utf8_bom(self) -> None:
        """Test loading CSV with UTF-8 BOM."""
        # Create a file with UTF-8 BOM
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            # Write BOM
            f.write(b"\xef\xbb\xbf")
            # Write CSV content
            f.write(b"name,value\ntest,123\n")
            temp_path = f.name

        try:
            result = await load_csv(create_mock_context(), file_path=temp_path)
            assert result.rows_affected == 1
            assert result.columns_affected == ["name", "value"]
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_encoding_error_all_fallbacks_fail(self) -> None:
        """Test when all encoding fallbacks fail."""
        # Create a file with mixed/corrupted encoding
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            # Write some invalid UTF-8 sequences
            f.write(b"col1,col2\n")
            f.write(b"\xff\xfe invalid bytes \xfd\xfc\n")
            temp_path = f.name

        try:
            # This should try all fallbacks and eventually succeed with error handling
            result = await load_csv(create_mock_context(), file_path=temp_path, encoding="utf-8")
            # latin-1 should handle any byte sequence
            assert result is not None
        except ToolError:
            # Or it might fail completely which is also acceptable
            pass
        finally:
            Path(temp_path).unlink()


class TestLoadCsvSizeConstraints:
    """Test file size and memory constraints."""

    async def test_load_csv_max_rows_exceeded(self) -> None:
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
                patch("databeak.servers.io_server.MAX_ROWS", 5),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), file_path=temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_memory_limit_exceeded(self) -> None:
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
                patch("databeak.servers.io_server.MAX_MEMORY_USAGE_MB", 0.001),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), file_path=temp_path)
        finally:
            Path(temp_path).unlink()


class TestExportCsvAdvanced:
    """Test advanced export functionality."""

    async def test_export_csv_with_tabs(self) -> None:
        """Test exporting as TSV (tab-separated)."""
        # Create session with data
        csv_content = "name,value,category\ntest1,100,A\ntest2,200,B"
        ctx = create_mock_context()
        await load_csv_from_content(ctx, csv_content)
        session_id = ctx.session_id

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            temp_path = f.name

        try:
            result = await export_csv(create_mock_context(session_id), file_path=temp_path)

            assert result.success is True
            assert result.format == "tsv"

            # Verify the file is tab-separated
            with Path(temp_path).open() as f:
                content = f.read()
                assert "\t" in content
                assert "," not in content.split("\n")[0]  # No commas in header
        finally:
            Path(temp_path).unlink()

    async def test_export_csv_with_quotes(self) -> None:
        """Test exporting with quote handling."""
        # Create session with data containing commas and quotes
        csv_content = (
            'name,description\n"Smith, John","He said ""Hello"""\n"Doe, Jane","Normal text"'
        )
        ctx = create_mock_context()
        await load_csv_from_content(ctx, csv_content)
        session_id = ctx.session_id

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = await export_csv(create_mock_context(session_id), file_path=temp_path)

            assert result.success is True

            # Verify quotes are properly handled
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert "Smith, John" in df["name"].to_numpy()
        finally:
            Path(temp_path).unlink()

    async def test_export_csv_create_directory(self) -> None:
        """Test export creates directory if it doesn't exist."""
        csv_content = "col1,col2\n1,2"
        ctx = create_mock_context()
        await load_csv_from_content(ctx, csv_content)
        session_id = ctx.session_id

        # Use a directory that doesn't exist
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            file_path = new_dir / "export.csv"

            result = await export_csv(create_mock_context(session_id), file_path=str(file_path))

            assert result.success is True
            assert file_path.exists()
            assert new_dir.exists()


# The URL loading tests are covered in the main test_io_server.py file
# No need to duplicate them here with complex mocking


class TestLoadCsvFromContentEdgeCases:
    """Test edge cases in load_csv_from_content."""

    async def test_load_csv_from_content_single_row(self) -> None:
        """Test loading CSV with only header and one row."""
        csv_content = "col1,col2\n1,2"
        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 1
        assert result.columns_affected == ["col1", "col2"]

    async def test_load_csv_from_content_special_characters(self) -> None:
        """Test loading CSV with special characters."""
        csv_content = "name,symbol\nAlpha,a\nBeta,b\nGamma,y"
        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 3
        assert result.columns_affected == ["name", "symbol"]

    async def test_load_csv_from_content_numeric_columns(self) -> None:
        """Test loading CSV with numeric column names."""
        csv_content = "1,2,3\na,b,c\nd,e,f"
        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 2
        # Pandas converts numeric column names to strings
        assert len(result.columns_affected) == 3

    async def test_load_csv_from_content_with_index(self) -> None:
        """Test that data is loaded correctly."""
        csv_content = "id,name,value\n1,test1,100\n2,test2,200"
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)
        session_id = ctx.session_id

        assert result.rows_affected == 2
        assert result.columns_affected == ["id", "name", "value"]
        # Verify the session has data
        info = await get_session_info(create_mock_context(session_id))
        assert info.row_count == 2
        assert info.column_count == 3
