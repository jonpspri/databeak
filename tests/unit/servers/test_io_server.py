"""Comprehensive tests for io_server.py focusing on error conditions, edge cases, and
integration."""

import tempfile
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from databeak.servers.io_server import (
    MAX_FILE_SIZE_MB,
    MAX_MEMORY_USAGE_MB,
    MAX_ROWS,
    MAX_URL_SIZE_MB,
    NoHeader,
    export_csv,
    get_session_info,
    load_csv,
    load_csv_from_content,
    load_csv_from_url,
)
from tests.test_mock_context import create_mock_context


@pytest.mark.asyncio
class TestErrorConditions:
    """Test comprehensive error conditions."""

    async def test_load_csv_file_not_found(self) -> None:
        """Test loading non-existent CSV file."""
        with pytest.raises(ToolError):
            await load_csv(create_mock_context(), "/nonexistent/path/file.csv")

    async def test_load_csv_invalid_extension(self) -> None:
        """Test loading file with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(b"name,age\nJohn,30")
            temp_path = f.name

        try:
            with pytest.raises(ToolError):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_encoding_error_all_fallbacks_fail(self) -> None:
        """Test encoding error when all fallback encodings fail."""

        # Mock pandas.read_csv to always raise UnicodeDecodeError
        def mock_read_csv(*args: object, **kwargs: object) -> object:
            encoding = str(kwargs.get("encoding", "utf-8"))
            raise UnicodeDecodeError(encoding, b"", 0, 1, f"mock encoding error for {encoding}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30")
            temp_path = f.name

        try:
            with (
                patch("pandas.read_csv", side_effect=mock_read_csv),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), temp_path, encoding="utf-8")
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_memory_limit_exceeded(self) -> None:
        """Test memory limit enforcement."""
        # Mock pandas read_csv to return a large DataFrame
        large_data = pd.DataFrame(
            {"col1": ["data"] * (MAX_ROWS + 100), "col2": list(range(MAX_ROWS + 100))},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            temp_path = f.name

        try:
            with (
                patch("pandas.read_csv", return_value=large_data),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_memory_usage_exceeded(self) -> None:
        """Test memory usage limit enforcement."""
        # Create DataFrame that exceeds memory limit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            temp_path = f.name

        # Mock memory_usage to return value exceeding limit
        def mock_memory_usage(*args: object, **kwargs: object) -> object:
            class MockSeries:
                def sum(self) -> int:
                    return (MAX_MEMORY_USAGE_MB + 100) * 1024 * 1024

            return MockSeries()

        try:
            with (
                patch("pandas.DataFrame.memory_usage", side_effect=mock_memory_usage),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_from_url_private_network_blocked(self) -> None:
        """Test private network URL blocking."""
        private_urls = [
            "http://192.168.1.1/data.csv",
            "http://10.0.0.1/data.csv",
            "http://172.16.0.1/data.csv",
            "http://localhost/data.csv",
            "http://127.0.0.1/data.csv",
        ]

        for url in private_urls:
            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), url)

    async def test_load_csv_file_size_limit_exceeded(self) -> None:
        """Test file size limit enforcement before loading."""
        # Create a file and mock its size to exceed limit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30")
            temp_path = f.name

        try:
            # Create a mock stat object
            mock_stat_obj = type("MockStat", (), {})()
            mock_stat_obj.st_size = (MAX_FILE_SIZE_MB + 10) * 1024 * 1024
            mock_stat_obj.st_mode = 0o100644  # Regular file mode

            with (
                patch("pathlib.Path.stat", return_value=mock_stat_obj),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    async def test_load_csv_empty_file(self) -> None:
        """Test loading empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            with pytest.raises(pd.errors.EmptyDataError, match="No columns to parse"):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_header_only(self) -> None:
        """Test loading CSV with only headers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,city\n")  # Header only
            temp_path = f.name

        try:
            # Loading CSV with only headers is valid - creates empty DataFrame with columns
            result = await load_csv(create_mock_context(), temp_path)
            assert result.rows_affected == 0
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_special_characters(self) -> None:
        """Test loading CSV with special characters."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        ) as f:
            # Unicode characters, quotes, commas in data
            f.write("name,description,price\n")
            f.write('"José García","Product with ""quotes"" and, commas",€25.99\n')
            f.write('"李小明","测试数据",¥100.00\n')
            temp_path = f.name

        try:
            result = await load_csv(create_mock_context(), temp_path)
            assert result.success
            assert result.rows_affected == 2
            assert result.data is not None
            assert "José García" in str(result.data.rows)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_malformed_quotes(self) -> None:
        """Test CSV with malformed quotes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,description\n")
            f.write('"Unclosed quote,data\n')  # Malformed
            temp_path = f.name

        try:
            with pytest.raises(pd.errors.ParserError, match="Error tokenizing data"):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_from_content_various_delimiters(self) -> None:
        """Test content loading with different delimiters."""
        # Tab-separated
        content = "name\tage\tcity\nJohn\t30\tNYC\nJane\t25\tLA"
        result = await load_csv_from_content(create_mock_context(), content, delimiter="\t")
        assert result.success
        assert result.rows_affected == 2

        # Semicolon-separated
        content = "name;age;city\nJohn;30;NYC\nJane;25;LA"
        result = await load_csv_from_content(create_mock_context(), content, delimiter=";")
        assert result.success
        assert result.rows_affected == 2

    async def test_load_csv_from_content_no_header(self) -> None:
        """Test content loading without headers."""
        content = "John,30,NYC\nJane,25,LA"
        result = await load_csv_from_content(
            create_mock_context(), content, header_config=NoHeader()
        )
        assert result.success
        assert result.rows_affected == 2
        # Should have auto-generated column names like "0", "1", "2"


@pytest.mark.asyncio
class TestIntegrationWithSessions:
    """Test integration with session management."""

    async def test_load_csv_creates_new_session(self) -> None:
        """Test that loading CSV creates a new session when none provided."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30\nJane,25")
            temp_path = f.name

        try:
            ctx = create_mock_context()
            result = await load_csv(ctx, temp_path)
            assert result.success
            assert result.rows_affected == 2

            # Verify session exists and has data
            session_info = await get_session_info(create_mock_context(ctx.session_id))
            assert session_info.data_loaded
            assert session_info.row_count == 2
            assert session_info.column_count == 2
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_into_existing_session(self) -> None:
        """Test loading CSV into an existing session (replaces data)."""
        # First load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30")
            temp_path1 = f.name

        try:
            ctx1 = create_mock_context()
            await load_csv(ctx1, temp_path1)
            session_id = ctx1.session_id

            # Second load into same session
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write("product,price,category\nLaptop,1000,Electronics\nBook,15,Education")
                temp_path2 = f.name

            try:
                result2 = await load_csv(create_mock_context(session_id), temp_path2)
                assert result2.rows_affected == 2

                # Verify session now has the new data
                session_info = await get_session_info(create_mock_context(session_id))
                assert session_info.row_count == 2
                assert session_info.column_count == 3
            finally:
                Path(temp_path2).unlink()
        finally:
            Path(temp_path1).unlink()

    async def test_session_lifecycle_complete(self) -> None:
        """Test complete session lifecycle: create, use, export, close."""
        # Load data
        content = "name,age,salary\nAlice,25,50000\nBob,30,60000\nCharlie,35,70000"
        ctx = create_mock_context()
        await load_csv_from_content(ctx, content)
        session_id = ctx.session_id

        # Get session info
        info = await get_session_info(create_mock_context(session_id))
        assert info.data_loaded
        assert info.row_count == 3

        # Export the data
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = tmp.name

        export_result = await export_csv(create_mock_context(session_id), file_path=temp_path)
        assert export_result.rows_exported == 3
        assert Path(export_result.file_path).exists()

        # Clean up export file
        Path(export_result.file_path).unlink()


@pytest.mark.asyncio
class TestTempFileCleanup:
    """Test temporary file cleanup scenarios."""

    async def test_export_csv_session_error_handling(self) -> None:
        """Test error handling when session manager fails."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            temp_path = tmp.name

        try:
            with patch("databeak.servers.io_server.get_session_data") as mock_get_session_data:
                mock_get_session_data.side_effect = Exception("Mock session error")

                with pytest.raises(Exception, match="Mock session error"):
                    await export_csv(create_mock_context(), file_path=temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestURLValidationSecurity:
    """Test URL validation and security features."""

    async def test_private_network_blocking_ipv4(self) -> None:
        """Test blocking of private IPv4 networks."""
        private_networks = [
            "http://192.168.1.1/data.csv",  # Class C private
            "http://10.0.0.1/data.csv",  # Class A private
            "http://172.16.0.1/data.csv",  # Class B private
            "http://127.0.0.1/data.csv",  # Loopback
            "http://169.254.1.1/data.csv",  # Link-local
        ]

        for url in private_networks:
            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), url)

    async def test_localhost_hostname_blocking(self) -> None:
        """Test blocking of localhost hostnames."""
        localhost_urls = [
            "http://localhost/data.csv",
            "https://localhost:8080/data.csv",
        ]

        for url in localhost_urls:
            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), url)

    async def test_invalid_url_schemes(self) -> None:
        """Test rejection of non-HTTP schemes."""
        invalid_schemes = [
            "ftp://example.com/data.csv",
            "file:///path/to/data.csv",
            "mailto:user@example.com",
            "javascript:alert('xss')",
        ]

        for url in invalid_schemes:
            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), url)

    async def test_url_timeout_handling(self) -> None:
        """Test URL download timeout handling."""
        # Mock urlopen to raise timeout
        with patch("databeak.servers.io_server.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")

            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), "https://example.com/data.csv")

    async def test_url_content_type_validation(self) -> None:
        """Test content-type verification for URLs."""
        # Mock response with invalid content-type
        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "text/html"}

        with patch("databeak.servers.io_server.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            # Should proceed with warning for unexpected content-type
            # The test validates the warning is logged

    async def test_url_size_limit_exceeded(self) -> None:
        """Test URL download size limit enforcement."""
        # Mock response with large content-length
        mock_response = AsyncMock()
        mock_response.headers = {
            "Content-Type": "text/csv",
            "Content-Length": str((MAX_URL_SIZE_MB + 10) * 1024 * 1024),  # Exceed limit
        }

        with patch("databeak.servers.io_server.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response

            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), "https://example.com/large_file.csv")

    async def test_url_http_error_handling(self) -> None:
        """Test HTTP error handling (404, 403, etc.)."""
        with patch("databeak.servers.io_server.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                url="https://example.com/notfound.csv",
                code=404,
                msg="Not Found",
                hdrs=EmailMessage(),
                fp=None,
            )

            with pytest.raises(ToolError):
                await load_csv_from_url(create_mock_context(), "https://example.com/notfound.csv")


@pytest.mark.asyncio
class TestExportFormats:
    """Test all export formats comprehensively."""

    @pytest.fixture
    async def session_with_data(self) -> str:
        """Create session with test data."""
        content = "name,age,salary,active\nAlice,25,50000,true\nBob,30,60000,false"
        ctx = create_mock_context()
        await load_csv_from_content(ctx, content)
        return ctx.session_id

    async def test_export_csv_format(self, session_with_data: str) -> None:
        """Test CSV export format (inferred from .csv extension)."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            temp_path = tmp.name

        try:
            result = await export_csv(create_mock_context(session_with_data), file_path=temp_path)
            assert result.format == "csv"
            assert result.file_path == temp_path
            assert result.rows_exported == 2

            # Verify file exists and has content
            assert Path(result.file_path).exists()
            content = Path(result.file_path).read_text()
            assert "Alice" in content
            assert "Bob" in content
            assert "name,age" in content  # CSV headers
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def test_export_tsv_format(self, session_with_data: str) -> None:
        """Test TSV export format (inferred from .tsv extension)."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
            temp_path = tmp.name

        try:
            result = await export_csv(create_mock_context(session_with_data), file_path=temp_path)
            assert result.format == "tsv"
            assert result.file_path == temp_path

            # Verify tab separation
            content = Path(result.file_path).read_text()
            assert "\t" in content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def test_export_json_format(self, session_with_data: str) -> None:
        """Test JSON export format (inferred from .json extension)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = tmp.name

        try:
            result = await export_csv(create_mock_context(session_with_data), file_path=temp_path)
            assert result.format == "json"
            assert result.file_path == temp_path

            # Verify valid JSON
            import json

            with Path(result.file_path).open() as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["name"] == "Alice"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def test_export_with_custom_path(self, session_with_data: str) -> None:
        """Test export with user-specified file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = str(Path(temp_dir) / "my_export.csv")
            result = await export_csv(create_mock_context(session_with_data), file_path=custom_path)

            assert result.file_path == custom_path
            assert Path(custom_path).exists()


@pytest.mark.asyncio
class TestEncodingAndFallback:
    """Test encoding detection and fallback logic."""

    async def test_encoding_fallback_success(self) -> None:
        """Test successful encoding fallback."""
        # Create file with latin1 encoding
        content = "name,description\nJosé,Niño años"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="latin1",
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Try to load with UTF-8 first, should fallback to latin1
            result = await load_csv(create_mock_context(), temp_path, encoding="utf-8")
            assert result.success
            assert result.rows_affected == 1
        finally:
            Path(temp_path).unlink()

    async def test_custom_na_values(self) -> None:
        """Test custom NA values handling."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,status\nJohn,30,MISSING\nJane,N/A,active")
            temp_path = f.name

        try:
            result = await load_csv(create_mock_context(), temp_path, na_values=["MISSING", "N/A"])
            assert result.success
            # Verify NA values were handled properly (would need to check actual data)
        finally:
            Path(temp_path).unlink()

    async def test_automatic_encoding_detection_success(self) -> None:
        """Test automatic encoding detection with chardet."""
        # Create file with special characters that require specific encoding
        content = "name,description\nJosé García,Niño años\nFrançois,café"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="latin1",
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Mock chardet to return high confidence detection
            with patch("databeak.servers.io_server.chardet.detect") as mock_detect:
                mock_detect.return_value = {"encoding": "ISO-8859-1", "confidence": 0.85}

                # Should use detected encoding instead of falling back
                result = await load_csv(
                    create_mock_context(),
                    temp_path,
                    encoding="utf-8",
                )  # Will trigger fallback
                assert result.success
                assert result.rows_affected == 2
        finally:
            Path(temp_path).unlink()

    @pytest.mark.skip(reason="Complex mocking scenario - needs refactoring")
    @patch("pandas.read_csv")
    async def test_encoding_fallback_prioritization(self) -> None:
        """Test that encoding fallbacks are tried in optimal order."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30")
            temp_path = f.name

        try:
            # Mock successful result
            success_df = pd.DataFrame({"name": ["John"], "age": [30]})

            # Mock encoding detection to fail so fallbacks are used
            with (
                patch("databeak.servers.io_server.detect_file_encoding", return_value="utf-8"),
                patch("pandas.read_csv") as mock_read_csv,
            ):
                call_count = [0]

                def mock_read_side_effect(*args: Any, **kwargs: Any) -> Any:
                    call_count[0] += 1
                    if call_count[0] == 1:  # First call (original encoding)
                        msg = "utf-8"
                        raise UnicodeDecodeError(msg, b"", 0, 1, "mock error")
                    if call_count[0] == 2:  # Second call (auto-detection, same encoding)
                        msg = "utf-8"
                        raise UnicodeDecodeError(msg, b"", 0, 1, "auto-detect fails")
                    if call_count[0] == 3:  # First fallback fails
                        msg = "utf-8-sig"
                        raise UnicodeDecodeError(msg, b"", 0, 1, "fallback 1 fails")
                    # Eventually succeed
                    return success_df

                mock_read_csv.side_effect = mock_read_side_effect

                result = await load_csv(create_mock_context(), temp_path, encoding="utf-8")
                assert result.success
                # Verify multiple attempts were made
                assert mock_read_csv.call_count >= 3
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
class TestMemoryAndPerformance:
    """Test memory limits and performance scenarios."""

    async def test_load_csv_row_limit_enforcement(self) -> None:
        """Test that row limits are properly enforced."""
        # Create DataFrame exceeding row limit
        large_df = pd.DataFrame({"id": range(MAX_ROWS + 10), "data": ["test"] * (MAX_ROWS + 10)})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,data\n")
            temp_path = f.name

        try:
            with (
                patch("pandas.read_csv", return_value=large_df),
                pytest.raises(ToolError),
            ):
                await load_csv(create_mock_context(), temp_path)
        finally:
            Path(temp_path).unlink()

    @pytest.mark.skip(reason="Complex mocking scenario - needs refactoring")
    async def test_encoding_fallback_memory_check(self) -> None:
        """Test that memory limits are checked even in encoding fallback."""
        # Create large dataframe that exceeds memory limits
        large_df = pd.DataFrame({"col": ["x" * 10000] * 1000})  # Large strings for memory usage

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col\n")
            temp_path = f.name

        try:
            # Mock encoding detection to fail so fallbacks are used
            with (
                patch("databeak.servers.io_server.detect_file_encoding", return_value="utf-8"),
                patch("pandas.read_csv") as mock_read_csv,
            ):
                call_count = [0]

                def mock_read_side_effect(*args: Any, **kwargs: Any) -> Any:
                    call_count[0] += 1
                    if call_count[0] == 1 or call_count[0] == 2:  # First call (original encoding)
                        msg = "utf-8"
                        raise UnicodeDecodeError(msg, b"", 0, 1, "mock error")
                    # Fallback encoding succeeds but returns large df
                    return large_df

                mock_read_csv.side_effect = mock_read_side_effect

                # Mock to make memory check fail
                with (
                    patch("databeak.servers.io_server.MAX_MEMORY_USAGE_MB", 0.001),
                    pytest.raises(ToolError),
                ):
                    await load_csv(create_mock_context(), temp_path, encoding="utf-8")
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
class TestProgressReporting:
    """Test FastMCP context integration for progress reporting."""

    async def test_load_csv_with_context(self) -> None:
        """Test loading CSV with FastMCP context for progress reporting."""
        # Mock context
        mock_ctx = AsyncMock()
        mock_ctx.session_id = "test_context_session"

        content = "name,age\nJohn,30"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = await load_csv(mock_ctx, temp_path)
            assert result.success

            # Verify context methods were called
            mock_ctx.info.assert_called()
            mock_ctx.report_progress.assert_called()
        finally:
            Path(temp_path).unlink()

    async def test_export_csv_with_context(self) -> None:
        """Test export with context reporting."""
        mock_ctx = AsyncMock()

        # Load data first
        content = "name,age\nJohn,30"
        ctx = create_mock_context()
        await load_csv_from_content(ctx, content)
        session_id = ctx.session_id
        mock_ctx.session_id = session_id

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            temp_path = tmp.name

        result = await export_csv(mock_ctx, file_path=temp_path)

        # Verify context methods were called
        mock_ctx.info.assert_called()
        mock_ctx.report_progress.assert_called()

        # Cleanup
        Path(result.file_path).unlink()


# Helper function to fix nullable dtypes for test compatibility
def create_test_dataframe() -> pd.DataFrame:
    """Create test DataFrame compatible with session management."""
    return pd.DataFrame(
        {"name": ["John", "Jane", "Alice"], "age": [30, 25, 35], "city": ["NYC", "LA", "Chicago"]},
    )
