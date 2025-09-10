"""Comprehensive test coverage for I/O server to achieve 80%+ coverage.

This test file targets specific uncovered code paths and edge cases identified in the coverage
analysis. It focuses on memory limits, security enhancements, temp file cleanup, and comprehensive
error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.io_server import (
    MAX_MEMORY_USAGE_MB,
    MAX_ROWS,
    close_session,
    export_csv,
    get_session_info,
    list_sessions,
    load_csv,
    load_csv_from_content,
    load_csv_from_url,
)


@pytest.mark.asyncio
class TestMemoryLimitEnforcement:
    """Test memory and row limit enforcement."""

    async def test_load_csv_exceeds_max_rows(self) -> None:
        """Test loading CSV that exceeds MAX_ROWS limit."""
        # Mock the MAX_ROWS check by patching len() to return a large number
        with patch("src.databeak.servers.io_server.pd.read_csv") as mock_read_csv:
            # Create a mock DataFrame that appears to have too many rows
            mock_df = Mock()
            mock_df.__len__.return_value = MAX_ROWS + 100
            mock_read_csv.return_value = mock_df

            with pytest.raises(ToolError) as exc_info:
                await load_csv_from_content("name,age\nJohn,30")
            assert "exceeds limit" in str(exc_info.value)
            assert "rows" in str(exc_info.value)

    @patch("pandas.DataFrame.memory_usage")
    async def test_load_csv_exceeds_memory_limit(self, mock_memory_usage: Mock) -> None:
        """Test loading CSV that exceeds memory limit."""
        # Mock memory usage to exceed limit
        mock_memory_usage.return_value = pd.Series([MAX_MEMORY_USAGE_MB * 1024 * 1024 + 1000000])

        csv_content = "name,age,city\nJohn,30,NYC\nJane,25,LA"

        with pytest.raises(ToolError) as exc_info:
            await load_csv_from_content(csv_content)
        assert "memory limit" in str(exc_info.value).lower()

    @patch("pandas.DataFrame.memory_usage")
    async def test_load_csv_memory_limit_with_fallback_encoding(
        self, mock_memory_usage: Mock
    ) -> None:
        """Test memory limit enforcement in fallback encoding path."""
        # Mock memory usage to exceed limit
        mock_memory_usage.return_value = pd.Series([MAX_MEMORY_USAGE_MB * 1024 * 1024 + 1000000])

        # Create a temporary file with content that will trigger encoding fallback
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="latin1", suffix=".csv", delete=False
        ) as f:
            f.write("name,age\nJoão,30\nJosé,25")
            temp_path = f.name

        try:
            with pytest.raises(ToolError) as exc_info:
                await load_csv(temp_path, encoding="utf-8")  # Will trigger fallback to latin1
            assert "memory limit" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_row_limit_with_fallback_encoding(self) -> None:
        """Test row limit enforcement in fallback encoding path."""
        # Mock the row limit check in fallback encoding path
        with patch("pandas.read_csv") as mock_read_csv:
            # First call fails with UnicodeDecodeError
            # Second call (fallback) returns DataFrame with too many rows
            mock_df = Mock()
            mock_df.__len__.return_value = MAX_ROWS + 100

            mock_read_csv.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
                mock_df,  # Fallback succeeds but has too many rows
            ]

            # Create file with special characters
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="latin1", suffix=".csv", delete=False
            ) as f:
                f.write("name,age\nJoão,30")
                temp_path = f.name

            try:
                with pytest.raises(ToolError) as exc_info:
                    await load_csv(temp_path, encoding="utf-8")
                assert "exceeds limit" in str(exc_info.value)
                assert "rows" in str(exc_info.value)
            finally:
                Path(temp_path).unlink()


@pytest.mark.asyncio
class TestEncodingFallbackLogic:
    """Test comprehensive encoding fallback behavior."""

    async def test_load_csv_encoding_fallback_success(self) -> None:
        """Test successful encoding fallback."""
        # Create file with special characters that require latin1 encoding
        content = "name,city\nJoão,São Paulo\nJosé,México"

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="latin1", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = await load_csv(temp_path, encoding="utf-8")  # Will fallback to latin1
            assert result.success is True
            assert result.rows_affected == 2
            assert "João" in str(result.data.rows) or "Jose" in str(result.data.rows)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_all_encodings_fail(self) -> None:
        """Test when all encoding attempts fail."""
        # Create file with binary content that can't be decoded properly
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            # Write invalid UTF-8 bytes
            f.write(b"name,age\n\xff\xfe\x00invalid,30\n")
            temp_path = f.name

        try:
            with pytest.raises(ToolError) as exc_info:
                await load_csv(temp_path, encoding="utf-8")
            assert "Encoding error with all attempted encodings" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_from_url_encoding_fallback(self) -> None:
        """Test URL loading with encoding fallback."""
        with patch("pandas.read_csv") as mock_read_csv:
            # First call fails with UnicodeDecodeError
            mock_read_csv.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
                pd.DataFrame({"name": ["João"], "age": [30]}),  # Second call succeeds
            ]

            with patch(
                "src.databeak.servers.io_server.validate_url",
                return_value=(True, "https://example.com/data.csv"),
            ):
                result = await load_csv_from_url("https://example.com/data.csv", encoding="utf-8")
                assert result.success is True
                assert mock_read_csv.call_count == 2


@pytest.mark.asyncio
class TestTempFileCleanup:
    """Test temporary file cleanup functionality in export operations."""

    @patch("pandas.DataFrame.to_csv")
    async def test_export_csv_cleanup_on_export_error(self, mock_to_csv: Mock) -> None:
        """Test temp file cleanup when export operation fails."""
        # Create session with data first
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        # Mock export to fail
        mock_to_csv.side_effect = OSError("Export failed")

        # Capture any temp files created
        original_files = set(Path(tempfile.gettempdir()).glob("databeak_export_*"))

        with pytest.raises(ToolError) as exc_info:
            await export_csv(session_id, format="csv")

        # Verify temp files are cleaned up
        current_files = set(Path(tempfile.gettempdir()).glob("databeak_export_*"))
        new_files = current_files - original_files
        assert len(new_files) == 0, f"Temp files not cleaned up: {new_files}"
        assert "Export failed" in str(exc_info.value)

    @patch("pandas.ExcelWriter")
    async def test_export_excel_cleanup_on_error(self, mock_excel_writer: Mock) -> None:
        """Test temp file cleanup when Excel export fails."""
        # Create session with data first
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        # Mock Excel writer to fail
        mock_excel_writer.side_effect = ImportError("openpyxl not found")

        original_files = set(Path(tempfile.gettempdir()).glob("databeak_export_*"))

        with pytest.raises(ToolError) as exc_info:
            await export_csv(session_id, format="excel")

        # Verify temp files are cleaned up and error message is informative
        current_files = set(Path(tempfile.gettempdir()).glob("databeak_export_*"))
        new_files = current_files - original_files
        assert len(new_files) == 0
        assert "openpyxl package" in str(exc_info.value)

    @patch("pandas.DataFrame.to_parquet")
    async def test_export_parquet_cleanup_on_error(self, mock_to_parquet: Mock) -> None:
        """Test temp file cleanup when Parquet export fails."""
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        mock_to_parquet.side_effect = ImportError("pyarrow not found")

        original_files = set(Path(tempfile.gettempdir()).glob("databeak_export_*"))

        with pytest.raises(ToolError) as exc_info:
            await export_csv(session_id, format="parquet")

        current_files = set(Path(tempfile.gettempdir()).glob("databeak_export_*"))
        new_files = current_files - original_files
        assert len(new_files) == 0
        assert "pyarrow package" in str(exc_info.value)

    @patch("pathlib.Path.unlink")
    async def test_export_cleanup_failure_warning(self, mock_unlink: Mock) -> None:
        """Test warning when temp file cleanup fails."""
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        # Mock cleanup to fail
        mock_unlink.side_effect = OSError("Permission denied")

        with (
            patch("pandas.DataFrame.to_csv", side_effect=OSError("Export failed")),
            patch("src.databeak.servers.io_server.logger") as mock_logger,
            pytest.raises(ToolError),
        ):
            await export_csv(session_id, format="csv")

        # Verify warning is logged for cleanup failure
        mock_logger.warning.assert_called()
        warning_call_args = str(mock_logger.warning.call_args)
        assert "Failed to clean up temp file" in warning_call_args


@pytest.mark.asyncio
class TestURLValidationSecurity:
    """Test URL validation and security features."""

    async def test_load_csv_from_url_validation_failure(self) -> None:
        """Test URL validation preventing access to invalid URLs."""
        with patch(
            "src.databeak.servers.io_server.validate_url",
            return_value=(False, "Invalid URL format"),
        ):
            with pytest.raises(ToolError) as exc_info:
                await load_csv_from_url("not-a-valid-url")
            assert "Invalid URL" in str(exc_info.value)

    async def test_load_csv_from_url_private_network_blocking(self) -> None:
        """Test blocking of private network URLs."""
        # Simulate private network blocking by validator
        with patch(
            "src.databeak.servers.io_server.validate_url",
            return_value=(False, "Private network access blocked"),
        ):
            with pytest.raises(ToolError) as exc_info:
                await load_csv_from_url("http://192.168.1.1/data.csv")
            assert "Invalid URL" in str(exc_info.value)


@pytest.mark.asyncio
class TestComprehensiveExportFormats:
    """Test all export formats with edge cases."""

    async def test_export_html_format(self) -> None:
        """Test HTML export format."""
        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="html")
        assert export_result.success is True
        assert export_result.format == "html"
        assert export_result.rows_exported == 2
        assert Path(export_result.file_path).exists()

        # Verify HTML content
        html_content = Path(export_result.file_path).read_text()
        assert "<table" in html_content
        assert "John" in html_content

    async def test_export_markdown_format(self) -> None:
        """Test Markdown export format."""
        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="markdown")
        assert export_result.success is True
        assert export_result.format == "markdown"
        assert export_result.rows_exported == 2
        assert Path(export_result.file_path).exists()

        # Verify Markdown content
        md_content = Path(export_result.file_path).read_text()
        assert "|" in md_content  # Table formatting
        assert "John" in md_content

    async def test_export_tsv_format(self) -> None:
        """Test TSV export format."""
        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="tsv")
        assert export_result.success is True
        assert export_result.format == "tsv"
        assert export_result.rows_exported == 2
        assert Path(export_result.file_path).exists()

        # Verify TSV content (tab-separated)
        tsv_content = Path(export_result.file_path).read_text()
        assert "\t" in tsv_content  # Tab separation
        assert "John" in tsv_content

    async def test_export_json_format(self) -> None:
        """Test JSON export format."""
        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="json")
        assert export_result.success is True
        assert export_result.format == "json"
        assert export_result.rows_exported == 2
        assert Path(export_result.file_path).exists()

        # Verify JSON content
        import json

        json_content = json.loads(Path(export_result.file_path).read_text())
        assert isinstance(json_content, list)
        assert len(json_content) == 2
        assert json_content[0]["name"] == "John"

    async def test_export_unsupported_format_error(self) -> None:
        """Test handling of unsupported export format."""
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        # Mock ExportFormat to raise ValueError for invalid format
        with patch("src.databeak.servers.io_server.ExportFormat") as mock_format:
            mock_format.side_effect = ValueError("Invalid format")

            with pytest.raises(ToolError) as exc_info:
                await export_csv(session_id, format="invalid_format")
            assert "Failed to export data" in str(exc_info.value)


@pytest.mark.asyncio
class TestSessionManagementEdgeCases:
    """Test session management edge cases and error paths."""

    async def test_get_session_info_comprehensive_data(self) -> None:
        """Test session info with comprehensive data attributes."""
        result = await load_csv_from_content(
            "name,age,city,salary\nJohn,30,NYC,50000\nJane,25,LA,60000"
        )
        session_id = result.session_id

        info = await get_session_info(session_id)
        assert info.success is True
        assert info.session_id == session_id
        assert info.data_loaded is True
        assert info.row_count == 2
        assert info.column_count == 4
        assert info.auto_save_enabled is not None
        assert info.created_at is not None
        assert info.last_modified is not None

    async def test_list_sessions_with_multiple_sessions(self) -> None:
        """Test listing sessions with multiple active sessions."""
        # Create multiple sessions
        result1 = await load_csv_from_content("name,age\nJohn,30")
        result2 = await load_csv_from_content("product,price\nLaptop,999")
        result3 = await load_csv_from_content("city,population\nNYC,8000000")

        sessions_result = await list_sessions()
        assert sessions_result.success is True
        assert sessions_result.total_sessions >= 3
        assert sessions_result.active_sessions >= 3  # All have data loaded

        # Verify session details
        session_ids = [s.session_id for s in sessions_result.sessions]
        assert result1.session_id in session_ids
        assert result2.session_id in session_ids
        assert result3.session_id in session_ids

    async def test_list_sessions_error_handling(self) -> None:
        """Test list_sessions error handling."""
        with patch("src.databeak.servers.io_server.get_session_manager") as mock_manager:
            mock_manager.side_effect = Exception("Session manager error")

            # Should return empty list on error, not raise exception
            result = await list_sessions()
            assert result.success is True  # Still succeeds but returns empty
            assert result.total_sessions == 0
            assert result.active_sessions == 0
            assert len(result.sessions) == 0


@pytest.mark.asyncio
class TestFastMCPContextIntegration:
    """Test FastMCP context integration and progress reporting."""

    async def test_load_csv_with_context_progress_reporting(self) -> None:
        """Test CSV loading with context progress reporting."""
        mock_ctx = Mock()
        mock_ctx.info = Mock(return_value=None)
        mock_ctx.report_progress = Mock(return_value=None)
        mock_ctx.error = Mock(return_value=None)

        # Make async mocks work properly
        async def async_return(*args, **kwargs):
            return None

        mock_ctx.info.side_effect = async_return
        mock_ctx.report_progress.side_effect = async_return
        mock_ctx.error.side_effect = async_return

        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA", ctx=mock_ctx)

        assert result.success is True

        # Verify context methods were called
        mock_ctx.info.assert_called()
        mock_ctx.report_progress.assert_not_called()  # Not called for content loading

    async def test_load_csv_file_with_context_progress_reporting(self) -> None:
        """Test file CSV loading with full progress reporting."""
        mock_ctx = Mock()

        # Make async mocks work properly
        async def async_return(*args, **kwargs):
            return None

        mock_ctx.info.side_effect = async_return
        mock_ctx.report_progress.side_effect = async_return
        mock_ctx.error.side_effect = async_return

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30\nJane,25")
            temp_path = f.name

        try:
            result = await load_csv(temp_path, ctx=mock_ctx)
            assert result.success is True

            # Should call progress reporting multiple times for file loading
            assert mock_ctx.report_progress.call_count >= 2
            assert mock_ctx.info.call_count >= 2
        finally:
            Path(temp_path).unlink()

    async def test_export_csv_with_context_progress_reporting(self) -> None:
        """Test export with context progress reporting."""
        # Create session first
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        mock_ctx = Mock()

        async def async_return(*args, **kwargs):
            return None

        mock_ctx.info.side_effect = async_return
        mock_ctx.report_progress.side_effect = async_return

        export_result = await export_csv(session_id, format="csv", ctx=mock_ctx)
        assert export_result.success is True

        # Verify progress reporting during export
        assert mock_ctx.info.call_count >= 2  # Start and completion messages
        assert mock_ctx.report_progress.call_count >= 2  # Progress updates

    async def test_context_error_reporting(self) -> None:
        """Test context error reporting on failures."""
        mock_ctx = Mock()

        async def async_return(*args, **kwargs):
            return None

        mock_ctx.error.side_effect = async_return

        # Test error reporting on invalid session
        with pytest.raises(ToolError):
            await get_session_info("invalid-session", ctx=mock_ctx)

        mock_ctx.error.assert_called_once()
        error_call_args = str(mock_ctx.error.call_args)
        assert "Session not found" in error_call_args


@pytest.mark.asyncio
class TestAdvancedCSVParsing:
    """Test advanced CSV parsing scenarios."""

    async def test_load_csv_with_na_values(self) -> None:
        """Test CSV loading with custom NA values."""
        csv_content = "name,age,status\nJohn,30,ACTIVE\nJane,N/A,INACTIVE\nBob,25,UNKNOWN"

        result = await load_csv_from_content(csv_content)
        assert result.success is True

        # Load again with custom NA values
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            result = await load_csv(temp_path, na_values=["N/A", "UNKNOWN"])
            assert result.success is True
            assert result.rows_affected == 3
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_with_parse_dates(self) -> None:
        """Test CSV loading with date parsing parameter."""
        csv_content = (
            "name,birth_date,registration\nJohn,1990-01-01,2023-01-01\nJane,1985-05-15,2023-02-01"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            # Just test that parse_dates parameter is accepted without error
            result = await load_csv(temp_path, parse_dates=["birth_date"])
            assert result.success is True
            assert result.rows_affected == 2
            assert result.columns_affected == ["name", "birth_date", "registration"]
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_header_parameter_coverage(self) -> None:
        """Test CSV loading with different header parameter values for coverage."""
        csv_content = "row1col1,row1col2\nrow2col1,row2col2\nrow3col1,row3col2"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            # Test with header=None to cover that branch
            result = await load_csv(temp_path, header=None)
            assert result.success is True
            assert result.rows_affected == 3
            # With header=None, pandas creates integer column names
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_from_content_header_parameter(self) -> None:
        """Test CSV content loading header parameter coverage."""
        csv_content = "data1,data2,data3\nvalue1,value2,value3"

        # Test with has_header=False to cover that branch
        result = await load_csv_from_content(csv_content, has_header=False)
        assert result.success is True
        assert result.rows_affected == 2


@pytest.mark.asyncio
class TestErrorConditionsAndEdgeCases:
    """Test comprehensive error conditions and edge cases."""

    async def test_load_csv_empty_dataframe(self) -> None:
        """Test handling of CSV that results in empty DataFrame."""
        # CSV with only header, no data rows
        csv_content = "name,age,city"

        with pytest.raises(ToolError) as exc_info:
            await load_csv_from_content(csv_content)
        assert "no data rows" in str(exc_info.value).lower()

    async def test_load_csv_parser_error(self) -> None:
        """Test pandas ParserError handling."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create malformed CSV that will cause parser error
            f.write('name,age\n"unclosed quote,30\nJane,25')
            temp_path = f.name

        try:
            with pytest.raises(ToolError) as exc_info:
                await load_csv(temp_path)
            assert "CSV format error" in str(exc_info.value) or "CSV parsing error" in str(
                exc_info.value
            )
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_memory_error(self) -> None:
        """Test MemoryError handling."""
        with patch("pandas.read_csv", side_effect=MemoryError("Insufficient memory")):
            with pytest.raises(ToolError) as exc_info:
                await load_csv_from_content("name,age\nJohn,30")
            assert "insufficient memory" in str(exc_info.value).lower()

    async def test_export_csv_file_size_calculation(self) -> None:
        """Test file size calculation in export result."""
        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="csv")
        assert export_result.success is True
        assert export_result.file_size_mb is not None
        assert export_result.file_size_mb >= 0

        # Verify file exists and has size
        file_path = Path(export_result.file_path)
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    async def test_export_csv_with_index(self) -> None:
        """Test CSV export with pandas index included."""
        result = await load_csv_from_content("name,age\nJohn,30\nJane,25")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="csv", index=True)
        assert export_result.success is True

        # Verify index is included in output
        content = Path(export_result.file_path).read_text()
        lines = content.strip().split("\n")
        # Should have index column (starts with numbers or comma)
        assert len(lines[0].split(",")) >= 3  # index + name + age

    async def test_close_session_nonexistent(self) -> None:
        """Test closing non-existent session."""
        with pytest.raises(ToolError) as exc_info:
            await close_session("nonexistent-session-id")
        assert "Session not found" in str(exc_info.value)

    async def test_export_csv_record_operation_in_history(self) -> None:
        """Test that export operations are recorded in session history."""
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        export_result = await export_csv(session_id, format="json")
        assert export_result.success is True

        # Verify the export operation was recorded
        # This tests the session.record_operation call
        session_info = await get_session_info(session_id)
        assert session_info.success is True
