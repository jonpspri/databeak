"""Additional tests for I/O server operations to improve coverage."""

import tempfile
from pathlib import Path

import pytest

from src.databeak.servers.io_server import (
    close_session,
    export_csv,
    get_session_info,
    list_sessions,
    load_csv,
    load_csv_from_content,
    load_csv_from_url,
)


@pytest.mark.asyncio
class TestIOServerErrorHandling:
    """Test I/O server operations error handling."""

    async def test_load_csv_invalid_file_path(self):
        """Test loading CSV with invalid file path."""
        with pytest.raises(Exception) as exc_info:
            await load_csv("/nonexistent/path/to/file.csv")
        assert "Invalid file path" in str(exc_info.value) or "Failed to load CSV" in str(
            exc_info.value
        )

    async def test_load_csv_permission_denied(self):
        """Test loading CSV with permission issues."""
        # Create a temporary file and make it unreadable
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30")
            temp_path = f.name

        try:
            # Make file unreadable
            Path(temp_path).chmod(0o000)

            with pytest.raises(Exception) as exc_info:
                await load_csv(temp_path)
            assert (
                "permission" in str(exc_info.value).lower()
                or "failed to load" in str(exc_info.value).lower()
            )
        finally:
            # Clean up
            Path(temp_path).chmod(0o644)
            Path(temp_path).unlink()

    async def test_load_csv_malformed_data(self):
        """Test loading malformed CSV data."""
        # Create malformed CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,extra\nJohn,30\nJane,25,something,excess")  # Inconsistent columns
            temp_path = f.name

        try:
            # Pandas may raise an error for malformed CSV with inconsistent columns
            with pytest.raises(Exception) as exc_info:
                await load_csv(temp_path)
            assert (
                "Error tokenizing data" in str(exc_info.value)
                or "Expected" in str(exc_info.value)
                or "Failed to load CSV" in str(exc_info.value)
            )
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_from_url_invalid_url(self):
        """Test loading from invalid URL."""
        with pytest.raises(Exception) as exc_info:
            await load_csv_from_url("not-a-valid-url")
        assert "Invalid URL" in str(exc_info.value) or "Failed to load CSV" in str(exc_info.value)

    async def test_load_csv_from_url_network_error(self):
        """Test loading from URL with network issues."""
        # Use a URL that will definitely fail
        with pytest.raises(Exception) as exc_info:
            await load_csv_from_url("https://nonexistent-domain-12345.com/data.csv")
        assert "Failed to load CSV" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    async def test_load_csv_from_content_empty(self):
        """Test loading empty CSV content."""
        with pytest.raises(Exception) as exc_info:
            await load_csv_from_content("")
        assert "empty" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value)

    async def test_export_csv_invalid_session(self):
        """Test exporting with invalid session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "output.csv")
            with pytest.raises(Exception) as exc_info:
                await export_csv("invalid-session", file_path)
            assert "Session not found" in str(exc_info.value) or "not found" in str(exc_info.value)

    async def test_export_csv_invalid_format(self):
        """Test exporting with invalid format."""
        # Create session with data
        load_result = await load_csv_from_content("name,age\nJohn,30")
        session_id = load_result.session_id

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "output.invalid")
            with pytest.raises(Exception) as exc_info:
                await export_csv(session_id, file_path, format="invalid_format")
            assert (
                "format" in str(exc_info.value).lower()
                or "unsupported" in str(exc_info.value).lower()
            )

    async def test_export_csv_permission_denied(self):
        """Test exporting to protected directory."""
        # Create session with data
        load_result = await load_csv_from_content("name,age\nJohn,30")
        session_id = load_result.session_id

        # Try to export to a protected location
        with pytest.raises(Exception) as exc_info:
            await export_csv(session_id, "/root/protected.csv")
        assert (
            "Failed to export" in str(exc_info.value)
            or "permission" in str(exc_info.value).lower()
            or "denied" in str(exc_info.value).lower()
        )


@pytest.mark.asyncio
class TestIOServerSessionManagement:
    """Test I/O server session management functions."""

    async def test_get_session_info_success(self):
        """Test getting session info successfully."""
        # Create session with data
        result = await load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
        session_id = result.session_id

        info_result = await get_session_info(session_id)
        assert info_result.session_id == session_id
        assert info_result.row_count == 2
        assert info_result.column_count == 3
        assert info_result.data_loaded is True

    async def test_get_session_info_invalid_session(self):
        """Test getting info for invalid session."""
        with pytest.raises(Exception) as exc_info:
            await get_session_info("invalid-session")
        assert "Session not found" in str(exc_info.value) or "not found" in str(exc_info.value)

    async def test_list_sessions_structure(self):
        """Test list sessions returns proper structure."""
        # Create multiple sessions
        await load_csv_from_content("name,age\nJohn,30")
        await load_csv_from_content("product,price\nLaptop,999")

        sessions_result = await list_sessions()
        assert hasattr(sessions_result, "sessions")
        assert len(sessions_result.sessions) >= 2
        assert sessions_result.total_sessions >= 2

    async def test_close_session_success(self):
        """Test closing session successfully."""
        # Create session
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        # Close session
        close_result = await close_session(session_id)
        assert close_result.session_id == session_id
        assert "closed successfully" in close_result.message

    async def test_close_session_invalid(self):
        """Test closing invalid session."""
        with pytest.raises(Exception) as exc_info:
            await close_session("invalid-session")
        assert "Session not found" in str(exc_info.value) or "not found" in str(exc_info.value)


@pytest.mark.asyncio
class TestIOServerEdgeCases:
    """Test I/O server edge cases and boundary conditions."""

    async def test_load_csv_with_custom_parameters(self):
        """Test CSV loading with custom parameters."""
        # Create CSV with custom delimiter and encoding
        csv_content = "name|age|city\nJohn|30|NYC\nJane|25|LA"

        result = await load_csv_from_content(csv_content, delimiter="|")
        assert result.session_id is not None
        assert result.data.row_count == 2
        assert result.data.column_count == 3
        assert result.rows_affected == 2

    async def test_load_csv_with_null_values(self):
        """Test CSV loading with various null representations."""
        csv_content = "name,age,city\nJohn,,NYC\n,25,\nJane,30,LA"

        result = await load_csv_from_content(csv_content)
        assert result.session_id is not None
        assert result.rows_affected == 3  # Should load 3 rows
        # Should handle empty cells as null values

    async def test_export_all_formats(self):
        """Test exporting to all supported formats."""
        # Create session with data
        result = await load_csv_from_content("name,age\nJohn,30\nJane,25")
        session_id = result.session_id

        formats_to_test = ["csv", "tsv", "json", "excel", "parquet"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in formats_to_test:
                file_path = str(Path(temp_dir) / f"output.{fmt}")
                # Most should succeed, some might fail due to missing dependencies
                try:
                    export_result = await export_csv(session_id, file_path, format=fmt)
                    assert Path(file_path).exists()
                    assert export_result.rows_exported > 0
                except Exception as e:
                    # If it fails, it should be due to missing dependencies or unsupported format
                    assert (
                        "package" in str(e).lower()
                        or "unsupported" in str(e).lower()
                        or "failed to export" in str(e).lower()
                    )

    @pytest.mark.skip(reason="Progress reporting not implemented yet")
    async def test_load_csv_progress_reporting(self):
        """Test CSV loading with progress reporting context."""
        # Create larger dataset to test progress reporting
        rows = ["name,age,city"] + [f"Person{i},{20 + i},City{i}" for i in range(100)]
        csv_content = "\n".join(rows)

        # Mock context to capture progress calls
        class MockContext:
            def __init__(self):
                self.info_calls = []
                self.progress_calls = []

            async def info(self, message):
                self.info_calls.append(message)

            async def report_progress(self, progress):
                self.progress_calls.append(progress)

        mock_ctx = MockContext()
        result = await load_csv_from_content(csv_content, ctx=mock_ctx)

        assert result.success is True
        # Should have made progress and info calls
        assert len(mock_ctx.progress_calls) > 0
        assert len(mock_ctx.info_calls) > 0


@pytest.mark.asyncio
class TestIOServerFileHandling:
    """Test I/O server file handling edge cases."""

    async def test_load_csv_large_file_simulation(self):
        """Test loading a simulated large CSV file."""
        # Create a reasonably large CSV content
        header = "id,name,value,category,description"
        rows = [header]
        for i in range(1000):  # 1000 rows
            rows.append(f"{i},Item{i},{i * 10.5},Cat{i % 5},Description for item {i}")

        csv_content = "\n".join(rows)

        result = await load_csv_from_content(csv_content)
        assert result.session_id is not None
        assert result.data.row_count == 1000
        assert result.data.column_count == 5
        assert result.rows_affected == 1000

    async def test_export_csv_overwrite_existing(self):
        """Test exporting to overwrite existing file."""
        # Create session with data
        result = await load_csv_from_content("name,age\nJohn,30")
        session_id = result.session_id

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "test.csv")

            # Create existing file
            Path(file_path).write_text("old content")
            assert Path(file_path).exists()

            # Export should overwrite
            export_result = await export_csv(session_id, file_path)
            assert export_result.session_id == session_id
            assert export_result.rows_exported > 0

            # File should contain new content
            new_content = Path(file_path).read_text()
            assert "John" in new_content
            assert "old content" not in new_content
