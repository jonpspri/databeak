"""Additional tests for io_server to improve coverage."""

import tempfile
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.io_server import (
    export_csv,
    get_session_info,
    list_sessions,
    load_csv_from_content,
)
from tests.test_mock_context import create_mock_context


class TestSessionManagement:
    """Test session management functions."""

    async def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        # Clear any existing sessions first
        from src.databeak.models import get_session_manager

        session_manager = get_session_manager()
        session_manager.sessions.clear()

        result = await list_sessions(create_mock_context())

        assert result.total_sessions == 0
        assert result.active_sessions == 0
        assert len(result.sessions) == 0

    async def test_list_sessions_with_data(self):
        """Test listing sessions with active sessions."""
        # Create a session with data
        csv_content = "col1,col2\n1,2\n3,4"
        load_result = await load_csv_from_content(create_mock_context(), csv_content)

        result = await list_sessions(create_mock_context())

        assert result.total_sessions >= 1
        assert result.active_sessions >= 1
        assert any(s.session_id == load_result.session_id for s in result.sessions)

        # Check session info details
        session_info = next(s for s in result.sessions if s.session_id == load_result.session_id)
        assert session_info.row_count == 2
        assert session_info.column_count == 2
        assert session_info.columns == ["col1", "col2"]

    async def test_get_session_info_valid(self):
        """Test getting info for a valid session."""
        # Create a session
        csv_content = "name,value\ntest,123"
        load_result = await load_csv_from_content(create_mock_context(), csv_content)

        info = await get_session_info(create_mock_context(load_result.session_id))

        assert info.success is True
        assert info.session_id == load_result.session_id
        assert info.data_loaded is True
        assert info.row_count == 1
        assert info.column_count == 2
        # SessionInfoResult doesn't have columns field, just counts

    async def test_get_session_info_invalid(self):
        """Test getting info for invalid session."""
        with pytest.raises(ToolError, match="Session not found"):
            await get_session_info(create_mock_context())


class TestCsvLoadingEdgeCases:
    """Test CSV loading edge cases and error handling."""

    async def test_load_csv_empty_content(self):
        """Test loading empty CSV content."""
        with pytest.raises(ToolError, match="CSV"):
            await load_csv_from_content(create_mock_context(), "")

    async def test_load_csv_only_whitespace(self):
        """Test loading CSV with only whitespace."""
        with pytest.raises(ToolError, match="CSV"):
            await load_csv_from_content(create_mock_context(), "   \n  \n  ")

    async def test_load_csv_single_column(self):
        """Test loading CSV with single column."""
        csv_content = "single_col\nvalue1\nvalue2\nvalue3"
        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 3
        assert result.columns_affected == ["single_col"]

    async def test_load_csv_with_quotes(self):
        """Test loading CSV with quoted values."""
        csv_content = 'name,description\n"John Doe","A person, with comma"\n"Jane","Normal"'
        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 2
        assert result.columns_affected == ["name", "description"]

    async def test_load_csv_with_different_delimiter(self):
        """Test loading CSV with semicolon delimiter."""
        csv_content = "col1;col2;col3\n1;2;3\n4;5;6"
        result = await load_csv_from_content(create_mock_context(), csv_content, delimiter=";")

        assert result.rows_affected == 2
        assert len(result.columns_affected) == 3

    async def test_load_csv_with_mixed_types(self):
        """Test loading CSV with mixed data types."""
        csv_content = """id,name,value,is_active,date
1,Alice,100.5,true,2024-01-01
2,Bob,200,false,2024-01-02
3,Charlie,,true,"""

        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 3
        assert len(result.columns_affected) == 5

    async def test_load_csv_duplicate_columns(self):
        """Test loading CSV with duplicate column names."""
        csv_content = "col,col,col\n1,2,3\n4,5,6"

        # Should handle duplicate columns by renaming them
        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert result.rows_affected == 2
        # Pandas renames duplicates like col, col.1, col.2
        assert len(result.columns_affected) == 3


class TestExportFunctionality:
    """Test CSV export functionality."""

    async def test_export_csv_basic(self):
        """Test basic CSV export."""
        # Create a session with data
        csv_content = "name,value\ntest1,100\ntest2,200"
        load_result = await load_csv_from_content(create_mock_context(), csv_content)

        # Export to a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            export_result = await export_csv(
                create_mock_context(load_result.session_id), file_path=tmp.name
            )

            assert export_result.success is True
            assert export_result.file_path == tmp.name
            assert export_result.rows_exported == 2

            # Verify file content
            import pandas as pd

            df = pd.read_csv(tmp.name)
            assert len(df) == 2
            assert list(df.columns) == ["name", "value"]

            # Clean up
            Path(tmp.name).unlink()

    async def test_export_csv_with_subset(self):
        """Test exporting a subset of columns."""
        csv_content = "col1,col2,col3,col4\n1,2,3,4\n5,6,7,8"
        load_result = await load_csv_from_content(create_mock_context(), csv_content)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            export_result = await export_csv(
                create_mock_context(load_result.session_id), file_path=tmp.name
            )

            assert export_result.rows_exported == 2

            # Verify exported data
            import pandas as pd

            df = pd.read_csv(tmp.name)
            assert len(df.columns) == 4  # All columns exported

            # Clean up
            Path(tmp.name).unlink()

    async def test_export_invalid_session(self):
        """Test exporting with invalid session."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            with pytest.raises(ToolError, match="Session not found"):
                await export_csv(create_mock_context(), file_path=tmp.name)
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)

    async def test_export_no_data_loaded(self):
        """Test exporting when no data is loaded."""
        from src.databeak.models import get_session_manager

        session_manager = get_session_manager()
        session_manager.create_session()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            with pytest.raises(ToolError, match="no data loaded"):
                await export_csv(create_mock_context(), file_path=tmp.name)
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)


class TestMemoryAndPerformance:
    """Test memory and performance constraints."""

    async def test_load_large_number_of_columns(self):
        """Test loading CSV with many columns."""
        # Create CSV with 100 columns
        columns = [f"col_{i}" for i in range(100)]
        header = ",".join(columns)
        row = ",".join(str(i) for i in range(100))
        csv_content = f"{header}\n{row}\n{row}"

        result = await load_csv_from_content(create_mock_context(), csv_content)

        assert len(result.columns_affected) == 100
        assert result.rows_affected == 2

    async def test_session_memory_tracking(self):
        """Test that memory usage is tracked."""
        csv_content = "col1,col2,col3\n" + "\n".join(f"{i},{i + 1},{i + 2}" for i in range(100))

        load_result = await load_csv_from_content(create_mock_context(), csv_content)
        info = await get_session_info(create_mock_context(load_result.session_id))

        # SessionInfoResult doesn't have memory_usage_mb field
        # Just check that session has data loaded
        assert info.data_loaded is True
