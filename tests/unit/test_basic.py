"""Basic unit tests for CSV Editor."""

import pytest

from src import databeak
from src.databeak.core.session import get_session_manager
from src.databeak.utils.validators import sanitize_filename, validate_column_name, validate_url


class TestValidators:
    """Test validation utilities."""

    def test_validate_column_name(self) -> None:
        """Test column name validation."""
        # Valid names
        assert validate_column_name("age")[0]
        assert validate_column_name("first_name")[0]
        assert validate_column_name("_id")[0]

        # Invalid names
        assert not validate_column_name("123name")[0]
        assert not validate_column_name("name-with-dash")[0]
        assert not validate_column_name("")[0]

    def test_sanitize_filename(self) -> None:
        """Test filename sanitization."""
        assert sanitize_filename("test.csv") == "test.csv"
        assert sanitize_filename("test<>file.csv") == "test__file.csv"
        assert sanitize_filename("../../../etc/passwd") == "passwd"

    def test_validate_url(self) -> None:
        """Test URL validation with enhanced security."""
        # Valid URLs (public addresses)
        assert validate_url("https://example.com/data.csv")[0]
        assert validate_url("https://raw.githubusercontent.com/user/repo/data.csv")[0]

        # Invalid URLs (now includes localhost due to security enhancement)
        assert not validate_url("http://localhost:8000/file.csv")[0]  # Now blocked
        assert not validate_url("ftp://example.com/data.csv")[0]
        assert not validate_url("not-a-url")[0]

        # Additional security tests for private networks
        assert not validate_url("http://192.168.1.1/data.csv")[0]  # Private network
        assert not validate_url("http://10.0.0.1/data.csv")[0]  # Private network


@pytest.mark.asyncio
class TestSessionManager:
    """Test session management."""

    async def test_get_or_create_session(self) -> None:
        """Test session creation."""
        manager = get_session_manager()
        test_session_id = "test_session_123"
        session = manager.get_or_create_session(test_session_id)

        assert session is not None
        assert session.session_id == test_session_id
        assert manager.get_or_create_session(test_session_id) is not None

        # Cleanup
        await manager.remove_session(test_session_id)

    async def test_session_cleanup(self) -> None:
        """Test session removal."""
        manager = get_session_manager()
        test_session_id = "test_cleanup_456"
        session = manager.get_or_create_session(test_session_id)
        session_id = session.session_id

        # Session should exist
        assert manager.get_or_create_session(session_id) is not None

        # Remove session
        await manager.remove_session(session_id)

        # Session should not exist (check sessions dict directly since get_session auto-creates)
        assert session_id not in manager.sessions


@pytest.mark.asyncio
class TestDataOperations:
    """Test basic data operations."""

    @pytest.mark.asyncio
    async def test_load_csv_from_content(self, isolated_context, temp_work_dir) -> None:
        """Test loading CSV from string content with isolated session."""
        from src.databeak.servers.io_server import load_csv_from_content

        csv_content = """a,b,c
1,2,3
4,5,6"""

        result = await load_csv_from_content(isolated_context, content=csv_content, delimiter=",")

        assert result.rows_affected == 2
        assert len(result.columns_affected) == 3
        assert result.columns_affected == ["a", "b", "c"]

        # Verify session isolation - each test gets its own unique session
        assert isolated_context.session_id is not None
        assert len(isolated_context.session_id) > 0

    @pytest.mark.asyncio
    async def test_filter_rows(self, csv_session_with_data) -> None:
        """Test filtering rows with isolated session."""
        from src.databeak.servers.transformation_server import filter_rows

        # Get the isolated context and load result from fixture
        isolated_context, _load_result = csv_session_with_data

        # Test the filter operation on price > 100 (should filter to Laptop, Desk, Chair)
        filter_result = filter_rows(
            isolated_context,
            conditions=[{"column": "price", "operator": ">", "value": 100}],
            mode="and",
        )

        assert filter_result.success
        assert filter_result.rows_after < filter_result.rows_before
        # Original data has 5 rows, filter should leave 3 rows (Laptop: 999.99, Desk: 299.99, Chair: 199.99)
        assert filter_result.rows_before == 5
        assert filter_result.rows_after == 3

    @pytest.mark.asyncio
    async def test_filter_rows_category(self, isolated_context, csv_session_with_data) -> None:
        # Test a different filter condition - Electronics category on fresh data
        # Need to reload data first since previous filter modified the session
        from src.databeak.servers.io_server import load_csv_from_content
        from src.databeak.servers.transformation_server import filter_rows

        csv_content = """product,price,quantity,category
Laptop,999.99,10,Electronics
Mouse,29.99,50,Electronics
Keyboard,79.99,25,Electronics
Desk,299.99,5,Furniture
Chair,199.99,8,Furniture"""

        await load_csv_from_content(isolated_context, content=csv_content, delimiter=",")

        filter_result2 = filter_rows(
            isolated_context,
            conditions=[{"column": "category", "operator": "==", "value": "Electronics"}],
            mode="and",
        )

        assert filter_result2.success
        # Should have 3 Electronics items from fresh data (5 total, 3 Electronics)
        assert filter_result2.rows_before == 5
        assert filter_result2.rows_after == 3
