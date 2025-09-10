"""Tests for I/O server MCP tools."""

from src.databeak.servers.io_server import (
    close_session,
    export_csv,
    get_session_info,
    list_sessions,
    load_csv,
    load_csv_from_content,
    load_csv_from_url,
)


class TestIOServerTools:
    """Test I/O server tools availability."""

    def test_csv_loading_server_tools_available(self) -> None:
        """Test that CSV loading server tools are available."""
        loading_ops = [
            load_csv,
            load_csv_from_url,
            load_csv_from_content,
        ]

        for op_func in loading_ops:
            assert callable(op_func)

    def test_session_management_server_tools_available(self) -> None:
        """Test that session management server tools are available."""
        session_ops = [
            get_session_info,
            list_sessions,
            close_session,
        ]

        for op_func in session_ops:
            assert callable(op_func)

    def test_export_server_tools_available(self) -> None:
        """Test that export server tools are available."""
        assert callable(export_csv)


class TestIOServerToolSignatures:
    """Test that I/O server tools have expected signatures."""

    def test_load_csv_server_signature(self) -> None:
        """Test load_csv server function has correct signature."""
        import inspect

        sig = inspect.signature(load_csv)
        params = list(sig.parameters.keys())

        expected_params = [
            "file_path",
            "encoding",
            "delimiter",
            "session_id",
            "header",
            "na_values",
            "parse_dates",
            "ctx",
        ]
        assert all(param in params for param in expected_params)

    def test_export_csv_server_signature(self) -> None:
        """Test export_csv server function has correct signature."""
        import inspect

        sig = inspect.signature(export_csv)
        params = list(sig.parameters.keys())

        expected_params = [
            "session_id",
            "file_path",
            "format",
            "encoding",
            "index",
            "ctx",
        ]
        assert all(param in params for param in expected_params)

    def test_session_info_server_signature(self) -> None:
        """Test get_session_info server function has correct signature."""
        import inspect

        sig = inspect.signature(get_session_info)
        params = list(sig.parameters.keys())

        expected_params = ["session_id", "ctx"]
        assert all(param in params for param in expected_params)
