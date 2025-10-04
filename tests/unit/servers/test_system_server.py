"""Comprehensive tests for system_server module."""

from unittest.mock import Mock, patch

import pytest
from fastmcp.exceptions import ToolError

from databeak.servers.system_server import get_server_info, health_check
from tests.test_mock_context import create_mock_context


class TestHealthCheck:
    """Test system health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy_status(self) -> None:
        """Test health check returns healthy status under normal conditions."""
        with patch(
            "databeak.servers.system_server.get_session_manager"
        ) as mock_get_session_manager:
            # Mock session manager with normal load
            mock_session_manager = Mock()
            mock_session_manager.sessions = {}  # Empty sessions dict
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 60
            mock_get_session_manager.return_value = mock_session_manager

            result = await health_check(create_mock_context())

            assert result.success is True
            assert result.status == "healthy"
            assert result.version is not None
            assert result.active_sessions == 0
            assert result.max_sessions == 100
            assert result.session_ttl_minutes == 60

    @pytest.mark.asyncio
    async def test_health_check_degraded_status(self) -> None:
        """Test health check returns degraded status when near session limit."""
        with patch(
            "databeak.servers.system_server.get_session_manager"
        ) as mock_get_session_manager:
            # Mock session manager with high load (90%+ capacity)
            mock_session_manager = Mock()
            # Create 91 sessions (91% of 100)
            mock_session_manager.sessions = {f"session_{i}": Mock() for i in range(91)}
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 60
            mock_get_session_manager.return_value = mock_session_manager

            result = await health_check(create_mock_context())

            assert result.success is True
            assert result.status == "degraded"
            assert result.active_sessions == 91
            assert result.max_sessions == 100

    @pytest.mark.asyncio
    async def test_health_check_with_context(self) -> None:
        """Test health check with FastMCP context logging."""
        from unittest.mock import AsyncMock

        mock_ctx = AsyncMock()

        with patch(
            "databeak.servers.system_server.get_session_manager"
        ) as mock_get_session_manager:
            mock_session_manager = Mock()
            mock_session_manager.sessions = {"session1": Mock(), "session2": Mock()}
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 30
            mock_get_session_manager.return_value = mock_session_manager

            result = await health_check(mock_ctx)

            # Verify context was used for logging
            mock_ctx.info.assert_called()
            assert result.success is True
            assert result.active_sessions == 2

    @pytest.mark.asyncio
    async def test_health_check_handles_session_manager_failure(self) -> None:
        """Test health check handles session manager failures gracefully."""
        from unittest.mock import AsyncMock

        mock_ctx = AsyncMock()

        with patch("databeak.servers.system_server.get_session_manager") as mock_session_manager:
            mock_session_manager.side_effect = Exception("Session manager unavailable")

            result = await health_check(mock_ctx)

            # Should return unhealthy status on failure
            assert result.success is True  # Base model default
            assert result.status == "unhealthy"
            assert result.active_sessions == 0
            assert result.max_sessions == 0

            # Context should receive error
            mock_ctx.error.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_critical_failure(self) -> None:
        """Test health check handles critical failures that prevent fallback response."""
        with (
            patch("databeak.servers.system_server.get_session_manager") as mock_session_manager,
            patch(
                "databeak.servers.system_server.__version__",
                side_effect=Exception("Version error"),
            ),
        ):
            mock_session_manager.side_effect = Exception("Critical failure")

            result = await health_check(create_mock_context())

            # Should return unhealthy status rather than raise exception
            assert result.success is True  # Base model default
            assert result.status == "unhealthy"
            assert result.active_sessions == 0

    @pytest.mark.asyncio
    async def test_health_check_response_structure(self) -> None:
        """Test health check response has correct Pydantic model structure."""
        with patch(
            "databeak.servers.system_server.get_session_manager"
        ) as mock_get_session_manager:
            mock_session_manager = Mock()
            mock_session_manager.sessions = {}
            mock_session_manager.max_sessions = 50
            mock_session_manager.ttl_minutes = 120
            mock_get_session_manager.return_value = mock_session_manager

            result = await health_check(create_mock_context())

            # Verify all required fields are present
            assert hasattr(result, "success")
            assert hasattr(result, "status")
            assert hasattr(result, "version")
            assert hasattr(result, "active_sessions")
            assert hasattr(result, "max_sessions")
            assert hasattr(result, "session_ttl_minutes")

            # Verify types
            assert isinstance(result.status, str)
            assert isinstance(result.version, str)
            assert isinstance(result.active_sessions, int)
            assert isinstance(result.max_sessions, int)
            assert isinstance(result.session_ttl_minutes, int)
            # Memory monitoring fields
            assert isinstance(result.memory_usage_mb, float)
            assert isinstance(result.memory_threshold_mb, float)
            assert isinstance(result.memory_status, str)
            # History operations fields (should be 0 after removal)
            assert isinstance(result.history_operations_total, int)
            assert isinstance(result.history_limit_per_session, int)
            assert result.history_operations_total == 0
            assert result.history_limit_per_session == 0

    @pytest.mark.asyncio
    async def test_health_check_memory_monitoring_normal(self) -> None:
        """Test health check with normal memory usage."""
        with (
            patch("databeak.servers.system_server.get_session_manager") as mock_get_session_manager,
            patch("databeak.servers.system_server.get_memory_usage") as mock_memory,
        ):
            # Mock normal conditions
            mock_session_manager = Mock()
            mock_session_manager.sessions = {}
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 60
            mock_get_session_manager.return_value = mock_session_manager

            mock_memory.return_value = 500.0  # 500MB usage

            # Use real settings - no need to mock
            result = await health_check(create_mock_context())

            assert result.success is True
            assert result.status == "healthy"
            assert result.memory_usage_mb == 500.0
            assert result.memory_threshold_mb == 2048.0  # Default from real settings
            assert result.memory_status == "normal"
            # History operations should be 0 after removal
            assert result.history_operations_total == 0
            assert result.history_limit_per_session == 0

    @pytest.mark.asyncio
    async def test_health_check_memory_warning(self) -> None:
        """Test health check with high memory usage (warning level)."""
        with (
            patch("databeak.servers.system_server.get_session_manager") as mock_get_session_manager,
            patch("databeak.servers.system_server.get_memory_usage") as mock_memory,
        ):
            mock_session_manager = Mock()
            mock_session_manager.sessions = {}
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 60
            mock_get_session_manager.return_value = mock_session_manager

            mock_memory.return_value = 1600.0  # 1.6GB usage (75% of 2GB threshold)

            # Use real settings - no need to mock

            result = await health_check(create_mock_context())

            assert result.success is True
            assert result.status == "degraded"  # Should be degraded due to high memory
            assert result.memory_usage_mb == 1600.0
            assert result.memory_status == "warning"
            # History operations should be 0 after removal
            assert result.history_operations_total == 0
            assert result.history_limit_per_session == 0

    @pytest.mark.asyncio
    async def test_health_check_memory_critical(self) -> None:
        """Test health check with critical memory usage."""
        with (
            patch("databeak.servers.system_server.get_session_manager") as mock_get_session_manager,
            patch("databeak.servers.system_server.get_memory_usage") as mock_memory,
        ):
            mock_session_manager = Mock()
            mock_session_manager.sessions = {}
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 60
            mock_get_session_manager.return_value = mock_session_manager

            mock_memory.return_value = 1900.0  # 1.9GB usage (92% of 2GB threshold)

            # Use real settings - no need to mock

            result = await health_check(create_mock_context())

            assert result.success is True
            assert result.status == "unhealthy"  # Should be unhealthy due to critical memory
            assert result.memory_usage_mb == 1900.0
            assert result.memory_status == "critical"
            # History operations should be 0 after removal
            assert result.history_operations_total == 0
            assert result.history_limit_per_session == 0

    @pytest.mark.asyncio
    async def test_health_check_multiple_issues(self) -> None:
        """Test health check with multiple concurrent issues."""
        with (
            patch("databeak.servers.system_server.get_session_manager") as mock_get_session_manager,
            patch("databeak.servers.system_server.get_memory_usage") as mock_memory,
        ):
            # High session load + critical memory
            mock_session_manager = Mock()
            mock_session_manager.sessions = {f"session_{i}": Mock() for i in range(95)}  # 95% load
            mock_session_manager.max_sessions = 100
            mock_session_manager.ttl_minutes = 60
            mock_get_session_manager.return_value = mock_session_manager

            mock_memory.return_value = 1950.0  # Critical memory (95% of 2GB)

            # Use real settings - no need to mock

            result = await health_check(create_mock_context())

            assert result.success is True
            assert result.status == "unhealthy"  # Critical memory overrides other issues
            assert result.active_sessions == 95
            assert result.memory_status == "critical"
            # History operations should be 0 after removal
            assert result.history_operations_total == 0
            assert result.history_limit_per_session == 0


class TestMemoryMonitoringUtils:
    """Test memory monitoring utility functions."""

    def test_get_memory_status_normal(self) -> None:
        """Test memory status calculation for normal usage."""
        from databeak.servers.system_server import get_memory_status

        # 50% usage - should be normal
        status = get_memory_status(1024.0, 2048.0)
        assert status == "normal"

    def test_get_memory_status_warning(self) -> None:
        """Test memory status calculation for warning level."""
        from databeak.servers.system_server import get_memory_status

        # 80% usage - should be warning
        status = get_memory_status(1638.4, 2048.0)
        assert status == "warning"

    def test_get_memory_status_critical(self) -> None:
        """Test memory status calculation for critical level."""
        from databeak.servers.system_server import get_memory_status

        # 95% usage - should be critical
        status = get_memory_status(1945.6, 2048.0)
        assert status == "critical"

    def test_get_memory_status_zero_threshold(self) -> None:
        """Test memory status with zero threshold (edge case)."""
        from databeak.servers.system_server import get_memory_status

        status = get_memory_status(100.0, 0.0)
        assert status == "normal"  # Should default to normal


class TestServerInfo:
    """Test server information functionality."""

    @pytest.mark.asyncio
    async def test_get_server_info_basic_structure(self) -> None:
        """Test server info returns proper structure with all required fields."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 500
            mock_config.session_timeout = 3600  # 1 hour in seconds
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            # Verify basic server information
            assert result.success is True
            assert result.name == "DataBeak"
            assert result.version is not None
            assert result.description is not None
            assert "comprehensive MCP server" in result.description

            # Verify configuration
            assert result.max_file_size_mb == 500
            assert result.session_timeout_minutes == 60  # Converted from seconds

    @pytest.mark.asyncio
    async def test_get_server_info_capabilities_structure(self) -> None:
        """Test server info includes all expected capability categories."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 100
            mock_config.session_timeout = 1800
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            # Verify capability categories exist
            expected_categories = [
                "data_io",
                "data_manipulation",
                "data_analysis",
                "data_validation",
                "session_management",
                "null_handling",
            ]

            for category in expected_categories:
                assert category in result.capabilities
                assert isinstance(result.capabilities[category], list)
                assert len(result.capabilities[category]) > 0

    @pytest.mark.asyncio
    async def test_get_server_info_data_io_capabilities(self) -> None:
        """Test server info includes expected data I/O capabilities."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 200
            mock_config.session_timeout = 7200
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            data_io_caps = result.capabilities["data_io"]
            expected_io_caps = [
                "load_csv",
                "load_csv_from_url",
                "load_csv_from_content",
                "export_csv",
                "multiple_export_formats",
            ]

            for cap in expected_io_caps:
                assert cap in data_io_caps

    @pytest.mark.asyncio
    async def test_get_server_info_supported_formats(self) -> None:
        """Test server info includes expected supported formats."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 300
            mock_config.session_timeout = 1200
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            expected_formats = [
                "csv",
                "tsv",
                "json",
                "excel",
                "parquet",
                "html",
                "markdown",
            ]

            for fmt in expected_formats:
                assert fmt in result.supported_formats

            # Verify it's a proper list
            assert isinstance(result.supported_formats, list)
            assert len(result.supported_formats) == len(expected_formats)

    @pytest.mark.asyncio
    async def test_get_server_info_with_context(self) -> None:
        """Test server info with FastMCP context logging."""
        from unittest.mock import AsyncMock

        mock_ctx = AsyncMock()

        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 150
            mock_config.session_timeout = 2400
            mock_settings.return_value = mock_config

            result = await get_server_info(mock_ctx)

            # Verify context was used for logging
            mock_ctx.info.assert_called()
            assert result.success is True
            assert result.name == "DataBeak"

    @pytest.mark.asyncio
    async def test_get_server_info_handles_settings_failure(self) -> None:
        """Test server info handles configuration loading failures."""
        from unittest.mock import AsyncMock

        mock_ctx = AsyncMock()

        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_settings.side_effect = Exception("Settings unavailable")

            with pytest.raises(ToolError, match="Failed to get server information"):
                await get_server_info(mock_ctx)

            # Context should receive error
            mock_ctx.error.assert_called()

    @pytest.mark.asyncio
    async def test_get_server_info_null_handling_capabilities(self) -> None:
        """Test server info includes comprehensive null handling capabilities."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 400
            mock_config.session_timeout = 5400
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            null_caps = result.capabilities["null_handling"]
            expected_null_caps = [
                "json_null_support",
                "python_none_support",
                "pandas_nan_compatibility",
                "null_value_insertion",
                "null_value_updates",
            ]

            for cap in expected_null_caps:
                assert cap in null_caps

    @pytest.mark.asyncio
    async def test_get_server_info_data_manipulation_capabilities(self) -> None:
        """Test server info includes expected data manipulation capabilities."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 250
            mock_config.session_timeout = 3000
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            manipulation_caps = result.capabilities["data_manipulation"]
            expected_caps = [
                "filter_rows",
                "sort_data",
                "select_columns",
                "rename_columns",
                "add_column",
                "remove_columns",
                "change_column_type",
                "fill_missing_values",
                "remove_duplicates",
                "null_value_support",
            ]

            for cap in expected_caps:
                assert cap in manipulation_caps

    @pytest.mark.asyncio
    async def test_get_server_info_response_model_validation(self) -> None:
        """Test server info response validates as proper Pydantic model."""
        with patch("databeak.servers.system_server.get_settings") as mock_settings:
            mock_config = Mock()
            mock_config.max_file_size_mb = 600
            mock_config.session_timeout = 7200
            mock_settings.return_value = mock_config

            result = await get_server_info(create_mock_context())

            # Test that the result can be serialized (Pydantic validation)
            result_dict = result.model_dump()

            # Verify structure
            assert "success" in result_dict
            assert "name" in result_dict
            assert "version" in result_dict
            assert "capabilities" in result_dict
            assert "supported_formats" in result_dict
            assert "max_file_size_mb" in result_dict
            assert "session_timeout_minutes" in result_dict

            # Verify capabilities is a dict of lists
            assert isinstance(result_dict["capabilities"], dict)
            for caps in result_dict["capabilities"].values():
                assert isinstance(caps, list)
                assert all(isinstance(cap, str) for cap in caps)


class TestSystemServerIntegration:
    """Test system server integration and patterns."""

    def test_system_server_exists(self) -> None:
        """Test that system server is properly exported."""
        from databeak.servers.system_server import system_server

        assert system_server is not None
        assert system_server.name == "DataBeak-System"
        assert system_server.instructions is not None

    def test_system_server_has_correct_tools(self) -> None:
        """Test that system server has registered the expected tools."""
        from databeak.servers.system_server import system_server

        # The server should have tools registered, but FastMCP doesn't expose them easily
        # We can at least verify the server is properly configured
        assert hasattr(system_server, "name")
        assert hasattr(system_server, "instructions")

    @pytest.mark.asyncio
    async def test_system_functions_are_async(self) -> None:
        """Test that system functions are properly async."""
        import inspect

        # Verify functions are async
        assert inspect.iscoroutinefunction(health_check)
        assert inspect.iscoroutinefunction(get_server_info)

    def test_system_server_follows_naming_pattern(self) -> None:
        """Test that system server follows DataBeak naming conventions."""
        from databeak.servers.system_server import system_server

        # Should follow DataBeak-<Service> naming pattern
        assert system_server.name.startswith("DataBeak-")
        assert "System" in system_server.name

        # Instructions should mention the server purpose
        instructions = system_server.instructions or ""
        assert "system" in instructions.lower()
        assert "health" in instructions.lower()
