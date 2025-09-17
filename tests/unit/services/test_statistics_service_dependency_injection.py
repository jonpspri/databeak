"""Unit tests for StatisticsService dependency injection patterns.

This test module demonstrates the improved testability achieved through dependency injection,
showing how services can be tested in isolation from the global session management system.
"""

from typing import cast

import pandas as pd
import pytest

from src.databeak.models.session_service import MockSessionManager, SessionServiceFactory
from src.databeak.services.statistics_service import StatisticsService


class TestStatisticsServiceDependencyInjection:
    """Test StatisticsService with dependency injection for improved testability."""

    def setup_method(self) -> None:
        """Set up test environment with mocked dependencies."""
        # Create mock session manager instead of using global one
        self.mock_session_manager = MockSessionManager()

        # Create service factory with injected mock
        self.service_factory = SessionServiceFactory(self.mock_session_manager)

        # Create statistics service with injected dependencies
        self.statistics_service = cast(
            StatisticsService, self.service_factory.create_service(StatisticsService)
        )

        # Create test data
        self.test_data = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4, 5, 10, 15],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5, 10.5, 15.5],
                "text_col": ["A", "B", "C", "A", "B", "C", "D"],
                "mixed_col": ["text1", "text2", "text1", "text2", "text3", "text3", "text4"],
            }
        )

        # Add test data to mock session manager
        self.session_id = "test_session_001"
        self.mock_session_manager.add_test_data(self.session_id, self.test_data)

    @pytest.mark.asyncio
    async def test_get_statistics_with_dependency_injection(self) -> None:
        """Test that statistics service works with injected session manager."""
        # Act
        result = await self.statistics_service.get_statistics(self.session_id)

        # Assert
        assert result.total_rows == 7
        assert len(result.statistics) == 2  # Only numeric columns
        assert "numeric_col" in result.statistics
        assert "float_col" in result.statistics

        # Verify statistics for numeric_col
        numeric_stats = result.statistics["numeric_col"]
        assert numeric_stats.count == 7
        assert numeric_stats.mean == pytest.approx(5.71, abs=0.1)
        assert numeric_stats.min == 1
        assert numeric_stats.max == 15

    @pytest.mark.asyncio
    async def test_get_statistics_with_column_filter(self) -> None:
        """Test statistics with column filtering using injected dependencies."""
        # Act
        result = await self.statistics_service.get_statistics(
            self.session_id, columns=["numeric_col"]
        )

        # Assert
        assert len(result.statistics) == 1
        assert "numeric_col" in result.statistics
        assert "float_col" not in result.statistics

    @pytest.mark.asyncio
    async def test_get_column_statistics_with_dependency_injection(self) -> None:
        """Test column statistics with injected dependencies."""
        # Act
        result = await self.statistics_service.get_column_statistics(self.session_id, "numeric_col")

        # Assert
        assert result.column == "numeric_col"
        assert result.data_type == "int64"
        assert result.non_null_count == 7
        assert result.statistics.count == 7

    @pytest.mark.asyncio
    async def test_get_correlation_matrix_with_dependency_injection(self) -> None:
        """Test correlation matrix with injected dependencies."""
        # Act
        result = await self.statistics_service.get_correlation_matrix(self.session_id)

        # Assert
        assert result.method == "pearson"
        assert len(result.columns_analyzed) == 2
        assert "numeric_col" in result.correlation_matrix
        assert "float_col" in result.correlation_matrix

    @pytest.mark.asyncio
    async def test_get_value_counts_with_dependency_injection(self) -> None:
        """Test value counts with injected dependencies."""
        # Act
        result = await self.statistics_service.get_value_counts(self.session_id, "text_col")

        # Assert
        assert result.column == "text_col"
        assert result.total_values == 7
        assert result.unique_values == 4  # A, B, C, D
        assert "A" in result.value_counts
        assert result.value_counts["A"] == 2

    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_session(self) -> None:
        """Test error handling with dependency injection."""
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await self.statistics_service.get_statistics("invalid_session")

        assert "No data loaded in session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_column(self) -> None:
        """Test error handling for invalid column with dependency injection."""
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await self.statistics_service.get_column_statistics(
                self.session_id, "nonexistent_column"
            )

        assert "Column 'nonexistent_column' not found" in str(exc_info.value)

    def test_service_can_be_tested_in_isolation(self) -> None:
        """Demonstrate that service can be tested without global session manager."""
        # Verify that we're using our mock, not the global session manager
        assert isinstance(self.statistics_service.session_manager, MockSessionManager)
        assert self.statistics_service.session_manager is self.mock_session_manager

        # Verify service name
        assert self.statistics_service.get_service_name() == "StatisticsService"

    def test_multiple_services_with_different_dependencies(self) -> None:
        """Test creating multiple services with different dependency configurations."""
        # Create another mock session manager with different data
        other_mock = MockSessionManager()
        other_factory = SessionServiceFactory(other_mock)
        other_service = other_factory.create_service(StatisticsService)

        # Add different data to the second mock
        other_data = pd.DataFrame({"other_col": [100, 200, 300]})
        other_mock.add_test_data("other_session", other_data)

        # Verify services are independent
        assert self.statistics_service.session_manager is not other_service.session_manager
        assert len(self.mock_session_manager.sessions) == 1
        assert len(other_mock.sessions) == 1
        assert "test_session_001" in self.mock_session_manager.sessions
        assert "other_session" in other_mock.sessions

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_server_functions(self) -> None:
        """Test that server functions still work with the new architecture."""
        from src.databeak.models import get_session_manager
        from src.databeak.servers.statistics_server import get_statistics

        # This should work with the global session manager
        # Note: This test would require actual session setup in a real integration test
        # For now, we just verify the function exists and can be imported
        assert callable(get_statistics)
        assert callable(get_session_manager)


class TestMockSessionManagerBehavior:
    """Test the mock session manager behavior for testing purposes."""

    def setup_method(self) -> None:
        """Set up mock session manager."""
        self.mock = MockSessionManager()

    def test_session_creation(self) -> None:
        """Test mock session creation."""
        session = self.mock.get_or_create_session("test_id")
        assert session.session_id == "test_id"
        assert "test_id" in self.mock.sessions

    def test_auto_id_generation(self) -> None:
        """Test automatic ID generation."""
        session1 = self.mock.get_or_create_session("test_session_1")
        session2 = self.mock.get_or_create_session("test_session_2")
        assert session1.session_id != session2.session_id
        assert session1.session_id.startswith("test_session_")
        assert session2.session_id.startswith("test_session_")

    def test_data_management(self) -> None:
        """Test data management in mock."""
        test_df = pd.DataFrame({"col1": [1, 2, 3]})
        self.mock.add_test_data("test_session", test_df)

        session = self.mock.get_or_create_session("test_session")
        assert session is not None
        assert session.has_data()
        pd.testing.assert_frame_equal(session.df, test_df)

    @pytest.mark.asyncio
    async def test_session_removal(self) -> None:
        """Test session removal."""
        self.mock.get_or_create_session("temp_session")
        assert "temp_session" in self.mock.sessions

        removed = await self.mock.remove_session("temp_session")
        assert removed is True
        assert "temp_session" not in self.mock.sessions

        # Try to remove non-existent session
        removed = await self.mock.remove_session("nonexistent")
        assert removed is False

    def test_list_sessions(self) -> None:
        """Test session listing."""
        # Create test sessions with data
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5]})

        self.mock.add_test_data("session1", df1)
        self.mock.add_test_data("session2", df2)

        sessions = self.mock.list_sessions()
        assert len(sessions) == 2

        session_ids = [s.session_id for s in sessions]
        assert "session1" in session_ids
        assert "session2" in session_ids

        # Check row counts
        for session in sessions:
            if session.session_id == "session1":
                assert session.row_count == 3
            elif session.session_id == "session2":
                assert session.row_count == 2
