"""Unit tests for statistics service."""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from databeak.services.statistics_service import StatisticsService


class TestStatisticsService:
    """Unit tests for StatisticsService class."""

    @pytest.fixture
    def service(self) -> StatisticsService:
        """Create service instance."""
        mock_session_manager = Mock()
        return StatisticsService(session_manager=mock_session_manager)

    @pytest.fixture
    def mock_session_and_data(self) -> tuple[MagicMock, pd.DataFrame]:
        """Mock session and data for testing."""
        mock_session = MagicMock()
        mock_df = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4, 5],
                "text_col": ["a", "b", "c", "d", "e"],
                "nullable_numeric": [1.0, 2.0, None, 4.0, 5.0],
            },
        )
        return mock_session, mock_df

    def test_get_service_name(self, service: StatisticsService) -> None:
        """Test service name."""
        assert service.get_service_name() == "StatisticsService"

    async def test_get_statistics_all_columns(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting statistics for all numeric columns."""
        mock_session, mock_df = mock_session_and_data

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session")

            assert result.column_count == 2  # Only numeric columns
            assert "numeric_col" in result.statistics
            assert "nullable_numeric" in result.statistics
            assert "text_col" not in result.statistics
            assert result.total_rows == 5
            assert set(result.numeric_columns) == {"numeric_col", "nullable_numeric"}

    async def test_get_statistics_specific_columns(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting statistics for specific columns."""
        mock_session, mock_df = mock_session_and_data

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session", columns=["numeric_col"])

            assert result.column_count == 1
            assert "numeric_col" in result.statistics
            assert "nullable_numeric" not in result.statistics

    async def test_get_statistics_no_numeric_columns(self, service: StatisticsService) -> None:
        """Test statistics with no numeric columns."""
        mock_session = MagicMock()
        mock_df = pd.DataFrame({"text_col1": ["a", "b", "c"], "text_col2": ["x", "y", "z"]})

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session")

            assert result.column_count == 0
            assert result.statistics == {}
            assert result.numeric_columns == []
            assert result.total_rows == 3

    async def test_get_statistics_invalid_columns(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test statistics with invalid column names."""
        mock_session, mock_df = mock_session_and_data

        with (
            patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)),
            patch.object(
                service,
                "validate_columns_exist",
                side_effect=ToolError("Column not found"),
            ),
            pytest.raises(ToolError, match="Column not found"),
        ):
            await service.get_statistics("test_session", columns=["invalid_col"])

    async def test_get_column_statistics_success(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting statistics for a single column."""
        mock_session, mock_df = mock_session_and_data

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_column_statistics("test_session", "numeric_col")

            assert result.column == "numeric_col"
            assert result.statistics.count == 5
            assert result.statistics.mean == 3.0
            assert result.statistics.std is not None
            assert result.statistics.std > 0

    async def test_get_column_statistics_non_numeric(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting statistics for non-numeric column."""
        mock_session, mock_df = mock_session_and_data

        with (
            patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)),
            patch.object(service, "validate_columns_exist"),
        ):
            result = await service.get_column_statistics("test_session", "text_col")
            assert result.column == "text_col"
            assert result.statistics.count > 0  # Should have count of non-null values
            assert result.statistics.mean is None  # No mean for non-numeric
            assert result.statistics.std is None  # No std for non-numeric

    async def test_get_value_counts_success(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting value counts."""
        mock_session, mock_df = mock_session_and_data

        with (
            patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)),
            patch.object(service, "validate_columns_exist"),
        ):
            result = await service.get_value_counts("test_session", "text_col")

            assert result.column == "text_col"
            assert result.total_values == 5
            assert len(result.value_counts) > 0

    async def test_get_value_counts_with_limit(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting value counts with limit."""
        mock_session, mock_df = mock_session_and_data

        with (
            patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)),
            patch.object(service, "validate_columns_exist"),
        ):
            result = await service.get_value_counts("test_session", "text_col", top_n=2)

            assert len(result.value_counts) <= 2

    async def test_get_correlation_matrix_success(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test getting correlation matrix."""
        mock_session, mock_df = mock_session_and_data

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_correlation_matrix("test_session")

            assert len(result.columns_analyzed) == 2  # Only numeric columns
            assert len(result.correlation_matrix) == 2
            # Check first column exists and has the right number of correlations
            first_col = result.columns_analyzed[0]
            assert len(result.correlation_matrix[first_col]) == 2

    async def test_get_correlation_matrix_specific_columns(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test correlation matrix with specific columns."""
        mock_session, mock_df = mock_session_and_data

        with (
            patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)),
            patch.object(service, "validate_columns_exist"),
        ):
            result = await service.get_correlation_matrix(
                "test_session",
                columns=["numeric_col", "nullable_numeric"],
            )

            assert len(result.columns_analyzed) == 2
            assert len(result.correlation_matrix) == 2

    async def test_get_correlation_matrix_insufficient_columns(
        self, service: StatisticsService
    ) -> None:
        """Test correlation matrix with insufficient numeric columns."""
        mock_session = MagicMock()
        mock_df = pd.DataFrame({"text_col": ["a", "b", "c"]})

        with (
            patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)),
            pytest.raises(ToolError, match="No numeric columns found"),
        ):
            await service.get_correlation_matrix("test_session")

    async def test_error_handling_empty_dataframe(self, service: StatisticsService) -> None:
        """Test error handling with empty DataFrame."""
        mock_session = MagicMock()
        mock_df = pd.DataFrame()

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session")

            assert result.column_count == 0
            assert result.total_rows == 0

    async def test_statistics_with_all_null_column(self, service: StatisticsService) -> None:
        """Test statistics calculation with all-null numeric column."""
        mock_session = MagicMock()
        mock_df = pd.DataFrame({"all_null": [None, None, None], "valid_numeric": [1, 2, 3]})

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session")

            # Should handle null columns gracefully
            assert result.column_count >= 1
            assert "valid_numeric" in result.statistics

    async def test_percentiles_calculation(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test percentiles calculation in statistics."""
        mock_session, mock_df = mock_session_and_data

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session", include_percentiles=True)

            for col_stats in result.statistics.values():
                # Should include percentiles as individual fields
                assert hasattr(col_stats, "percentile_25")
                assert hasattr(col_stats, "percentile_50")
                assert hasattr(col_stats, "percentile_75")
                # Values should be set for numeric columns
                if col_stats.mean is not None:  # It's a numeric column
                    assert col_stats.percentile_25 is not None
                    assert col_stats.percentile_50 is not None
                    assert col_stats.percentile_75 is not None

    async def test_no_percentiles_calculation(
        self, service: StatisticsService, mock_session_and_data: tuple[MagicMock, pd.DataFrame]
    ) -> None:
        """Test statistics without percentiles."""
        mock_session, mock_df = mock_session_and_data

        with patch.object(service, "get_session_and_data", return_value=(mock_session, mock_df)):
            result = await service.get_statistics("test_session", include_percentiles=False)

            # Should still calculate basic statistics
            assert result.column_count > 0
            assert len(result.statistics) > 0
