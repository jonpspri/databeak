"""Integration tests for DataBeak CSV MCP Server.

Tests the core functionality across multiple tool domains using proper TestCase structure with
isolated sessions for each test.
"""

import unittest
from pathlib import Path
from typing import Any

from src.databeak.servers.discovery_server import (
    detect_outliers,
    group_by_aggregate,
    profile_data,
)
from src.databeak.servers.io_server import (
    close_session,
    export_csv,
    get_session_info,
    list_sessions,
    load_csv_from_content,
)
from src.databeak.servers.statistics_server import (
    get_correlation_matrix,
    get_statistics,
)
from src.databeak.servers.validation_server import (
    ValidationSchema,
    check_data_quality,
    find_anomalies,
    validate_schema,
)
from src.databeak.tools.transformations import (
    add_column,
    fill_missing_values,
    filter_rows,
    sort_data,
)

# Test data
TEST_CSV_CONTENT = """name,age,salary,department,hire_date
Alice,28,55000,Engineering,2021-01-15
Bob,35,65000,Engineering,2019-06-01
Charlie,42,75000,Management,2018-03-20
Diana,31,58000,Marketing,2020-08-10
Eve,29,52000,Sales,2021-03-25
Frank,45,85000,Management,2017-11-30
Grace,26,48000,Marketing,2022-02-14
Henry,38,70000,Engineering,2019-09-15
Iris,33,62000,Sales,2020-05-20
Jack,41,78000,Management,2018-07-12
Kate,27,,Marketing,2021-11-01
Leo,36,68000,Engineering,2019-04-18
Mia,30,56000,Sales,2020-12-05
Nathan,44,82000,Management,2017-09-08
Olivia,25,45000,Marketing,2022-06-30
Peter,39,72000,Engineering,2018-10-22
Quinn,32,60000,Sales,2020-03-15
Rachel,28,54000,Marketing,2021-07-20
Sam,37,69000,Engineering,2019-02-28
Tina,34,64000,Sales,2020-01-10
"""


def get_attr(obj: Any, attr: str, default: Any | None = None):
    """Get attribute from object, works with both dict and Pydantic models."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)


class IntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    """Base test case for integration tests with session management.

    Provides a fresh session with test data for each test method. Handles session creation in setUp
    and cleanup in tearDown.
    """

    def setUp(self):
        """Create a fresh session with test data for each test."""
        self.session_id = None

    async def asyncSetUp(self):
        """Async setup - create session with test data."""
        result = await load_csv_from_content(content=TEST_CSV_CONTENT, delimiter=",")
        self.session_id = get_attr(result, "session_id")
        self.assertIsNotNone(self.session_id, "Failed to create test session")

    async def asyncTearDown(self):
        """Async cleanup - close the test session."""
        if self.session_id:
            try:
                await close_session(self.session_id)
            except Exception:
                # Ignore errors during cleanup
                pass


class TestIOOperations(IntegrationTestCase):
    """Test I/O operations with isolated session."""

    async def test_load_and_session_info(self):
        """Test CSV loading and session info retrieval."""
        # Session already created in setUp
        info = await get_session_info(session_id=self.session_id)
        self.assertTrue(get_attr(info, "success"))
        self.assertTrue(get_attr(info, "data_loaded"))

    async def test_list_sessions(self):
        """Test session listing functionality."""
        sessions = await list_sessions()
        self.assertTrue(get_attr(sessions, "success"))
        session_list = get_attr(sessions, "sessions", [])
        self.assertGreater(len(session_list), 0)


class TestTransformations(IntegrationTestCase):
    """Test data transformation operations with isolated session."""

    async def test_filter_rows(self):
        """Test row filtering functionality."""
        result = await filter_rows(
            session_id=self.session_id,
            conditions=[
                {"column": "salary", "operator": ">", "value": 60000},
                {
                    "column": "department",
                    "operator": "in",
                    "value": ["Engineering", "Management"],
                },
            ],
            mode="and",
        )
        self.assertTrue(get_attr(result, "success"))
        rows_before = get_attr(result, "rows_before")
        rows_after = get_attr(result, "rows_after")
        self.assertLess(rows_after, rows_before)

    async def test_sort_data(self):
        """Test data sorting functionality."""
        result = await sort_data(
            session_id=self.session_id,
            columns=[
                {"column": "department", "ascending": True},
                {"column": "salary", "ascending": False},
            ],
        )
        self.assertTrue(get_attr(result, "success"))

    async def test_add_column(self):
        """Test adding calculated columns."""
        result = await add_column(
            session_id=self.session_id,
            name="salary_level",
            value="Medium",  # Use a simple value instead of complex formula
        )
        self.assertTrue(get_attr(result, "success"))

    async def test_fill_missing_values(self):
        """Test missing value imputation."""
        result = await fill_missing_values(
            session_id=self.session_id, strategy="mean", columns=["salary"]
        )
        self.assertTrue(get_attr(result, "success"))


class TestAnalytics(IntegrationTestCase):
    """Test analytics operations with isolated session."""

    async def test_get_statistics(self):
        """Test statistical analysis."""
        result = await get_statistics(session_id=self.session_id, columns=["salary"])
        self.assertTrue(get_attr(result, "success"))
        statistics = get_attr(result, "statistics", {})
        self.assertIn("salary", statistics)

    async def test_correlation_matrix(self):
        """Test correlation analysis."""
        result = await get_correlation_matrix(
            session_id=self.session_id, method="pearson", min_correlation=0.1
        )
        self.assertTrue(get_attr(result, "success"))

    async def test_group_by_aggregate(self):
        """Test grouping and aggregation."""
        result = await group_by_aggregate(
            session_id=self.session_id,
            group_by=["department"],
            aggregations={"salary": ["mean", "min", "max", "count"]},
        )
        self.assertTrue(get_attr(result, "success"))

    async def test_detect_outliers(self):
        """Test outlier detection."""
        result = await detect_outliers(
            session_id=self.session_id, columns=["salary"], method="iqr", threshold=1.5
        )
        self.assertTrue(get_attr(result, "success"))

    async def test_profile_data(self):
        """Test comprehensive data profiling."""
        result = await profile_data(
            session_id=self.session_id, include_correlations=True, include_outliers=True
        )
        self.assertTrue(get_attr(result, "success"))


class TestValidation(IntegrationTestCase):
    """Test validation operations with isolated session."""

    async def test_validate_schema(self):
        """Test schema validation with new validation server."""
        schema_dict = {
            "name": {"type": "str", "nullable": False},
            "age": {"type": "int", "min": 0, "max": 200},
            "salary": {"type": "int", "min": 0, "max": 200000},
            "department": {
                "type": "str",
                "values": ["Engineering", "Management", "Marketing", "Sales"],
            },
        }

        result = validate_schema(session_id=self.session_id, schema=ValidationSchema(schema_dict))
        self.assertIsInstance(result.valid, bool)
        self.assertIsInstance(result.errors, list)

    async def test_check_data_quality(self):
        """Test data quality checking."""
        result = check_data_quality(session_id=self.session_id)
        quality_results = result.quality_results
        self.assertIsInstance(quality_results.overall_score, float)
        self.assertGreaterEqual(quality_results.overall_score, 0)
        self.assertLessEqual(quality_results.overall_score, 100)

    async def test_find_anomalies(self):
        """Test anomaly detection."""
        result = find_anomalies(session_id=self.session_id, columns=["salary"])
        summary = result.anomalies.summary
        self.assertIsInstance(summary.total_anomalies, int)
        self.assertGreaterEqual(summary.total_anomalies, 0)


class TestExport(IntegrationTestCase):
    """Test export operations with isolated session."""

    async def test_export_csv(self):
        """Test CSV export functionality."""
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)

        try:
            output_file = output_dir / "test_export.csv"
            result = await export_csv(
                session_id=self.session_id, file_path=str(output_file), format="csv"
            )
            self.assertTrue(get_attr(result, "success"))
            self.assertTrue(output_file.exists())
        finally:
            # Clean up
            if output_file.exists():
                output_file.unlink()
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
