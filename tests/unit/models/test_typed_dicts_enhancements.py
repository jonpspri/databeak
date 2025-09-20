"""Tests for enhanced TypedDict definitions from Issue #45 improvements."""

from src.databeak.models.typed_dicts import (
    ExportOperationDetails,
    FilterOperationDetails,
    LoadOperationDetails,
    OperationDetails,
    SessionMetadataKeys,
    StatisticsOperationDetails,
    TransformOperationDetails,
    ValidationOperationDetails,
)


class TestOperationDetailsTypes:
    """Test specific operation details TypedDict structures."""

    def test_load_operation_details(self):
        """Test LoadOperationDetails structure."""
        details: LoadOperationDetails = {
            "file_path": "/test.csv",
            "shape": (100, 5),
            "encoding": "utf-8",
            "delimiter": ",",
            "rows_loaded": 100,
        }

        assert details["file_path"] == "/test.csv"
        assert details["shape"] == (100, 5)
        assert len(details) == 5

    def test_validation_operation_details(self):
        """Test ValidationOperationDetails structure."""
        details: ValidationOperationDetails = {
            "validation_type": "schema_validation",
            "is_valid": True,
            "errors_count": 0,
            "validation_engine": "builtin",
            "rules_count": 5,
        }

        assert details["validation_type"] == "schema_validation"
        assert details["is_valid"] is True
        assert details["errors_count"] == 0

    def test_filter_operation_details(self):
        """Test FilterOperationDetails structure."""
        details: FilterOperationDetails = {
            "conditions_applied": 2,
            "rows_before": 1000,
            "rows_after": 500,
            "rows_filtered": 500,
            "filter_mode": "and",
        }

        assert details["conditions_applied"] == 2
        assert details["rows_filtered"] == 500

    def test_statistics_operation_details(self):
        """Test StatisticsOperationDetails structure."""
        details: StatisticsOperationDetails = {
            "columns_analyzed": ["age", "salary", "score"],
            "operation_type": "basic_stats",
            "method": "pearson",
            "numeric_columns": ["age", "salary"],
        }

        assert len(details["columns_analyzed"]) == 3
        assert details["operation_type"] == "basic_stats"

    def test_export_operation_details(self):
        """Test ExportOperationDetails structure."""
        details: ExportOperationDetails = {
            "file_path": "/output.csv",
            "format": "csv",
            "rows_exported": 1000,
            "columns_exported": ["id", "name", "value"],
            "encoding": "utf-8",
        }

        assert details["file_path"] == "/output.csv"
        assert len(details["columns_exported"]) == 3

    def test_transform_operation_details(self):
        """Test TransformOperationDetails structure."""
        details: TransformOperationDetails = {
            "operation_type": "filter_rows",
            "columns_affected": ["status"],
            "rows_affected": 250,
            "transformation_params": {"threshold": 0.5},
        }

        assert details["operation_type"] == "filter_rows"
        assert details["rows_affected"] == 250


class TestOperationDetailsUnion:
    """Test the OperationDetails union type."""

    def test_operation_details_union_usage(self):
        """Test that OperationDetails union accepts all operation types."""
        load_details: OperationDetails = {
            "file_path": "/test.csv",
            "shape": (100, 5),
        }

        filter_details: OperationDetails = {
            "conditions_applied": 1,
            "rows_before": 100,
            "rows_after": 50,
            "rows_filtered": 50,
        }

        validation_details: OperationDetails = {
            "validation_type": "quality_check",
            "is_valid": False,
            "errors_count": 5,
        }

        # All should be valid OperationDetails
        assert isinstance(load_details, dict)
        assert isinstance(filter_details, dict)
        assert isinstance(validation_details, dict)


class TestSessionMetadataStructures:
    """Test session metadata type structures."""

    def test_session_metadata_keys(self):
        """Test SessionMetadataKeys structure."""
        metadata: SessionMetadataKeys = {
            "created_at": "2023-01-01T00:00:00Z",
            "operations_count": 10,
            "needs_autosave": True,
            "memory_usage_mb": 25.5,
            "data_source": "file",
            "user_metadata": {"custom_field": "value", "priority": 1},
        }

        assert metadata["created_at"] == "2023-01-01T00:00:00Z"
        assert metadata["operations_count"] == 10
        assert metadata["needs_autosave"] is True
        assert metadata["user_metadata"]["priority"] == 1

    def test_session_metadata_flexibility(self):
        """Test that SessionMetadataKeys supports flexible user metadata."""
        metadata: SessionMetadataKeys = {
            "created_at": "2023-01-01T00:00:00Z",
            "operations_count": 10,
            "user_metadata": {
                "custom_field": "value",
                "priority": 1,
                "active": True,
                "score": 85.5,
            },
        }

        assert metadata["created_at"] == "2023-01-01T00:00:00Z"
        assert metadata["operations_count"] == 10
        user_data = metadata["user_metadata"]
        assert user_data["custom_field"] == "value"
        assert user_data["priority"] == 1


class TestTypeStructureUsage:
    """Test that the new type structures integrate well."""

    def test_type_annotation_works(self):
        """Test that type annotations work correctly."""
        # This test ensures that the TypedDict structures provide proper type hints

        def process_load_operation(details: LoadOperationDetails) -> str:
            """Process load operation details."""
            return f"Loaded {details.get('rows_loaded', 0)} rows"

        def process_validation_operation(details: ValidationOperationDetails) -> bool:
            """Process validation operation details."""
            return details["is_valid"]

        # Test usage
        load_details: LoadOperationDetails = {"rows_loaded": 100}
        validation_details: ValidationOperationDetails = {
            "validation_type": "schema_validation",
            "is_valid": True,
            "errors_count": 0,
        }

        result1 = process_load_operation(load_details)
        result2 = process_validation_operation(validation_details)

        assert result1 == "Loaded 100 rows"
        assert result2 is True
