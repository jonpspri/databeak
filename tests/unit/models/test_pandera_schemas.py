"""Tests for Pandera schema integration."""

import pandas as pd
import pandera as pa
import pytest

from src.databeak.models.pandera_schemas import (
    DataBeakBaseSchema,
    FinancialDataSchema,
    NumericDataSchema,
    TimeSeriesSchema,
    create_pandera_schema_from_validation_rules,
    create_typed_dataframe,
    validate_dataframe_with_pandera,
)


class TestPanderaSchemaCreation:
    """Test dynamic Pandera schema creation from DataBeak validation rules."""

    def test_create_schema_with_numeric_constraints(self):
        """Test creating schema with numeric min/max constraints."""
        rules = {
            "age": {"type": "int", "min": 0, "max": 120, "nullable": False},
            "score": {"type": "float", "min": 0.0, "max": 100.0, "nullable": True},
        }

        schema_class = create_pandera_schema_from_validation_rules(rules)

        # Test valid data
        valid_data = pd.DataFrame({"age": [25, 30, 35], "score": [85.5, 92.0, 78.5]})
        validated_df = schema_class.validate(valid_data)

        assert len(validated_df) == 3
        assert list(validated_df.columns) == ["age", "score"]

    def test_create_schema_with_string_constraints(self):
        """Test creating schema with string length and pattern constraints."""
        rules = {
            "name": {"type": "str", "min_length": 2, "max_length": 50, "nullable": False},
            "email": {"type": "str", "pattern": r"^[^@]+@[^@]+\.[^@]+$", "nullable": False},
        }

        schema_class = create_pandera_schema_from_validation_rules(rules)

        # Test valid data
        valid_data = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["alice@example.com", "bob@test.org", "charlie@company.net"],
            }
        )
        validated_df = schema_class.validate(valid_data)

        assert len(validated_df) == 3

    def test_create_schema_with_allowed_values(self):
        """Test creating schema with allowed values constraint."""
        rules = {
            "status": {
                "type": "str",
                "values": ["active", "inactive", "pending"],
                "nullable": False,
            },
            "priority": {"type": "int", "values": [1, 2, 3, 4, 5], "nullable": True},
        }

        schema_class = create_pandera_schema_from_validation_rules(rules)

        # Test valid data
        valid_data = pd.DataFrame(
            {
                "status": ["active", "pending", "inactive"],
                "priority": [1, 3, 5],
            }
        )
        validated_df = schema_class.validate(valid_data)

        assert len(validated_df) == 3

    def test_create_schema_with_uniqueness(self):
        """Test creating schema with uniqueness constraints."""
        rules = {
            "id": {"type": "str", "unique": True, "nullable": False},
            "name": {"type": "str", "nullable": False},
        }

        schema_class = create_pandera_schema_from_validation_rules(rules)

        # Test valid data (unique IDs)
        valid_data = pd.DataFrame({"id": ["1", "2", "3"], "name": ["Alice", "Bob", "Charlie"]})
        validated_df = schema_class.validate(valid_data)

        assert len(validated_df) == 3


class TestDataBeakValidationIntegration:
    """Test integration between Pandera and DataBeak validation."""

    def test_validate_dataframe_with_pandera_success(self):
        """Test successful validation with Pandera."""
        df = pd.DataFrame({"age": [25, 30, 35], "name": ["Alice", "Bob", "Charlie"]})

        rules = {
            "age": {"type": "int", "min": 0, "max": 120, "nullable": False},
            "name": {"type": "str", "min_length": 1, "nullable": False},
        }

        validated_df, errors = validate_dataframe_with_pandera(df, rules)

        assert len(errors) == 0  # No validation errors
        assert len(validated_df) == 3
        assert list(validated_df.columns) == ["age", "name"]

    def test_validate_dataframe_with_pandera_violations(self):
        """Test validation with Pandera when there are violations."""
        df = pd.DataFrame(
            {
                "age": [25, -5, 150],  # -5 and 150 violate min/max constraints
                "name": ["Alice", "", "Charlie"],  # Empty string violates min_length
            }
        )

        rules = {
            "age": {"type": "int", "min": 0, "max": 120, "nullable": False},
            "name": {"type": "str", "min_length": 1, "nullable": False},
        }

        validated_df, errors = validate_dataframe_with_pandera(df, rules)

        # Should have validation errors but still return the DataFrame
        assert len(errors) > 0
        assert len(validated_df) == 3  # Original DataFrame returned on validation errors

    def test_pandera_error_format_conversion(self):
        """Test that Pandera errors are properly converted to DataBeak format."""
        df = pd.DataFrame({"score": [-10, 50, 150]})  # -10 and 150 violate constraints

        rules = {"score": {"type": "float", "min": 0.0, "max": 100.0, "nullable": False}}

        _validated_df, errors = validate_dataframe_with_pandera(df, rules)

        assert len(errors) > 0
        for error in errors:
            # Check that error has expected DataBeak format
            assert "column" in error
            assert "error_type" in error
            assert "message" in error
            assert "failing_cases" in error


class TestBuiltInSchemas:
    """Test the predefined Pandera schemas."""

    def test_numeric_data_schema(self):
        """Test NumericDataSchema with valid data."""
        valid_data = pd.DataFrame(
            {
                "value": [10.5, 20.0, 30.5],
                "category": ["A", "B", "C"],
            }
        )

        validated_df = NumericDataSchema.validate(valid_data)
        assert len(validated_df) == 3

    def test_time_series_schema(self):
        """Test TimeSeriesSchema with valid data."""
        valid_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3),
                "value": [10.5, 20.0, 30.5],
                "metric": ["cpu", "memory", "disk"],
            }
        )

        validated_df = TimeSeriesSchema.validate(valid_data)
        assert len(validated_df) == 3

    def test_financial_data_schema(self):
        """Test FinancialDataSchema with valid data."""
        valid_data = pd.DataFrame(
            {
                "amount": [100.50, 200.75, 50.25],
                "currency": ["USD", "EUR", "GBP"],
                "date": pd.date_range("2023-01-01", periods=3),
                "account_id": ["ACC12345", "ACC67890", "ACC11111"],
            }
        )

        validated_df = FinancialDataSchema.validate(valid_data)
        assert len(validated_df) == 3

    def test_financial_data_schema_violations(self):
        """Test FinancialDataSchema with constraint violations."""
        invalid_data = pd.DataFrame(
            {
                "amount": [-100.50, 200.75],  # Negative amount violates constraint
                "currency": ["USD", "JPY"],  # JPY not in allowed currencies
                "date": pd.date_range("2023-01-01", periods=2),
                "account_id": ["ACC1", "ACC67890"],  # ACC1 too short
            }
        )

        with pytest.raises(pa.errors.SchemaError):  # Should raise validation error
            FinancialDataSchema.validate(invalid_data, lazy=False)


class TestTypedDataFrameCreation:
    """Test creation of type-annotated DataFrames."""

    def test_create_typed_dataframe_success(self):
        """Test successful creation of typed DataFrame."""
        data = {"value": [10.5, 20.0, 30.5], "category": ["A", "B", "C"]}

        typed_df = create_typed_dataframe(data, NumericDataSchema)

        assert len(typed_df) == 3
        assert list(typed_df.columns) == ["value", "category"]

    def test_create_typed_dataframe_validation_failure(self):
        """Test typed DataFrame creation with validation failure."""
        data = {
            "value": [-10.5, 20.0],
            "category": ["A", "B"],
        }  # Negative value violates constraint

        with pytest.raises(pa.errors.SchemaError):  # Should raise validation error
            create_typed_dataframe(data, NumericDataSchema)


class TestPanderaResourceLimits:
    """Test that Pandera integration respects DataBeak resource limits."""

    def test_large_dataset_sampling(self):
        """Test that large datasets are sampled appropriately."""
        # This would need to be tested in integration with the validation server
        # as it requires the full context and settings

    def test_violation_limit_enforcement(self):
        """Test that validation violations are limited to prevent resource exhaustion."""
        # This would be tested through the validation server integration
        # with actual violation limit enforcement


class TestPanderaConfiguration:
    """Test Pandera configuration and settings integration."""

    def test_base_schema_configuration(self):
        """Test that DataBeakBaseSchema has proper configuration."""
        # Check that base schema has the expected configuration
        config = DataBeakBaseSchema.Config

        assert hasattr(config, "coerce")
        assert hasattr(config, "lazy")
        assert hasattr(config, "strict")

    def test_schema_flexibility(self):
        """Test that schemas handle flexible data loading."""
        # Test coercion and type conversion
        mixed_data = pd.DataFrame(
            {
                "value": ["10.5", "20", "30.5"],  # String numbers should be coerced
                "category": ["A", "B", "C"],
            }
        )

        # Should succeed with coercion enabled
        validated_df = NumericDataSchema.validate(mixed_data)
        assert len(validated_df) == 3
