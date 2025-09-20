"""Pandera schema definitions for DataFrame validation in DataBeak.

This module provides Pandera schema integration for enhanced DataFrame validation with type safety
and comprehensive error reporting. It complements the existing validation_server.py functionality
with schema-based validation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera as pa
from pandera import DataFrameModel, Field
from pandera.typing import DataFrame, Series

from .csv_session import get_csv_settings


class DataBeakBaseSchema(DataFrameModel):
    """Base schema class for DataBeak with common configuration."""

    class Config:
        """Pandera schema configuration."""

        # Enable coercion for flexible data loading
        coerce = True
        # Report all errors, not just the first one
        lazy = True
        # Allow additional columns not defined in schema
        strict = False


def create_pandera_schema_from_validation_rules(
    validation_rules: dict[str, dict[str, Any]],
) -> type[DataFrameModel]:
    """Create a Pandera SchemaModel from DataBeak validation rules.

    Args:
        validation_rules: Dictionary mapping column names to validation rules

    Returns:
        Dynamically created Pandera SchemaModel class

    Example:
        rules = {
            "age": {"type": "int", "min": 0, "max": 120, "nullable": False},
            "name": {"type": "str", "min_length": 1, "max_length": 100}
        }
        Schema = create_pandera_schema_from_validation_rules(rules)
        validated_df = Schema.validate(df)
    """
    settings = get_csv_settings()

    # Build schema attributes dynamically
    schema_attrs = {}

    for col_name, rules in validation_rules.items():
        # Determine pandas dtype
        expected_type = rules.get("type", "str")
        nullable = rules.get("nullable", True)

        if expected_type == "int":
            dtype = pd.Int64Dtype() if nullable else int
        elif expected_type == "float":
            dtype = pd.Float64Dtype() if nullable else float
        elif expected_type == "bool":
            dtype = pd.BooleanDtype() if nullable else bool
        elif expected_type == "datetime":
            dtype = "datetime64[ns]"
        else:  # str or object
            dtype = pd.StringDtype() if nullable else str

        # Build Field constraints
        field_constraints = {}

        # Numeric constraints
        if expected_type in ("int", "float"):
            if "min" in rules:
                field_constraints["ge"] = rules["min"]
            if "max" in rules:
                field_constraints["le"] = rules["max"]

        # String length constraints
        if expected_type == "str":
            if "min_length" in rules:
                field_constraints["str_length"] = {"min_value": rules["min_length"]}
            if "max_length" in rules:
                if "str_length" in field_constraints:
                    field_constraints["str_length"]["max_value"] = rules["max_length"]
                else:
                    field_constraints["str_length"] = {"max_value": rules["max_length"]}

        # Pattern matching
        if "pattern" in rules:
            field_constraints["str_matches"] = rules["pattern"]

        # Allowed values
        if "values" in rules:
            field_constraints["isin"] = rules["values"]

        # Uniqueness (handled at schema level, not field level in Pandera)
        unique = rules.get("unique", False)

        # Create the field with constraints
        if field_constraints:
            schema_attrs[col_name] = Series[dtype](Field(**field_constraints), nullable=nullable)
        else:
            schema_attrs[col_name] = Series[dtype](nullable=nullable)

        # Handle uniqueness at schema level
        if unique:
            # Add uniqueness check as a schema-level check
            def uniqueness_check(df: pd.DataFrame, col=col_name) -> bool:
                return df[col].nunique() == len(df[col].dropna())

            schema_attrs[f"_check_unique_{col_name}"] = pa.Check(
                uniqueness_check, element_wise=False, name=f"unique_{col_name}"
            )

    # Create schema class with Config
    schema_attrs["Config"] = type(
        "Config",
        (),
        {
            "coerce": True,
            "lazy": True,
            "strict": False,
            # Limit error reporting to prevent resource exhaustion
            "max_errors": settings.max_validation_violations,
        },
    )

    # Create and return the schema class
    return type("GeneratedDataBeakSchema", (DataBeakBaseSchema,), schema_attrs)


def validate_dataframe_with_pandera(
    df: pd.DataFrame, validation_rules: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate a DataFrame using Pandera with DataBeak validation rules.

    Args:
        df: DataFrame to validate
        validation_rules: DataBeak-style validation rules

    Returns:
        Tuple of (validated_dataframe, validation_errors)

    Raises:
        SchemaError: If validation fails with critical errors
    """
    try:
        # Create schema from validation rules
        schema_class = create_pandera_schema_from_validation_rules(validation_rules)

        # Validate the DataFrame
        validated_df = schema_class.validate(df, lazy=True)

        return validated_df, []

    except pa.errors.SchemaErrors as schema_errors:
        # Convert Pandera errors to DataBeak format
        databeak_errors = []

        for error in schema_errors.schema_errors:
            databeak_errors.append(
                {
                    "column": error.column if hasattr(error, "column") else None,
                    "error_type": error.check if hasattr(error, "check") else "schema_error",
                    "message": str(error),
                    "failing_cases": getattr(error, "failure_cases", []),
                }
            )

        return df, databeak_errors

    except pa.errors.SchemaError as schema_error:
        # Single schema error
        databeak_errors = [
            {
                "column": getattr(schema_error, "column", None),
                "error_type": "schema_error",
                "message": str(schema_error),
                "failing_cases": getattr(schema_error, "failure_cases", []),
            }
        ]

        return df, databeak_errors


# Example schemas for common DataBeak use cases
class NumericDataSchema(DataBeakBaseSchema):
    """Schema for numeric data with basic constraints."""

    value: Series[float] = Field(ge=0, nullable=True)
    category: Series[str] = Field(nullable=False)


class TimeSeriesSchema(DataBeakBaseSchema):
    """Schema for time series data."""

    timestamp: Series[pd.Timestamp] = Field(nullable=False)
    value: Series[float] = Field(nullable=True)
    metric: Series[str] = Field(nullable=False)


class FinancialDataSchema(DataBeakBaseSchema):
    """Schema for financial data with business rules."""

    amount: Series[float] = Field(ge=0, nullable=False)
    currency: Series[str] = Field(isin=["USD", "EUR", "GBP"], nullable=False)
    date: Series[pd.Timestamp] = Field(nullable=False)
    account_id: Series[str] = Field(str_length={"min_value": 5, "max_value": 20}, nullable=False)

    @pa.check("amount", element_wise=False)
    def check_amount_distribution(self, series: pd.Series) -> bool:
        """Check that amounts follow reasonable distribution."""
        return series.std() < series.mean() * 2  # Basic sanity check


# Utility function for DataFrame type annotation with Pandera
def create_typed_dataframe(
    data: dict[str, Any], schema_class: type[DataFrameModel]
) -> DataFrame[DataFrameModel]:
    """Create a type-annotated DataFrame with Pandera schema validation.

    Args:
        data: Dictionary of column data
        schema_class: Pandera SchemaModel class for validation

    Returns:
        Type-annotated DataFrame validated against schema

    Example:
        data = {"age": [25, 30, 35], "name": ["Alice", "Bob", "Charlie"]}
        typed_df = create_typed_dataframe(data, MySchema)
        # typed_df is now type-annotated with MySchema
    """
    df = pd.DataFrame(data)
    return schema_class.validate(df)
