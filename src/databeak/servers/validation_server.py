"""Standalone validation server for DataBeak using FastMCP server composition."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

# Import session management from the main package
from ..models import OperationType, get_session_manager
from ..models.csv_session import get_csv_settings

# from ..models.pandera_schemas import validate_dataframe_with_pandera

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class ValidationError(BaseModel):
    """Individual validation error details."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(description="Type of validation error encountered")
    message: str = Field(description="Human-readable error message")
    actual_type: str | None = Field(
        default=None,
        description="Actual data type found (for type mismatch errors)",
    )
    null_count: int | None = Field(default=None, description="Number of null/missing values found")
    null_indices: list[int] | None = Field(
        default=None,
        description="Row indices where null values occur (limited to 100)",
    )
    violation_count: int | None = Field(
        default=None,
        description="Number of values violating the rule",
    )
    min_found: float | None = Field(
        default=None,
        description="Minimum value found (for min/max violations)",
    )
    max_found: float | None = Field(
        default=None,
        description="Maximum value found (for min/max violations)",
    )
    sample_violations: list[str] | None = Field(
        default=None,
        description="Sample of values that violated the rule",
    )
    invalid_values: list[str] | None = Field(
        default=None,
        description="List of invalid values found",
    )
    duplicate_count: int | None = Field(
        default=None,
        description="Number of duplicate values found",
    )


class ValidationSummary(BaseModel):
    """Summary of validation results."""

    model_config = ConfigDict(extra="forbid")

    total_columns: int = Field(description="Total number of columns in schema")
    valid_columns: int = Field(description="Number of columns that passed validation")
    invalid_columns: int = Field(description="Number of columns that failed validation")
    missing_columns: list[str] = Field(
        description="Columns defined in schema but missing from data",
    )
    extra_columns: list[str] = Field(description="Columns in data but not defined in schema")


class ValidateSchemaResult(BaseModel):
    """Response model for schema validation operations."""

    valid: bool = Field(description="Whether validation passed overall")
    errors: list[ValidationError] = Field(description="All validation errors found")
    summary: ValidationSummary = Field(description="Summary of validation results")
    validation_errors: dict[str, list[ValidationError]] = Field(
        description="Validation errors grouped by column name",
    )


class QualityIssue(BaseModel):
    """Individual quality issue details."""

    type: str = Field(description="Type of quality issue identified")
    severity: str = Field(description="Severity level: low, medium, high, or critical")
    column: str | None = Field(
        default=None,
        description="Column name where issue was found (None for dataset-wide issues)",
    )
    message: str = Field(description="Human-readable description of the quality issue")
    affected_rows: int = Field(description="Number of rows affected by this issue")
    metric_value: float = Field(description="Measured metric value that triggered the issue")
    threshold: float = Field(description="Threshold value that was exceeded or not met")


class QualityRuleResult(BaseModel):
    """Result of a single quality rule check."""

    rule_type: str = Field(description="Type of quality rule that was checked")
    passed: bool = Field(description="Whether the quality rule passed")
    score: float = Field(description="Quality score for this rule (0-100)")
    issues: list[QualityIssue] = Field(description="List of quality issues found by this rule")
    column: str | None = Field(
        default=None,
        description="Column name if rule applies to specific column",
    )


class QualityResults(BaseModel):
    """Comprehensive quality check results."""

    model_config = ConfigDict(extra="forbid")

    overall_score: float = Field(description="Overall data quality score (0-100)")
    passed_rules: int = Field(description="Number of quality rules that passed")
    failed_rules: int = Field(description="Number of quality rules that failed")
    total_issues: int = Field(description="Total number of quality issues found")
    rule_results: list[QualityRuleResult] = Field(
        description="Detailed results for each quality rule",
    )
    issues: list[QualityIssue] = Field(description="All quality issues found across all rules")
    recommendations: list[str] = Field(description="Suggested actions to improve data quality")


class DataQualityResult(BaseModel):
    """Response model for data quality check operations."""

    quality_results: QualityResults = Field(description="Comprehensive quality assessment results")


class StatisticalAnomaly(BaseModel):
    """Statistical anomaly detection result."""

    anomaly_count: int = Field(description="Number of statistical anomalies detected")
    anomaly_indices: list[int] = Field(description="Row indices where anomalies were found")
    anomaly_values: list[float] = Field(description="Sample of anomalous values found")
    mean: float = Field(description="Mean value of the column")
    std: float = Field(description="Standard deviation of the column")
    lower_bound: float = Field(description="Lower bound for normal values")
    upper_bound: float = Field(description="Upper bound for normal values")


class PatternAnomaly(BaseModel):
    """Pattern-based anomaly detection result."""

    anomaly_count: int = Field(description="Number of pattern anomalies detected")
    anomaly_indices: list[int] = Field(description="Row indices where pattern anomalies were found")
    sample_values: list[str] = Field(description="Sample values that don't match expected patterns")
    expected_patterns: list[str] = Field(description="List of expected patterns that were violated")


class MissingAnomaly(BaseModel):
    """Missing value anomaly detection result.

    Represents anomalies found in missing value patterns within a column.
    This includes both the quantity of missing values and their distribution
    patterns (clustered vs random), which can indicate data quality issues
    or systematic collection problems.

    Attributes:
        missing_count: Total number of missing/null values in the column
        missing_ratio: Proportion of missing values (0.0 to 1.0)
        missing_indices: Row indices where missing values occur (limited to first 100)
        sequential_clusters: Number of consecutive missing value sequences found
        pattern: Distribution pattern of missing values ('clustered' or 'random')
    """

    missing_count: int = Field(
        description="Total number of missing/null values found in the column",
    )
    missing_ratio: float = Field(
        description="Ratio of missing values to total values (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    missing_indices: list[int] = Field(
        description="Row indices where missing values occur (limited to first 100 for performance)",
    )
    sequential_clusters: int = Field(
        description="Number of consecutive missing value sequences detected",
        ge=0,
    )
    pattern: Literal["clustered", "random"] = Field(
        description="Distribution pattern of missing values ('clustered' or 'random')",
    )


class AnomalySummary(BaseModel):
    """Summary of anomaly detection results."""

    total_anomalies: int = Field(description="Total number of anomalies found across all columns")
    affected_rows: int = Field(description="Number of rows containing at least one anomaly")
    affected_columns: list[str] = Field(
        description="Names of columns where anomalies were detected",
    )


class AnomalyResults(BaseModel):
    """Comprehensive anomaly detection results."""

    model_config = ConfigDict(extra="forbid")

    summary: AnomalySummary = Field(description="Summary statistics of anomaly detection results")
    by_column: dict[str, StatisticalAnomaly | PatternAnomaly | MissingAnomaly] = Field(
        description="Anomalies organized by column name",
    )
    by_method: dict[str, dict[str, StatisticalAnomaly | PatternAnomaly | MissingAnomaly]] = Field(
        description="Anomalies organized by detection method",
    )


class FindAnomaliesResult(BaseModel):
    """Response model for anomaly detection operations."""

    anomalies: AnomalyResults = Field(description="Comprehensive anomaly detection results")
    columns_analyzed: list[str] = Field(
        description="Names of columns that were analyzed for anomalies",
    )
    methods_used: list[str] = Field(description="Detection methods that were applied")
    sensitivity: float = Field(description="Sensitivity threshold used for detection (0.0-1.0)")


# ============================================================================
# VALIDATION RULE MODELS
# ============================================================================


class ColumnValidationRules(BaseModel):
    """Validation rules for a single column."""

    type: Literal["int", "float", "str", "bool", "datetime"] | None = Field(
        None,
        description="Expected data type: int, float, str, bool, datetime",
    )
    nullable: bool = Field(default=True, description="Whether null values are allowed")
    min: int | float | None = Field(default=None, description="Minimum value for numeric columns")
    max: int | float | None = Field(default=None, description="Maximum value for numeric columns")
    pattern: str | None = Field(default=None, description="Regex pattern for string validation")
    values: list[str] | None = Field(default=None, description="List of allowed values")
    unique: bool = Field(default=False, description="Whether values must be unique")
    min_length: int | None = Field(default=None, description="Minimum string length")
    max_length: int | None = Field(default=None, description="Maximum string length")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str | None) -> str | None:
        """Validate that pattern is a valid regular expression."""
        if v is None:
            return v

        import re

        try:
            re.compile(v)
            return v
        except re.error as e:
            msg = f"Invalid regular expression: {e}"
            raise ValueError(msg) from e


class QualityRule(BaseModel):
    """Base class for quality rules."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Type of quality rule")


class CompletenessRule(QualityRule):
    """Rule for checking data completeness."""

    type: Literal["completeness"] = Field(
        default="completeness",
        description="Rule type identifier",
    )
    threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum completeness ratio required (0.0-1.0)",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Specific columns to check (None for all columns)",
    )


class DuplicatesRule(QualityRule):
    """Rule for checking duplicate rows."""

    type: Literal["duplicates"] = Field(default="duplicates", description="Rule type identifier")
    threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Maximum allowable duplicate ratio (0.0-1.0)",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Columns to consider for duplicate detection (None for all columns)",
    )


class UniquenessRule(QualityRule):
    """Rule for checking column uniqueness."""

    type: Literal["uniqueness"] = Field(default="uniqueness", description="Rule type identifier")
    column: str = Field(description="Column name to check for uniqueness")
    expected_unique: bool = Field(
        default=True,
        description="Whether column values are expected to be unique",
    )


class DataTypesRule(QualityRule):
    """Rule for checking data type consistency."""

    type: Literal["data_types"] = Field(default="data_types", description="Rule type identifier")


class OutliersRule(QualityRule):
    """Rule for checking outliers in numeric columns."""

    type: Literal["outliers"] = Field(default="outliers", description="Rule type identifier")
    threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowable outlier ratio (0.0-1.0)",
    )


class ConsistencyRule(QualityRule):
    """Rule for checking data consistency between columns."""

    type: Literal["consistency"] = Field(default="consistency", description="Rule type identifier")
    columns: list[str] = Field(
        default_factory=list,
        description="Column pairs to check for consistency",
    )


class ValidationSchema(RootModel[dict[str, ColumnValidationRules]]):
    """Schema definition for data validation."""


# Discriminated union type for all quality rules
QualityRuleType = Annotated[
    CompletenessRule
    | DuplicatesRule
    | UniquenessRule
    | DataTypesRule
    | OutliersRule
    | ConsistencyRule,
    Field(discriminator="type"),
]


def _default_anomaly_methods() -> list[Literal["statistical", "pattern", "missing"]]:
    """Return default methods for anomaly detection."""
    return ["statistical", "pattern", "missing"]


class AnomalyDetectionParams(BaseModel):
    """Parameters for anomaly detection."""

    columns: list[str] | None = Field(
        None,
        description="Specific columns to analyze (None for all columns)",
    )
    sensitivity: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity threshold (higher = more strict)",
    )
    methods: list[Literal["statistical", "pattern", "missing"]] = Field(
        default_factory=_default_anomaly_methods,
        description="Anomaly detection methods to apply",
    )


# ============================================================================
# VALIDATION RESOURCE MANAGEMENT
# ============================================================================


def apply_violation_limits(violations: list, limit: int, operation_name: str) -> tuple[list, bool]:
    """Apply resource limits to violation collections.

    Args:
        violations: List of violations to limit
        limit: Maximum number of violations to keep
        operation_name: Name of operation for logging

    Returns:
        Tuple of (limited_violations, was_truncated)
    """
    if len(violations) > limit:
        logger.info(
            "%s found %d violations, limiting to %d for resource management",
            operation_name,
            len(violations),
            limit,
        )
        return violations[:limit], True
    return violations, False


def sample_large_dataset(
    df: pd.DataFrame, max_sample_size: int, operation_name: str
) -> pd.DataFrame:
    """Sample large datasets for memory-efficient operations.

    Args:
        df: DataFrame to sample
        max_sample_size: Maximum sample size
        operation_name: Name of operation for logging

    Returns:
        Sampled DataFrame (or original if under limit)
    """
    if len(df) > max_sample_size:
        logger.info(
            "%s dataset has %d rows, sampling %d for resource management",
            operation_name,
            len(df),
            max_sample_size,
        )
        return df.sample(n=max_sample_size, random_state=42)
    return df


# ============================================================================
# VALIDATION LOGIC
# ============================================================================


def validate_schema(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    schema: Annotated[
        ValidationSchema,
        Field(description="Schema definition with column validation rules"),
    ],
) -> ValidateSchemaResult:
    """Validate data against a schema definition.

    Args:
        ctx: FastMCP context for session access
        schema: Schema definition with column validation rules

    Returns:
        ValidateSchemaResult with validation status and detailed error information
    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

        df = session.df
        settings = get_csv_settings()
        validation_errors: dict[str, list[ValidationError]] = {}

        parsed_schema = schema.root

        # Apply resource management for large datasets
        logger.info("Validating schema for %d rows, %d columns", len(df), len(df.columns))
        if len(df) > settings.max_anomaly_sample_size:
            logger.warning(
                "Large dataset (%d rows), using sample of %d for validation",
                len(df),
                settings.max_anomaly_sample_size,
            )
            df = sample_large_dataset(df, settings.max_anomaly_sample_size, "Schema validation")

        # Convert validation_summary to ValidationSummary
        validation_summary = ValidationSummary(
            total_columns=len(parsed_schema),
            valid_columns=0,
            invalid_columns=0,
            missing_columns=[],
            extra_columns=[],
        )

        # Check for missing and extra columns
        schema_columns = set(parsed_schema.keys())
        df_columns = set(df.columns)

        validation_summary.missing_columns = list(schema_columns - df_columns)
        validation_summary.extra_columns = list(df_columns - schema_columns)

        # Validate each column in schema
        for col_name, rules_model in parsed_schema.items():
            rules = rules_model.model_dump(exclude_none=True)
            if col_name not in df.columns:
                validation_errors[col_name] = [
                    ValidationError(
                        error="column_missing",
                        message=f"Column '{col_name}' not found in data",
                    ),
                ]
                validation_summary.invalid_columns += 1
                continue

            col_errors: list[ValidationError] = []
            col_data = df[col_name]

            # Type validation
            expected_type = rules.get("type")
            if expected_type:
                type_valid = False
                if expected_type == "int":
                    type_valid = pd.api.types.is_integer_dtype(col_data)
                elif expected_type == "float":
                    type_valid = pd.api.types.is_float_dtype(col_data)
                elif expected_type == "str":
                    type_valid = pd.api.types.is_string_dtype(col_data) or col_data.dtype == object
                elif expected_type == "bool":
                    type_valid = pd.api.types.is_bool_dtype(col_data)
                elif expected_type == "datetime":
                    type_valid = pd.api.types.is_datetime64_any_dtype(col_data)

                if not type_valid:
                    col_errors.append(
                        ValidationError(
                            error="type_mismatch",
                            message=f"Expected type '{expected_type}', got '{col_data.dtype}'",
                            actual_type=str(col_data.dtype),
                        ),
                    )

            # Nullable validation
            if not rules.get("nullable", True):
                null_count = col_data.isna().sum()
                if null_count > 0:
                    col_errors.append(
                        ValidationError(
                            error="null_values",
                            message=f"Column contains {null_count} null values",
                            null_count=int(null_count),
                            null_indices=df[col_data.isna()].index.tolist()[:100],
                        ),
                    )

            # Min/Max validation for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                if "min" in rules:
                    min_val = rules["min"]
                    violations = col_data[col_data < min_val]
                    if len(violations) > 0:
                        col_errors.append(
                            ValidationError(
                                error="min_violation",
                                message=f"{len(violations)} values below minimum {min_val}",
                                violation_count=len(violations),
                                min_found=float(violations.min()),
                            ),
                        )

                if "max" in rules:
                    max_val = rules["max"]
                    violations = col_data[col_data > max_val]
                    if len(violations) > 0:
                        col_errors.append(
                            ValidationError(
                                error="max_violation",
                                message=f"{len(violations)} values above maximum {max_val}",
                                violation_count=len(violations),
                                max_found=float(violations.max()),
                            ),
                        )

            # Pattern validation for string columns
            if "pattern" in rules and (
                col_data.dtype == object or pd.api.types.is_string_dtype(col_data)
            ):
                pattern = rules["pattern"]
                try:
                    non_null = col_data.dropna()
                    if len(non_null) > 0:
                        matches = non_null.astype(str).str.match(pattern)
                        violations = non_null[~matches]
                        if len(violations) > 0:
                            col_errors.append(
                                ValidationError(
                                    error="pattern_violation",
                                    message=f"{len(violations)} values don't match pattern '{pattern}'",
                                    violation_count=len(violations),
                                    sample_violations=[
                                        str(v) for v in violations.head(10).tolist()
                                    ],
                                ),
                            )
                except Exception as e:
                    col_errors.append(
                        ValidationError(
                            error="pattern_error",
                            message=f"Invalid regex pattern: {e!s}",
                        ),
                    )

            # Allowed values validation
            if "values" in rules:
                values = rules["values"]
                # Schema validation already ensures values is a list
                allowed = set(values)
                actual = set(col_data.dropna().unique())
                invalid = actual - allowed
                if invalid:
                    col_errors.append(
                        ValidationError(
                            error="invalid_values",
                            message=f"Found {len(invalid)} invalid values",
                            invalid_values=[str(v) for v in list(invalid)[:50]],
                        ),
                    )

            # Uniqueness validation
            if rules.get("unique", False):
                duplicates = col_data.duplicated()
                if duplicates.any():
                    col_errors.append(
                        ValidationError(
                            error="duplicate_values",
                            message=f"Column contains {duplicates.sum()} duplicate values",
                            duplicate_count=int(duplicates.sum()),
                        ),
                    )

            # Length validation for strings
            if col_data.dtype == object or pd.api.types.is_string_dtype(col_data):
                if "min_length" in rules:
                    min_len = rules["min_length"]
                    # Schema validation already ensures min_len is numeric
                    str_data = col_data.dropna().astype(str)
                    short = str_data[str_data.str.len() < int(min_len)]
                    if len(short) > 0:
                        col_errors.append(
                            ValidationError(
                                error="min_length_violation",
                                message=f"{len(short)} values shorter than {min_len} characters",
                                violation_count=len(short),
                            ),
                        )

                if "max_length" in rules:
                    max_len = rules["max_length"]
                    # Schema validation already ensures max_len is numeric
                    str_data = col_data.dropna().astype(str)
                    long = str_data[str_data.str.len() > int(max_len)]
                    if len(long) > 0:
                        col_errors.append(
                            ValidationError(
                                error="max_length_violation",
                                message=f"{len(long)} values longer than {max_len} characters",
                                violation_count=len(long),
                            ),
                        )

            if col_errors:
                validation_errors[col_name] = col_errors
                validation_summary.invalid_columns += 1
            else:
                validation_summary.valid_columns += 1

        is_valid = len(validation_errors) == 0 and len(validation_summary.missing_columns) == 0

        session.record_operation(
            OperationType.VALIDATE,
            {
                "type": "schema_validation",
                "is_valid": is_valid,
                "errors_count": len(validation_errors),
            },
        )

        # Flatten all validation errors with resource limits
        all_errors = []
        for error_list in validation_errors.values():
            all_errors.extend(error_list)

        # Apply violation limits to prevent resource exhaustion
        limited_errors, was_truncated = apply_violation_limits(
            all_errors, settings.max_validation_violations, "Schema validation"
        )

        if was_truncated:
            logger.warning(
                "Validation found %d errors, limited to %d",
                len(all_errors),
                settings.max_validation_violations,
            )

        return ValidateSchemaResult(
            valid=is_valid,
            errors=limited_errors,
            summary=validation_summary,
            validation_errors=validation_errors,
        )

    except Exception as e:
        logger.error("Error validating schema: %s", str(e))
        msg = f"Error validating schema: {e!s}"
        raise ToolError(msg) from e


def check_data_quality(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    rules: Annotated[
        list[QualityRuleType] | None,
        Field(description="List of quality rules to check (None = use default rules)"),
    ] = None,
) -> DataQualityResult:
    """Check data quality based on predefined or custom rules.

    Args:
        ctx: FastMCP context for session access
        rules: List of quality rules to check (None = use default rules)

    Returns:
        DataQualityResult with comprehensive quality assessment results
    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

        df = session.df
        settings = get_csv_settings()
        rule_results: list[QualityRuleResult] = []
        quality_issues: list[QualityIssue] = []
        recommendations: list[str] = []

        # Apply resource management for large datasets
        logger.info("Checking data quality for %d rows, %d columns", len(df), len(df.columns))
        if len(df) > settings.max_anomaly_sample_size:
            logger.warning(
                "Large dataset (%d rows), using sample of %d for quality check",
                len(df),
                settings.max_anomaly_sample_size,
            )
            df = sample_large_dataset(df, settings.max_anomaly_sample_size, "Data quality check")

        # Use default rules if none provided
        if rules is None:
            rules = [
                CompletenessRule(threshold=0.95),
                DuplicatesRule(threshold=0.01),
                DataTypesRule(),
                OutliersRule(threshold=0.05),
                ConsistencyRule(),
            ]

        total_score: float = 0
        score_count = 0

        for rule in rules:
            if isinstance(rule, CompletenessRule):
                # Check data completeness
                threshold = rule.threshold
                columns = rule.columns if rule.columns is not None else df.columns.tolist()

                for col in columns:
                    if col in df.columns:
                        completeness = 1 - (df[col].isna().sum() / len(df))
                        passed = completeness >= threshold
                        score = completeness * 100

                        # Create issue if failed
                        rule_issues = []
                        if not passed:
                            issue = QualityIssue(
                                type="incomplete_data",
                                severity="high" if completeness < 0.5 else "medium",
                                column=col,
                                message=f"Column '{col}' is only {round(completeness * 100, 2)}% complete",
                                affected_rows=int(df[col].isna().sum()),
                                metric_value=completeness,
                                threshold=float(threshold),
                            )
                            rule_issues.append(issue)
                            quality_issues.append(issue)

                        # Add rule result
                        rule_results.append(
                            QualityRuleResult(
                                rule_type="completeness",
                                passed=passed,
                                score=round(score, 2),
                                issues=rule_issues,
                                column=col,
                            ),
                        )

                        total_score += score
                        score_count += 1

            elif isinstance(rule, DuplicatesRule):
                # Check for duplicate rows
                threshold = rule.threshold
                subset = rule.columns

                duplicates = df.duplicated(subset=subset)
                duplicate_ratio = duplicates.sum() / len(df)
                passed = duplicate_ratio <= threshold
                score = (1 - duplicate_ratio) * 100

                # Create issue if failed
                rule_issues = []
                if not passed:
                    issue = QualityIssue(
                        type="duplicate_rows",
                        severity="high" if duplicate_ratio > 0.1 else "medium",
                        message=f"Found {duplicates.sum()} duplicate rows ({round(duplicate_ratio * 100, 2)}%)",
                        affected_rows=int(duplicates.sum()),
                        metric_value=duplicate_ratio,
                        threshold=float(threshold),
                    )
                    rule_issues.append(issue)
                    quality_issues.append(issue)
                    recommendations.append(
                        "Consider removing duplicate rows using the remove_duplicates tool",
                    )

                # Add rule result
                rule_results.append(
                    QualityRuleResult(
                        rule_type="duplicates",
                        passed=passed,
                        score=round(score, 2),
                        issues=rule_issues,
                    ),
                )

                total_score += score
                score_count += 1

            elif isinstance(rule, UniquenessRule):
                # Check column uniqueness
                column = rule.column
                if column in df.columns:
                    unique_ratio = df[column].nunique() / len(df)
                    expected_unique = rule.expected_unique

                    if expected_unique:
                        passed = unique_ratio >= 0.99
                        score = unique_ratio * 100
                    else:
                        passed = True
                        score = 100.0

                    # Create issue if failed
                    rule_issues = []
                    if not passed and expected_unique:
                        duplicate_count = len(df) - df[column].nunique()
                        issue = QualityIssue(
                            type="non_unique_values",
                            severity="high",
                            column=str(column),
                            message=f"Column '{column}' expected to be unique but has duplicates",
                            affected_rows=duplicate_count,
                            metric_value=unique_ratio,
                            threshold=0.99,
                        )
                        rule_issues.append(issue)
                        quality_issues.append(issue)

                    # Add rule result
                    rule_results.append(
                        QualityRuleResult(
                            rule_type="uniqueness",
                            passed=passed,
                            score=round(score, 2),
                            issues=rule_issues,
                            column=str(column),
                        ),
                    )

                    total_score += score
                    score_count += 1

            elif isinstance(rule, DataTypesRule):
                # Check data type consistency
                for col in df.columns:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Check for mixed types
                        types = col_data.apply(lambda x: type(x).__name__).unique()
                        mixed_types = len(types) > 1

                        # Check for numeric strings
                        if col_data.dtype == object:
                            numeric_strings = col_data.astype(str).str.match(r"^-?\d+\.?\d*$").sum()
                            numeric_ratio = numeric_strings / len(col_data)
                        else:
                            numeric_ratio = 0

                        score = 100.0 if not mixed_types else 50.0

                        # Create recommendations for numeric strings
                        if numeric_ratio > 0.9:
                            recommendations.append(
                                f"Column '{col}' appears to contain numeric data stored as strings. "
                                f"Consider converting to numeric type using change_column_type tool",
                            )

                        # Add rule result
                        rule_results.append(
                            QualityRuleResult(
                                rule_type="data_type_consistency",
                                passed=not mixed_types,
                                score=score,
                                issues=[],
                                column=col,
                            ),
                        )

                        total_score += score
                        score_count += 1

            elif isinstance(rule, OutliersRule):
                # Check for outliers in numeric columns
                threshold = rule.threshold
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1

                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_ratio = outliers / len(df)
                    passed = outlier_ratio <= threshold
                    score = (1 - min(outlier_ratio, 1)) * 100

                    # Create issue if failed
                    rule_issues = []
                    if not passed:
                        issue = QualityIssue(
                            type="outliers",
                            severity="medium",
                            column=col,
                            message=f"Column '{col}' has {outliers} outliers ({round(outlier_ratio * 100, 2)}%)",
                            affected_rows=int(outliers),
                            metric_value=outlier_ratio,
                            threshold=float(threshold),
                        )
                        rule_issues.append(issue)
                        quality_issues.append(issue)

                    # Add rule result
                    rule_results.append(
                        QualityRuleResult(
                            rule_type="outliers",
                            passed=passed,
                            score=round(score, 2),
                            issues=rule_issues,
                            column=col,
                        ),
                    )

                    total_score += score
                    score_count += 1

            elif isinstance(rule, ConsistencyRule):
                # Check data consistency
                columns = rule.columns

                # Date consistency check
                date_cols = df.select_dtypes(include=["datetime64"]).columns
                if len(date_cols) >= 2 and not columns:
                    columns = date_cols.tolist()

                if len(columns) >= 2:
                    col1, col2 = str(columns[0]), str(columns[1])
                    if (
                        col1 in df.columns
                        and col2 in df.columns
                        and pd.api.types.is_datetime64_any_dtype(df[col1])
                        and pd.api.types.is_datetime64_any_dtype(df[col2])
                    ):
                        inconsistent = (df[col1] > df[col2]).sum()
                        consistency_ratio = 1 - (inconsistent / len(df))
                        passed = consistency_ratio >= 0.99
                        score = consistency_ratio * 100

                        # Create issue if failed
                        rule_issues = []
                        if not passed:
                            issue = QualityIssue(
                                type="data_inconsistency",
                                severity="high",
                                message=f"Found {inconsistent} rows where {col1} > {col2}",
                                affected_rows=int(inconsistent),
                                metric_value=consistency_ratio,
                                threshold=0.99,
                            )
                            rule_issues.append(issue)
                            quality_issues.append(issue)

                        # Add rule result
                        rule_results.append(
                            QualityRuleResult(
                                rule_type="consistency",
                                passed=passed,
                                score=round(score, 2),
                                issues=rule_issues,
                            ),
                        )

                        total_score += score
                        score_count += 1

        # Calculate overall score
        overall_score = round(total_score / score_count, 2) if score_count > 0 else 100.0

        # Add general recommendations
        if not recommendations and overall_score < 85:
            recommendations.append(
                "Consider running profile_data to get a comprehensive overview of data issues",
            )

        # Count passed/failed rules
        passed_rules = sum(1 for rule in rule_results if rule.passed)
        failed_rules = len(rule_results) - passed_rules

        # Apply limits to quality issues to prevent resource exhaustion
        limited_issues, was_truncated = apply_violation_limits(
            quality_issues, settings.max_validation_violations, "Data quality check"
        )

        if was_truncated:
            logger.warning(
                "Quality check found %d issues, limited to %d",
                len(quality_issues),
                settings.max_validation_violations,
            )

        # Create QualityResults
        quality_results = QualityResults(
            overall_score=overall_score,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            total_issues=len(limited_issues),
            rule_results=rule_results,
            issues=limited_issues,
            recommendations=recommendations,
        )

        session.record_operation(
            OperationType.QUALITY_CHECK,
            {
                "rules_count": len(rules),
                "overall_score": overall_score,
                "issues_count": len(quality_issues),
            },
        )

        return DataQualityResult(
            quality_results=quality_results,
        )

    except Exception as e:
        logger.error("Error checking data quality: %s", str(e))
        msg = f"Error checking data quality: {e!s}"
        raise ToolError(msg) from e


def find_anomalies(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    columns: Annotated[
        list[str] | None,
        Field(description="List of columns to analyze (None = all columns)"),
    ] = None,
    sensitivity: Annotated[
        float,
        Field(description="Sensitivity threshold for anomaly detection (0-1)"),
    ] = 0.95,
    methods: Annotated[
        list[Literal["statistical", "pattern", "missing"]] | None,
        Field(description="Detection methods to use (None = all methods)"),
    ] = None,
) -> FindAnomaliesResult:
    """Find anomalies in the data using multiple detection methods.

    Args:
        ctx: FastMCP context for session access
        columns: List of columns to analyze (None = all columns)
        sensitivity: Sensitivity threshold for anomaly detection (0-1)
        methods: Detection methods to use (None = all methods)

    Returns:
        FindAnomaliesResult with comprehensive anomaly detection results
    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

        df = session.df
        settings = get_csv_settings()

        # Apply resource management for large datasets
        logger.info("Finding anomalies in %d rows, %d columns", len(df), len(df.columns))
        if len(df) > settings.max_anomaly_sample_size:
            logger.warning(
                "Large dataset (%d rows), using sample of %d for anomaly detection",
                len(df),
                settings.max_anomaly_sample_size,
            )
            df = sample_large_dataset(df, settings.max_anomaly_sample_size, "Anomaly detection")

        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                msg = f"Columns not found: {missing_cols}"
                raise ToolError(msg)
            target_cols = columns
        else:
            target_cols = df.columns.tolist()

        if not methods:
            methods = ["statistical", "pattern", "missing"]

        # Track anomalies using proper data structures
        total_anomalies = 0
        affected_rows: set[int] = set()
        affected_columns: list[str] = []
        by_column: dict[str, StatisticalAnomaly | PatternAnomaly | MissingAnomaly] = {}
        by_method: dict[str, dict[str, StatisticalAnomaly | PatternAnomaly | MissingAnomaly]] = {}

        # Statistical anomalies (outliers)
        if "statistical" in methods:
            numeric_cols = df[target_cols].select_dtypes(include=[np.number]).columns
            statistical_anomalies: dict[str, StatisticalAnomaly] = {}

            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Z-score method
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    z_threshold = 3 * (
                        1 - sensitivity + 0.5
                    )  # Adjust threshold based on sensitivity
                    z_anomalies = col_data.index[z_scores > z_threshold].tolist()

                    # IQR method
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    iqr_factor = 1.5 * (2 - sensitivity)  # Adjust factor based on sensitivity
                    lower = q1 - iqr_factor * iqr
                    upper = q3 + iqr_factor * iqr
                    iqr_anomalies = df.index[(df[col] < lower) | (df[col] > upper)].tolist()

                    # Combine both methods
                    combined_anomalies = list(set(z_anomalies) | set(iqr_anomalies))

                    if combined_anomalies:
                        statistical_anomaly = StatisticalAnomaly(
                            anomaly_count=len(combined_anomalies),
                            anomaly_indices=combined_anomalies[:100],
                            anomaly_values=[
                                float(v) for v in df.loc[combined_anomalies[:10], col].tolist()
                            ],
                            mean=float(col_data.mean()),
                            std=float(col_data.std()),
                            lower_bound=float(lower),
                            upper_bound=float(upper),
                        )
                        statistical_anomalies[col] = statistical_anomaly

                        total_anomalies += len(combined_anomalies)
                        affected_rows.update(combined_anomalies)
                        affected_columns.append(col)

            if statistical_anomalies:
                # Type cast for mypy
                by_method["statistical"] = dict(statistical_anomalies.items())

        # Pattern anomalies
        if "pattern" in methods:
            pattern_anomalies: dict[str, PatternAnomaly] = {}

            for col in target_cols:
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Detect unusual patterns
                        value_counts = col_data.value_counts()
                        total_count = len(col_data)

                        # Find rare values (appearing less than threshold)
                        threshold = (1 - sensitivity) * 0.01  # Adjust threshold
                        rare_values = value_counts[value_counts / total_count < threshold]

                        if len(rare_values) > 0:
                            rare_indices = df[df[col].isin(rare_values.index)].index.tolist()

                            # Check for format anomalies (e.g., different case, special characters)
                            common_pattern = None
                            if len(value_counts) > 10:
                                # Detect common pattern from frequent values
                                top_values = value_counts.head(10).index

                                # Check if most values are uppercase/lowercase
                                upper_count = sum(1 for v in top_values if str(v).isupper())
                                lower_count = sum(1 for v in top_values if str(v).islower())

                                if upper_count > 7:
                                    common_pattern = "uppercase"
                                elif lower_count > 7:
                                    common_pattern = "lowercase"

                            format_anomalies = []
                            if common_pattern:
                                for idx, val in col_data.items():
                                    if (
                                        common_pattern == "uppercase" and not str(val).isupper()
                                    ) or (common_pattern == "lowercase" and not str(val).islower()):
                                        format_anomalies.append(idx)

                            all_pattern_anomalies = list(set(rare_indices + format_anomalies))

                            if all_pattern_anomalies:
                                pattern_anomaly = PatternAnomaly(
                                    anomaly_count=len(all_pattern_anomalies),
                                    anomaly_indices=all_pattern_anomalies[:100],
                                    sample_values=[
                                        str(v) for v in rare_values.head(10).index.tolist()
                                    ],
                                    expected_patterns=[common_pattern] if common_pattern else [],
                                )
                                pattern_anomalies[col] = pattern_anomaly

                                total_anomalies += len(all_pattern_anomalies)
                                affected_rows.update(all_pattern_anomalies)
                                if col not in affected_columns:
                                    affected_columns.append(col)

            if pattern_anomalies:
                # Type cast for mypy
                by_method["pattern"] = dict(pattern_anomalies.items())

        # Missing value anomalies
        if "missing" in methods:
            missing_anomalies: dict[str, MissingAnomaly] = {}

            for col in target_cols:
                null_mask = df[col].isna()
                null_count = null_mask.sum()

                if null_count > 0:
                    null_ratio = null_count / len(df)

                    # Check for suspicious missing patterns
                    if 0 < null_ratio < 0.5:  # Partially missing
                        # Check if missing values are clustered
                        null_indices = df.index[null_mask].tolist()

                        # Check for sequential missing values
                        sequential_missing: list[list[int]] = []
                        if len(null_indices) > 1:
                            for i in range(len(null_indices) - 1):
                                if null_indices[i + 1] - null_indices[i] == 1 and (
                                    not sequential_missing
                                    or null_indices[i] - sequential_missing[-1][-1] == 1
                                ):
                                    if sequential_missing:
                                        sequential_missing[-1].append(null_indices[i + 1])
                                    else:
                                        sequential_missing.append(
                                            [null_indices[i], null_indices[i + 1]],
                                        )

                        # Flag as anomaly if there are suspicious patterns
                        is_anomaly = (
                            len(sequential_missing) > 0
                            and len(sequential_missing) > len(null_indices) * 0.3
                        )

                        if is_anomaly or (null_ratio > 0.1 and null_ratio < 0.3):
                            missing_anomaly = MissingAnomaly(
                                missing_count=int(null_count),
                                missing_ratio=round(null_ratio, 4),
                                missing_indices=null_indices[:100],
                                sequential_clusters=len(sequential_missing),
                                pattern="clustered" if sequential_missing else "random",
                            )
                            missing_anomalies[col] = missing_anomaly

                            if col not in affected_columns:
                                affected_columns.append(col)

            if missing_anomalies:
                # Type cast for mypy
                by_method["missing"] = dict(missing_anomalies.items())

        # Organize anomalies by column
        for _method_name, method_anomalies in by_method.items():
            for col, col_anomalies in method_anomalies.items():
                if col not in by_column:
                    by_column[col] = col_anomalies
                # Note: For simplicity, we're taking the first anomaly type per column
                # In practice, you might want to combine multiple anomaly types

        # Create summary
        affected_rows_list = list(affected_rows)[:1000]  # Limit for performance
        unique_affected_columns = list(set(affected_columns))

        summary = AnomalySummary(
            total_anomalies=total_anomalies,
            affected_rows=len(affected_rows_list),
            affected_columns=unique_affected_columns,
        )

        # Create final results
        anomaly_results = AnomalyResults(
            summary=summary,
            by_column=by_column,
            by_method=by_method,
        )

        session.record_operation(
            OperationType.ANOMALY_DETECTION,
            {
                "methods": methods,
                "sensitivity": sensitivity,
                "anomalies_found": total_anomalies,
            },
        )

        return FindAnomaliesResult(
            anomalies=anomaly_results,
            columns_analyzed=target_cols,
            methods_used=[str(m) for m in methods],  # Convert to list[str] for compatibility
            sensitivity=sensitivity,
        )

    except Exception as e:
        logger.error("Error finding anomalies: %s", str(e))
        msg = f"Error finding anomalies: {e!s}"
        raise ToolError(msg) from e


# TODO: Complete Pandera integration - https://github.com/jonpspri/databeak/issues/100
# Complex FastMCP/Pydantic compatibility issues need resolution


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================


# Create validation server
validation_server = FastMCP(
    "DataBeak-Validation",
    instructions="Data validation server for DataBeak",
)


# Register the logic functions directly as MCP tools (no wrapper functions needed)
validation_server.tool(name="validate_schema")(validate_schema)
# validation_server.tool(name="validate_with_pandera_schema")(validate_with_pandera_schema)
validation_server.tool(name="check_data_quality")(check_data_quality)
validation_server.tool(name="find_anomalies")(find_anomalies)
