"""Data validation tools for CSV data quality checks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, RootModel

from ..models.csv_session import get_session_manager
from ..models.data_models import OperationType
from ..models.tool_responses import (
    AnomalyResults,
    AnomalySummary,
    DataQualityResult,
    FindAnomaliesResult,
    MissingAnomaly,
    PatternAnomaly,
    QualityIssue,
    QualityResults,
    QualityRuleResult,
    StatisticalAnomaly,
    ValidateSchemaResult,
    ValidationError,
    ValidationSummary,
)

if TYPE_CHECKING:
    from fastmcp import Context

logger = logging.getLogger(__name__)


class ColumnValidationRules(BaseModel):
    """Validation rules for a single column."""
    
    type: str | None = Field(None, description="Expected data type: int, float, str, bool, datetime")
    nullable: bool = Field(True, description="Whether null values are allowed")
    min: int | float | None = Field(None, description="Minimum value for numeric columns")
    max: int | float | None = Field(None, description="Maximum value for numeric columns")
    pattern: str | None = Field(None, description="Regex pattern for string validation")
    values: list[str] | None = Field(None, description="List of allowed values")
    unique: bool = Field(False, description="Whether values must be unique")
    min_length: int | None = Field(None, description="Minimum string length")
    max_length: int | None = Field(None, description="Maximum string length")


class ValidationSchema(RootModel[dict[str, ColumnValidationRules]]):
    """Schema definition for data validation.
    
    Example:
        {
            "column_name": {
                "type": "int",
                "nullable": False,
                "min": 0,
                "max": 100,
                "pattern": "^[A-Z]+$",
                "values": ["A", "B", "C"],
                "unique": True
            }
        }
    """


class QualityRule(BaseModel):
    """Base class for quality rules."""
    
    type: str


class CompletenessRule(QualityRule):
    """Rule for checking data completeness."""
    
    type: Literal["completeness"] = "completeness"
    threshold: float = Field(0.95, ge=0.0, le=1.0)
    columns: list[str] | None = None


class DuplicatesRule(QualityRule):
    """Rule for checking duplicate rows."""
    
    type: Literal["duplicates"] = "duplicates"
    threshold: float = Field(0.01, ge=0.0, le=1.0)
    columns: list[str] | None = None


class UniquenessRule(QualityRule):
    """Rule for checking column uniqueness."""
    
    type: Literal["uniqueness"] = "uniqueness"
    column: str
    expected_unique: bool = True


class DataTypesRule(QualityRule):
    """Rule for checking data type consistency."""
    
    type: Literal["data_types"] = "data_types"


class OutliersRule(QualityRule):
    """Rule for checking outliers in numeric columns."""
    
    type: Literal["outliers"] = "outliers"
    threshold: float = Field(0.05, ge=0.0, le=1.0)


class ConsistencyRule(QualityRule):
    """Rule for checking data consistency between columns."""
    
    type: Literal["consistency"] = "consistency"
    columns: list[str] = Field(default_factory=list)


# Union type for all quality rules
QualityRuleType = CompletenessRule | DuplicatesRule | UniquenessRule | DataTypesRule | OutliersRule | ConsistencyRule


def _default_anomaly_methods() -> list[Literal["statistical", "pattern", "missing"]]:
    """Default methods for anomaly detection."""
    return ["statistical", "pattern", "missing"]


class AnomalyDetectionParams(BaseModel):
    """Parameters for anomaly detection."""
    
    columns: list[str] | None = None
    sensitivity: float = Field(0.95, ge=0.0, le=1.0)
    methods: list[Literal["statistical", "pattern", "missing"]] = Field(
        default_factory=_default_anomaly_methods
    )


async def validate_schema(
    session_id: str,
    schema: ValidationSchema,
    ctx: Context | None = None,  # noqa: ARG001
) -> ValidateSchemaResult:
    """Validate data against a schema definition.

    Args:
        session_id: Session identifier
        schema: Schema definition with column rules
                Example: {
                    "column_name": {
                        "type": "int",  # int, float, str, bool, datetime
                        "nullable": False,
                        "min": 0,
                        "max": 100,
                        "pattern": "^[A-Z]+$",
                        "values": ["A", "B", "C"],  # allowed values
                        "unique": True
                    }
                }
        ctx: FastMCP context

    Returns:
        ValidateSchemaResult with validation results
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.data_session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.data_session.df
        validation_errors: dict[str, list[ValidationError]] = {}

        # Parse and validate schema structure using Pydantic
        try:
            if isinstance(schema, ValidationSchema):
                parsed_schema = schema.root
            else:
                schema_model = ValidationSchema(schema)
                parsed_schema = schema_model.root
        except Exception as e:
            raise ToolError(f"Invalid schema format: {e}") from e

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
                    )
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
                        )
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
                        )
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
                            )
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
                            )
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
                                )
                            )
                except Exception as e:
                    col_errors.append(
                        ValidationError(
                            error="pattern_error",
                            message=f"Invalid regex pattern: {e!s}",
                        )
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
                        )
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
                        )
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
                            )
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
                            )
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

        # Flatten all validation errors
        all_errors = []
        for error_list in validation_errors.values():
            all_errors.extend(error_list)

        return ValidateSchemaResult(
            session_id=session_id,
            valid=is_valid,
            errors=all_errors,
            summary=validation_summary,
            validation_errors=validation_errors,
        )

    except Exception as e:
        logger.error(f"Error validating schema: {e!s}")
        raise ToolError(f"Error validating schema: {e!s}") from e


async def check_data_quality(
    session_id: str,
    rules: list[QualityRuleType] | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> DataQualityResult:
    """Check data quality based on predefined or custom rules.

    Args:
        session_id: Session identifier
        rules: Custom quality rules to check. If None, uses default rules.
               Example: [
                   {"type": "completeness", "threshold": 0.95},
                   {"type": "uniqueness", "column": "id"},
                   {"type": "consistency", "columns": ["start_date", "end_date"]}
               ]
        ctx: FastMCP context

    Returns:
        DataQualityResult with quality check results
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.data_session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.data_session.df
        rule_results: list[QualityRuleResult] = []
        quality_issues: list[QualityIssue] = []
        recommendations: list[str] = []

        # Default rules if none provided
        if not rules:
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
                            )
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
                        "Consider removing duplicate rows using the remove_duplicates tool"
                    )

                # Add rule result
                rule_results.append(
                    QualityRuleResult(
                        rule_type="duplicates",
                        passed=passed,
                        score=round(score, 2),
                        issues=rule_issues,
                    )
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
                        issue = QualityIssue(
                            type="non_unique_values",
                            severity="high",
                            column=str(column),
                            message=f"Column '{column}' expected to be unique but has duplicates",
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
                        )
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
                                f"Consider converting to numeric type using change_column_type tool"
                            )

                        # Add rule result
                        rule_results.append(
                            QualityRuleResult(
                                rule_type="data_type_consistency",
                                passed=not mixed_types,
                                score=score,
                                issues=[],
                                column=col,
                            )
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
                        )
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
                            )
                        )

                        total_score += score
                        score_count += 1

        # Calculate overall score
        overall_score = round(total_score / score_count, 2) if score_count > 0 else 100.0

        # Add general recommendations
        if not recommendations and overall_score < 85:
            recommendations.append(
                "Consider running profile_data to get a comprehensive overview of data issues"
            )

        # Count passed/failed rules
        passed_rules = sum(1 for rule in rule_results if rule.passed)
        failed_rules = len(rule_results) - passed_rules

        # Create QualityResults
        quality_results = QualityResults(
            overall_score=overall_score,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            total_issues=len(quality_issues),
            rule_results=rule_results,
            issues=quality_issues,
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
            session_id=session_id,
            quality_results=quality_results,
        )

    except Exception as e:
        logger.error(f"Error checking data quality: {e!s}")
        raise ToolError(f"Error checking data quality: {e!s}") from e


async def find_anomalies(
    session_id: str,
    columns: list[str] | None = None,
    sensitivity: float = 0.95,
    methods: list[Literal["statistical", "pattern", "missing"]] | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> FindAnomaliesResult:
    """Find anomalies in the data using multiple detection methods.

    Args:
        session_id: Session identifier
        columns: Columns to check (None for all)
        sensitivity: Detection sensitivity (0.0 to 1.0, higher = more sensitive)
        methods: Detection methods to use (default: ["statistical", "pattern"])
        ctx: FastMCP context

    Returns:
        FindAnomaliesResult with anomaly detection results
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.data_session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.data_session.df

        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
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
                                            [null_indices[i], null_indices[i + 1]]
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
            session_id=session_id,
            anomalies=anomaly_results,
            columns_analyzed=target_cols,
            methods_used=[str(m) for m in methods],  # Convert to list[str] for compatibility
            sensitivity=sensitivity,
        )

    except Exception as e:
        logger.error(f"Error finding anomalies: {e!s}")
        raise ToolError(f"Error finding anomalies: {e!s}") from e
