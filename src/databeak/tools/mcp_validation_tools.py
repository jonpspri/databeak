"""FastMCP validation tool definitions for DataBeak."""

from __future__ import annotations

from typing import Any, Literal, cast

from fastmcp import Context, FastMCP

from ..models.tool_responses import (  # TC001 - these are needed at runtime for FastMCP decorators
    DataQualityResult,
    FindAnomaliesResult,
    ValidateSchemaResult,
)
from .validation import (
    CompletenessRule,
    ConsistencyRule, 
    DataTypesRule,
    DuplicatesRule,
    OutliersRule,
    QualityRuleType,
    UniquenessRule,
    ValidationSchema,
    check_data_quality as _check_data_quality,
    find_anomalies as _find_anomalies,
    validate_schema as _validate_schema,
)


def register_validation_tools(mcp: FastMCP) -> None:
    """Register validation tools with FastMCP server."""

    @mcp.tool
    async def validate_schema(
        session_id: str, schema: dict[str, dict[str, Any]], ctx: Context | None = None
    ) -> ValidateSchemaResult:
        """Validate data against a schema definition."""
        # Convert dict to ValidationSchema for proper type validation
        schema_model = ValidationSchema(schema)
        return await _validate_schema(session_id, schema_model, ctx)

    @mcp.tool
    async def check_data_quality(
        session_id: str,
        rules: list[dict[str, Any]] | None = None,
        ctx: Context | None = None,
    ) -> DataQualityResult:
        """Check data quality based on predefined or custom rules."""
        if rules is None:
            parsed_rules = None
        else:
            parsed_rules: list[QualityRuleType] = []
            for rule_dict in rules:
                rule_type = rule_dict.get("type")
                if rule_type == "completeness":
                    parsed_rules.append(CompletenessRule(**rule_dict))
                elif rule_type == "duplicates":
                    parsed_rules.append(DuplicatesRule(**rule_dict))
                elif rule_type == "uniqueness":
                    parsed_rules.append(UniquenessRule(**rule_dict))
                elif rule_type == "data_types":
                    parsed_rules.append(DataTypesRule(**rule_dict))
                elif rule_type == "outliers":
                    parsed_rules.append(OutliersRule(**rule_dict))
                elif rule_type == "consistency":
                    parsed_rules.append(ConsistencyRule(**rule_dict))
                else:
                    raise ValueError(f"Unknown rule type: {rule_type}")
        return await _check_data_quality(session_id, parsed_rules, ctx)

    @mcp.tool
    async def find_anomalies(
        session_id: str,
        columns: list[str] | None = None,
        sensitivity: float = 0.95,
        methods: list[str] | None = None,
        ctx: Context | None = None,
    ) -> FindAnomaliesResult:
        """Find anomalies in the data using multiple detection methods."""
        # Convert and validate method strings
        if methods is not None:
            valid_methods = []
            for method in methods:
                if method in ["statistical", "pattern", "missing"]:
                    valid_methods.append(cast(Literal["statistical", "pattern", "missing"], method))
                else:
                    raise ValueError(f"Invalid method: {method}")
            typed_methods = valid_methods
        else:
            typed_methods = None
        return await _find_anomalies(session_id, columns, sensitivity, typed_methods, ctx)
