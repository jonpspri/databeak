"""Unified secure expression model for DataBeak mathematical expressions.

This module provides a single, comprehensive SecureExpression type that replaces
all previous expression types (Formula, Expression, ColumnFormula, ApplyExpression,
etc.) with a unified model that supports all use cases.

Author: DataBeak Security Team
Issue: #46 - Address pandas.eval() code injection vulnerability
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.secure_evaluator import validate_expression_safety


class SecureExpression(BaseModel):
    """Unified secure mathematical expression with validation and context support.

    This is the single expression type used throughout DataBeak for all mathematical
    operations. It provides safety validation, variable substitution, and optional
    metadata for flexible usage across different scenarios.

    Examples of valid expressions:
        - "col1 + col2"                    # Column references
        - "abs(col1) * 2.5"               # Mathematical functions
        - "np.sqrt(col1 + col2)"          # Numpy functions
        - "x * 2 + 10"                    # Variable expressions (for apply operations)
        - "max(col1, 100)"                # Element-wise operations

    Examples of blocked expressions:
        - "__import__('os').system('rm -rf /')"  # System access
        - "exec('malicious_code')"               # Code execution
        - "open('/etc/passwd').read()"           # File access

    Usage patterns:
        # Simple formula
        SecureExpression(expression="col1 + col2")

        # Apply operation with variable
        SecureExpression(expression="x * 2", variable_name="x")

        # Formula with description
        SecureExpression(expression="col1 + col2", description="Sum of two columns")
    """

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        description="Safe mathematical expression using column names and mathematical functions",
        min_length=1,
        max_length=1000,  # Prevent extremely long expressions
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what the expression computes",
        max_length=200,
    )
    variable_name: str = Field(
        default="x",
        description="Variable name used in expression for apply operations",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",  # Valid Python identifier
    )

    @field_validator("expression")
    @classmethod
    def validate_expression_safety(cls, v: str) -> str:
        """Validate that expression contains only safe mathematical operations.

        Returns:
            The validated expression string

        Raises:
            ValueError: If expression contains unsafe operations

        """
        try:
            validate_expression_safety(v)
        except Exception as e:
            msg = f"Unsafe expression: {e}"
            raise ValueError(msg) from e
        return v

    @field_validator("variable_name")
    @classmethod
    def validate_variable_name(cls, v: str) -> str:
        """Validate that variable name is safe."""
        if v in ("exec", "eval", "open", "import", "__import__"):
            msg = f"Variable name '{v}' is not allowed"
            raise ValueError(msg)
        return v

    def get_expression_with_column(self, column_name: str) -> str:
        """Get expression with variable substituted for column reference.

        Used for apply operations where a variable represents column values.

        Returns:
            Expression string with column reference

        """
        # Safely quote column name for pandas
        safe_column_ref = f"`{column_name.replace('`', '')}`"
        return self.expression.replace(self.variable_name, safe_column_ref)

    def __str__(self) -> str:
        """Return the expression string."""
        return self.expression

    def __repr__(self) -> str:
        """Return detailed representation."""
        parts = [f"expression='{self.expression}'"]
        if self.description:
            parts.append(f"description='{self.description}'")
        if self.variable_name != "x":
            parts.append(f"variable_name='{self.variable_name}'")
        return f"SecureExpression({', '.join(parts)})"

    # Convenience class methods for common use cases
    @classmethod
    def formula(cls, expression: str, description: str | None = None) -> SecureExpression:
        """Create expression for column formulas.

        Returns:
            SecureExpression configured for formula usage

        Example:
            SecureExpression.formula("col1 + col2", "Sum of columns")

        """
        return cls(expression=expression, description=description)

    @classmethod
    def apply_operation(cls, expression: str, variable: str = "x") -> SecureExpression:
        """Create expression for apply operations.

        Returns:
            SecureExpression configured for apply operations

        Example:
            SecureExpression.apply_operation("x * 2 + 1", "x")

        """
        return cls(expression=expression, variable_name=variable)

    @classmethod
    def condition(cls, expression: str, description: str | None = None) -> SecureExpression:
        """Create expression for conditional operations.

        Returns:
            SecureExpression configured for conditional usage

        Example:
            SecureExpression.condition("col1 > 0", "Positive values")

        """
        return cls(expression=expression, description=description or "Conditional expression")


# For backward compatibility - these will be deprecated
Formula = SecureExpression
Expression = SecureExpression
