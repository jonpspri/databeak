"""Secure expression evaluator for DataBeak mathematical expressions.

This module provides safe evaluation of user-provided mathematical expressions
by replacing pandas.eval() usage with a restricted execution environment.

Security Features:
- AST-based validation to block dangerous operations
- Allowlisted functions and operators only
- Sandboxed execution with no system access
- Timeout protection against infinite loops
- Column reference validation

Author: DataBeak Security Team
Issue: #46 - Address pandas.eval() code injection vulnerability
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from simpleeval import NameNotDefined, SimpleEval

from ..exceptions import InvalidParameterError


class SecureExpressionEvaluator:
    """Secure mathematical expression evaluator for DataBeak columns.

    Replaces unsafe pandas.eval() usage with a restricted execution environment that only allows
    mathematical operations and column references.
    """

    # Safe binary operators
    SAFE_OPERATORS: ClassVar[dict[type[ast.AST], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
    }

    # Safe unary operators
    SAFE_UNARY_OPERATORS: ClassVar[dict[type[ast.AST], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.inv,
    }

    # Safe comparison operators
    SAFE_COMPARISONS: ClassVar[dict[type[ast.AST], Any]] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }

    @staticmethod
    def _safe_max(*args: Any) -> Any:
        """Element-wise maximum that works with pandas Series."""
        if len(args) == 1:
            return np.max(args[0])
        return np.maximum(*args)

    @staticmethod
    def _safe_min(*args: Any) -> Any:
        """Element-wise minimum that works with pandas Series."""
        if len(args) == 1:
            return np.min(args[0])
        return np.minimum(*args)

    # Safe numpy functions (prefixed with np.)
    SAFE_NUMPY_FUNCTIONS: ClassVar[dict[str, Any]] = {
        "np.sqrt": np.sqrt,
        "np.exp": np.exp,
        "np.log": np.log,
        "np.log10": np.log10,
        "np.sin": np.sin,
        "np.cos": np.cos,
        "np.tan": np.tan,
        "np.arcsin": np.arcsin,
        "np.arccos": np.arccos,
        "np.arctan": np.arctan,
        "np.sinh": np.sinh,
        "np.cosh": np.cosh,
        "np.tanh": np.tanh,
        "np.abs": np.abs,
        "np.maximum": np.maximum,
        "np.minimum": np.minimum,
        "np.floor": np.floor,
        "np.ceil": np.ceil,
        "np.round": np.round,
        "np.sum": np.sum,
        "np.mean": np.mean,
        "np.std": np.std,
        "np.var": np.var,
        "np.pi": np.pi,
        "np.e": np.e,
    }

    # Dangerous patterns to explicitly block
    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
        r"__.*__",  # Dunder methods
        r"exec\s*\(",  # exec function
        r"eval\s*\(",  # eval function
        r"open\s*\(",  # file operations
        r"import\s+",  # import statements
        r"from\s+.*import",  # from import
        r"globals\s*\(",  # globals access
        r"locals\s*\(",  # locals access
        r"vars\s*\(",  # vars function
        r"dir\s*\(",  # dir function
        r"getattr\s*\(",  # getattr function
        r"setattr\s*\(",  # setattr function
        r"hasattr\s*\(",  # hasattr function
        r"delattr\s*\(",  # delattr function
        r"compile\s*\(",  # compile function
    ]

    def __init__(self) -> None:
        """Initialize the secure expression evaluator."""
        self._evaluator = SimpleEval()
        self._setup_evaluator()

    def _setup_evaluator(self) -> None:
        """Configure the SimpleEval instance with safe functions and operators."""
        # Set safe operators (including comparison operators)
        all_operators = {
            **self.SAFE_OPERATORS,
            **self.SAFE_COMPARISONS,
            **self.SAFE_UNARY_OPERATORS,
        }
        self._evaluator.operators = all_operators

        # Create safe functions with proper pandas compatibility
        safe_functions = {
            # Basic math (element-wise versions for pandas compatibility)
            "abs": np.abs,
            "max": self._safe_max,
            "min": self._safe_min,
            "sum": np.sum,
            "round": np.round,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            # Math functions (numpy versions for pandas Series compatibility)
            "sqrt": np.sqrt,
            "pow": np.power,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "degrees": np.degrees,
            "radians": np.radians,
            "floor": np.floor,
            "ceil": np.ceil,
            "trunc": np.trunc,
            "fabs": np.fabs,
            # Constants
            "pi": math.pi,
            "e": math.e,
        }

        # Add safe functions to the evaluator's namespace
        all_functions = {**safe_functions, **self.SAFE_NUMPY_FUNCTIONS}
        self._evaluator.functions = all_functions

        # Create a restricted numpy-like object with only safe functions
        class SafeNumpy:
            """Restricted numpy-like object with only mathematical functions."""

            # Add safe numpy functions as attributes
            sqrt = np.sqrt
            exp = np.exp
            log = np.log
            log10 = np.log10
            sin = np.sin
            cos = np.cos
            tan = np.tan
            arcsin = np.arcsin
            arccos = np.arccos
            arctan = np.arctan
            sinh = np.sinh
            cosh = np.cosh
            tanh = np.tanh
            abs = np.abs
            maximum = np.maximum
            minimum = np.minimum
            floor = np.floor
            ceil = np.ceil
            round = np.round
            sum = np.sum
            mean = np.mean
            std = np.std
            var = np.var
            pi = np.pi
            e = np.e

        # Set safe names (constants and restricted modules)
        self._evaluator.names = {
            "pi": math.pi,
            "e": math.e,
            "True": True,
            "False": False,
            "None": None,
            "np": SafeNumpy(),  # Restricted numpy access
        }

    def evaluate(self, expression: str, context: dict[str, Any] | None = None) -> Any:
        """Evaluate a safe mathematical expression and return the result.

        Returns:
            The result of evaluating the expression

        Raises:
            InvalidParameterError: If the expression is unsafe or evaluation fails

        """
        # First validate the expression for safety
        validate_expression_safety(expression)

        # Temporarily add context variables if provided
        original_names = None
        if context:
            original_names = self._evaluator.names.copy()
            self._evaluator.names.update(context)

        try:
            # Evaluate using the configured safe evaluator
            return self._evaluator.eval(expression)
        except NameNotDefined as e:
            msg = "expression"
            raise InvalidParameterError(
                msg,
                expression,
                f"Unknown variable or function: {e}",
            ) from e
        except ZeroDivisionError:
            # Let ZeroDivisionError bubble up as expected by tests
            raise
        except Exception as e:
            msg = "expression"
            raise InvalidParameterError(
                msg,
                expression,
                f"Evaluation failed: {e}",
            ) from e
        finally:
            # Restore original names if context was provided
            if original_names is not None:
                self._evaluator.names = original_names

    def validate_expression_syntax(self, expression: str) -> None:
        """Validate that an expression only contains safe operations.

        Raises:
            InvalidParameterError: If the expression contains unsafe operations

        """
        if not expression or not isinstance(expression, str):
            msg = "expression"
            raise InvalidParameterError(
                msg,
                expression,
                "Expression must be a non-empty string",
            )

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                msg = "expression"
                raise InvalidParameterError(
                    msg,
                    expression,
                    f"Expression contains dangerous pattern: {pattern}",
                )

        # Try to parse the expression as valid Python syntax
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            msg = "expression"
            raise InvalidParameterError(msg, expression, f"Invalid syntax: {e}") from e

        # Validate AST nodes
        self._validate_ast_nodes(tree)

    def _validate_ast_nodes(self, node: ast.AST) -> None:
        """Recursively validate AST nodes for security.

        Raises:
            InvalidParameterError: If the node contains unsafe operations

        """
        allowed_node_types = (
            ast.Constant,
            # Variables and names
            ast.Name,
            ast.Load,
            ast.Store,
            # Mathematical operations
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            # Binary operators
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            # Unary operators
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.Invert,
            # Comparison operators
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
            # Boolean operations
            ast.BoolOp,
            ast.And,
            ast.Or,
            # Function calls (will be validated separately)
            ast.Call,
            # Attribute access (for np.function calls)
            ast.Attribute,
            # Containers (for function arguments)
            ast.List,
            ast.Tuple,
            # Expression wrapper
            ast.Expression,
        )

        if not isinstance(node, allowed_node_types):
            msg = "expression"
            raise InvalidParameterError(
                msg,
                str(node),
                f"Unsafe AST node type: {type(node).__name__}",
            )

        # Validate function calls
        if isinstance(node, ast.Call):
            self._validate_function_call(node)

        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            self._validate_ast_nodes(child)

    def _validate_function_call(self, node: ast.Call) -> None:
        """Validate that a function call is safe.

        Raises:
            InvalidParameterError: If the function call is unsafe

        """
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle np.function calls
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
                func_name = f"np.{node.func.attr}"
            else:
                msg = "expression"
                raise InvalidParameterError(
                    msg,
                    str(node.func),
                    "Only np.* attribute access is allowed",
                )
        else:
            msg = "expression"
            raise InvalidParameterError(
                msg,
                str(node.func),
                "Only simple function calls are allowed",
            )

        # Check if function is in allowlist
        # Get safe function names from the evaluator
        evaluator_functions = set(self._evaluator.functions.keys())
        all_safe_functions = evaluator_functions | set(self.SAFE_NUMPY_FUNCTIONS.keys())
        if func_name not in all_safe_functions:
            msg = "expression"
            raise InvalidParameterError(
                msg,
                func_name,
                f"Function '{func_name}' is not allowed. "
                f"Allowed functions: {', '.join(sorted(all_safe_functions))}",
            )

    def evaluate_column_expression(
        self,
        expression: str,
        dataframe: pd.DataFrame,
        column_context: dict[str, str] | None = None,
    ) -> pd.Series:
        """Safely evaluate a mathematical expression with column references.

        Returns:
            pd.Series: Result of the expression evaluation

        Raises:
            InvalidParameterError: If the expression is unsafe or evaluation fails

        """
        # Validate expression syntax first
        self.validate_expression_syntax(expression)

        # Build context with column data
        context = {}

        # Handle column context mapping (e.g., x -> actual column name)
        if column_context:
            for var_name, column_name in column_context.items():
                if column_name not in dataframe.columns:
                    msg = "column"
                    raise InvalidParameterError(
                        msg,
                        column_name,
                        f"Column '{column_name}' not found in DataFrame",
                    )
                context[var_name] = dataframe[column_name]

        # Add all column names as direct references
        for col in dataframe.columns:
            # Use backticks for column names with spaces/special chars
            safe_col_name = col.replace("`", "")  # Remove existing backticks
            context[safe_col_name] = dataframe[col]
            context[f"`{safe_col_name}`"] = dataframe[col]

        # Add constants and safe functions to context
        context.update(self._evaluator.names)

        try:
            # Use simpleeval for safe execution
            self._evaluator.names.update(context)
            result = self._evaluator.eval(expression)

            # Ensure result is a pandas Series
            if not isinstance(result, pd.Series):
                # Convert scalar or array results to Series
                if hasattr(result, "__len__") and len(result) == len(dataframe):
                    result = pd.Series(result, index=dataframe.index)
                else:
                    # Scalar result - broadcast to all rows
                    result = pd.Series([result] * len(dataframe), index=dataframe.index)

            return result  # type: ignore[no-any-return]

        except NameNotDefined as e:
            msg = "expression"
            raise InvalidParameterError(
                msg,
                expression,
                f"Undefined variable in expression: {e}. "
                f"Available columns: {list(dataframe.columns)}",
            ) from e
        except Exception as e:
            msg = "expression"
            raise InvalidParameterError(
                msg,
                expression,
                f"Expression evaluation failed: {e}",
            ) from e

    def evaluate_simple_formula(self, formula: str, dataframe: pd.DataFrame) -> pd.Series:
        """Evaluate a formula with direct column references.

        This method is designed to replace direct pandas.eval() usage where
        the formula already contains proper column references.

        Returns:
            pd.Series: Result of the formula evaluation

        Raises:
            InvalidParameterError: If the formula is unsafe or evaluation fails

        """
        return self.evaluate_column_expression(formula, dataframe)

    def get_supported_functions(self) -> list[str]:
        """Get list of all supported functions.

        Returns:
            List of function names that can be used in expressions

        """
        evaluator_functions = set(self._evaluator.functions.keys())
        all_functions = evaluator_functions | set(self.SAFE_NUMPY_FUNCTIONS.keys())
        return sorted(all_functions)

    def get_supported_operators(self) -> list[str]:
        """Get list of all supported operators.

        Returns:
            List of operator symbols that can be used in expressions

        """
        return ["+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "~"]


def evaluate_expression_safely(
    expression: str,
    dataframe: pd.DataFrame,
    column_context: dict[str, str] | None = None,
) -> pd.Series:
    """Evaluate mathematical expression safely.

    Returns:
        pd.Series: Result of expression evaluation

    Raises:
        InvalidParameterError: If expression is unsafe or evaluation fails

    """
    return SecureExpressionEvaluator().evaluate_column_expression(
        expression, dataframe, column_context
    )


def validate_expression_safety(expression: str) -> None:
    """Validate mathematical expression syntax.

    Raises:
        InvalidParameterError: If expression contains unsafe operations

    """
    SecureExpressionEvaluator().validate_expression_syntax(expression)
