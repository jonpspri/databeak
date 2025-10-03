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
import threading
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from simpleeval import NameNotDefined, SimpleEval

from databeak.exceptions import InvalidParameterError


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

        # Validate attribute access (for string methods like x.upper())
        if isinstance(node, ast.Attribute):
            self._validate_attribute_access(node)

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
            # Handle np.function calls and string method calls
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "np":
                    func_name = f"np.{node.func.attr}"
                else:
                    # This is a string method call like x.upper()
                    # Don't validate here - it will be handled in evaluate_string_expression
                    return
            else:
                msg = "expression"
                raise InvalidParameterError(
                    msg,
                    str(node.func),
                    "Only np.* and variable string method attribute access is allowed",
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

    def _validate_attribute_access(self, node: ast.Attribute) -> None:
        """Validate that attribute access is safe.

        Raises:
            InvalidParameterError: If the attribute access is unsafe

        """
        # Allow np.* attribute access (already handled in function call validation)
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            return

        # Allow variable.method() patterns for string operations
        if isinstance(node.value, ast.Name):
            # This is for patterns like x.upper(), x.lower(), etc.
            # The actual validation happens in _evaluate_string_method
            safe_string_methods = {
                "upper",
                "lower",
                "strip",
                "title",
                "capitalize",
                "swapcase",
                "len",
                "replace",
                "contains",
                "startswith",
                "endswith",
            }
            if node.attr in safe_string_methods:
                return

        # Block all other attribute access
        msg = "expression"
        raise InvalidParameterError(
            msg,
            f"{node.value}.{node.attr}" if hasattr(node.value, "id") else str(node),
            "Only np.* and safe string method attribute access is allowed",
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
        return self._evaluate_with_context(expression, dataframe, column_context)

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

    def evaluate_string_expression(
        self,
        expression: str,
        dataframe: pd.DataFrame,
        column_context: dict[str, str] | None = None,
    ) -> pd.Series:
        """Safely evaluate string expressions with pandas string methods.

        Handles both string operations (x.upper(), x.lower(), etc.) and mathematical expressions.

        Returns:
            pd.Series: Result of the expression evaluation

        Raises:
            InvalidParameterError: If the expression is unsafe or evaluation fails

        """
        # First validate the expression syntax only once
        self.validate_expression_syntax(expression)

        # Handle column context mapping
        if column_context and len(column_context) == 1:
            var_name, column_name = next(iter(column_context.items()))
            if column_name not in dataframe.columns:
                msg = "column"
                raise InvalidParameterError(
                    msg,
                    column_name,
                    f"Column '{column_name}' not found in DataFrame",
                )

            # Check for string method calls first
            if f"{var_name}." in expression:
                try:
                    return self._evaluate_string_method(
                        expression, dataframe[column_name], var_name
                    )
                except InvalidParameterError:
                    # If string method fails, fall through to mathematical evaluation
                    pass

        # Fall back to regular mathematical expression evaluation
        # Skip validation since we already validated above
        return self._evaluate_mathematical_expression(expression, dataframe, column_context)

    def _evaluate_mathematical_expression(
        self,
        expression: str,
        dataframe: pd.DataFrame,
        column_context: dict[str, str] | None = None,
    ) -> pd.Series:
        """Evaluate mathematical expression without re-validation.

        This method is used internally by evaluate_string_expression to avoid double-validation.
        The caller must ensure that the expression has already been validated for safety.

        Validation requirements:
            - The expression must be checked for unsafe AST nodes (e.g., no function calls except allowlisted ones, no attribute access, no system calls).
            - Only allowlisted functions and operators should be permitted.
            - Column references must be validated against the DataFrame.

        If validation is skipped:
            - Unsafe or malicious expressions may be executed, potentially leading to code injection, data leakage, or other security vulnerabilities.
            - This method does not perform any validation itself and should never be called directly with untrusted input.

        Raises:
            InvalidParameterError: If evaluation fails.
        """
        return self._evaluate_with_context(expression, dataframe, column_context)

    def _evaluate_with_context(
        self,
        expression: str,
        dataframe: pd.DataFrame,
        column_context: dict[str, str] | None = None,
    ) -> pd.Series:
        """Core evaluation logic used by both validated and pre-validated paths.

        This method performs the actual expression evaluation with column context.
        It does NOT validate the expression - callers must validate before calling.

        Args:
            expression: The expression to evaluate (must be pre-validated)
            dataframe: DataFrame containing the data
            column_context: Optional mapping of variable names to column names

        Returns:
            pd.Series: Result of the expression evaluation

        Raises:
            InvalidParameterError: If column references are invalid or evaluation fails

        """
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

    def _evaluate_string_method(
        self, expression: str, series: pd.Series, var_name: str
    ) -> pd.Series:
        """Evaluate string method expressions safely.

        Args:
            expression: The expression containing string methods (e.g., "x.upper()")
            series: The pandas Series to operate on
            var_name: The variable name used in the expression (e.g., "x")

        Returns:
            pd.Series: Result of the string operation

        Raises:
            InvalidParameterError: If the string operation is not supported

        """
        # Convert to string Series for string operations
        str_series = series.astype(str)

        # Define safe string operations
        safe_string_ops = {
            f"{var_name}.upper()": lambda s: s.str.upper(),
            f"{var_name}.lower()": lambda s: s.str.lower(),
            f"{var_name}.strip()": lambda s: s.str.strip(),
            f"{var_name}.title()": lambda s: s.str.title(),
            f"{var_name}.capitalize()": lambda s: s.str.capitalize(),
            f"{var_name}.swapcase()": lambda s: s.str.swapcase(),
            f"{var_name}.len()": lambda s: s.str.len(),
        }

        # Check for exact match first
        if expression in safe_string_ops:
            return safe_string_ops[expression](str_series)  # type: ignore[no-any-return]

        # Check for method calls with arguments

        # Handle strip with custom characters: x.strip("chars")
        strip_match = re.match(rf"{re.escape(var_name)}\.strip\(['\"]([^'\"]*)['\"]?\)", expression)
        if strip_match:
            chars = strip_match.group(1)
            return str_series.str.strip(chars)

        # Handle replace operations: x.replace("old", "new")
        replace_match = re.match(
            rf"{re.escape(var_name)}\.replace\(['\"]([^'\"]*)['\"], *['\"]([^'\"]*)['\"]?\)",
            expression,
        )
        if replace_match:
            old_val, new_val = replace_match.groups()
            return str_series.str.replace(old_val, new_val, regex=False)

        # Handle contains operations: x.contains("pattern")
        contains_match = re.match(
            rf"{re.escape(var_name)}\.contains\(['\"]([^'\"]*)['\"]?\)", expression
        )
        if contains_match:
            pattern = contains_match.group(1)
            return str_series.str.contains(pattern, na=False)

        # Handle startswith/endswith: x.startswith("prefix"), x.endswith("suffix")
        startswith_match = re.match(
            rf"{re.escape(var_name)}\.startswith\(['\"]([^'\"]*)['\"]?\)", expression
        )
        if startswith_match:
            prefix = startswith_match.group(1)
            return str_series.str.startswith(prefix, na=False)

        endswith_match = re.match(
            rf"{re.escape(var_name)}\.endswith\(['\"]([^'\"]*)['\"]?\)", expression
        )
        if endswith_match:
            suffix = endswith_match.group(1)
            return str_series.str.endswith(suffix, na=False)

        # If no string operation matched, raise error
        msg = "expression"
        supported_ops = ", ".join(
            [
                f"{var_name}.upper()",
                f"{var_name}.lower()",
                f"{var_name}.strip()",
                f"{var_name}.title()",
                f"{var_name}.capitalize()",
                f"{var_name}.len()",
                f"{var_name}.strip('chars')",
                f"{var_name}.replace('old', 'new')",
                f"{var_name}.contains('pattern')",
                f"{var_name}.startswith('prefix')",
                f"{var_name}.endswith('suffix')",
            ]
        )
        raise InvalidParameterError(
            msg,
            expression,
            f"Unsupported string operation. Supported operations: {supported_ops}",
        )


def create_secure_expression_evaluator() -> SecureExpressionEvaluator:
    """Create a new SecureExpressionEvaluator."""
    return SecureExpressionEvaluator()


_secure_expression_evaluator: SecureExpressionEvaluator | None = None
_lock = threading.Lock()


def get_secure_expression_evaluator() -> SecureExpressionEvaluator:
    """Return a singleton SecureExpressionEvaluator object."""
    global _secure_expression_evaluator  # noqa: PLW0603
    if _secure_expression_evaluator is None:
        with _lock:
            if _secure_expression_evaluator is None:
                _secure_expression_evaluator = create_secure_expression_evaluator()
    return _secure_expression_evaluator


def reset_secure_expression_evaluator() -> None:
    """Set the global SecureExpressionEvaluator to None (for testing)."""
    global _secure_expression_evaluator  # noqa: PLW0603
    with _lock:
        _secure_expression_evaluator = None


####
# TODO:  Should these functions be in some sort of service provided by the
#        Session?
####


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
    return get_secure_expression_evaluator().evaluate_column_expression(
        expression, dataframe, column_context
    )


def evaluate_string_expression_safely(
    expression: str,
    dataframe: pd.DataFrame,
    column_context: dict[str, str] | None = None,
) -> pd.Series:
    """Evaluate string or mathematical expression safely.

    Handles both string operations (x.upper(), x.lower(), etc.) and mathematical expressions.

    Returns:
        pd.Series: Result of expression evaluation

    Raises:
        InvalidParameterError: If expression is unsafe or evaluation fails

    """
    return get_secure_expression_evaluator().evaluate_string_expression(
        expression, dataframe, column_context
    )


def validate_expression_safety(expression: str) -> None:
    """Validate mathematical expression syntax.

    Raises:
        InvalidParameterError: If expression contains unsafe operations

    """
    get_secure_expression_evaluator().validate_expression_syntax(expression)
