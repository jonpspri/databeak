"""Unit tests for secure evaluator."""

import pytest

from databeak.exceptions import InvalidParameterError
from databeak.utils.secure_evaluator import (
    SecureExpressionEvaluator,
    get_secure_expression_evaluator,
    reset_secure_expression_evaluator,
    validate_expression_safety,
)


def is_safe_expression(expression: str) -> bool:
    """Helper function to check if expression is safe (returns boolean)."""
    try:
        validate_expression_safety(expression)
        return True
    except InvalidParameterError:
        return False


class TestSecureExpressionEvaluator:
    """Test SecureExpressionEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> SecureExpressionEvaluator:
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_safe_arithmetic_expressions(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test safe arithmetic expressions."""
        # Basic arithmetic
        assert evaluator.evaluate("2 + 3") == 5
        assert evaluator.evaluate("10 - 4") == 6
        assert evaluator.evaluate("6 * 7") == 42
        assert evaluator.evaluate("15 / 3") == 5.0

    def test_safe_comparison_expressions(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test safe comparison expressions."""
        assert evaluator.evaluate("5 > 3") is True
        assert evaluator.evaluate("2 < 1") is False
        assert evaluator.evaluate("4 == 4") is True
        assert evaluator.evaluate("3 != 5") is True
        assert evaluator.evaluate("7 >= 7") is True
        assert evaluator.evaluate("2 <= 1") is False

    def test_safe_logical_expressions(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test safe logical expressions."""
        assert evaluator.evaluate("True and False") is False
        assert evaluator.evaluate("True or False") is True
        assert evaluator.evaluate("not False") is True
        assert evaluator.evaluate("(5 > 3) and (2 < 4)") is True

    def test_safe_variable_access(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test safe variable access with context."""
        context = {"x": 10, "y": 5, "name": "test"}

        assert evaluator.evaluate("x + y", context) == 15
        assert evaluator.evaluate("x > y", context) is True
        assert evaluator.evaluate("name == 'test'", context) is True

    def test_safe_string_operations(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test safe string operations."""
        context = {"text": "hello world", "pattern": "world"}

        # String comparison
        assert evaluator.evaluate("'hello' == 'hello'") is True
        assert evaluator.evaluate("text == 'hello world'", context) is True

        # String membership (if supported)
        try:
            result = evaluator.evaluate("'world' in text", context)
            assert result is True
        except InvalidParameterError:
            # Skip if 'in' operator is not allowed
            pass

    def test_blocked_function_calls(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that function calls are blocked."""
        unsafe_expressions = [
            "print('hello')",
            "open('/etc/passwd')",
            "eval('2+2')",
            "exec('import os')",
            "len([1,2,3])",  # Even safe functions should be blocked
            "__import__('os')",
            "getattr(object, '__class__')",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_blocked_attribute_access(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that attribute access is blocked."""
        unsafe_expressions = [
            "''.__class__",
            "x.__dict__",
            "object.__subclasses__()",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_blocked_import_statements(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that import statements are blocked."""
        unsafe_expressions = [
            "import os",
            "from os import path",
            "__import__('sys')",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_blocked_dangerous_operations(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that dangerous operations are blocked."""
        unsafe_expressions = [
            "globals()",
            "locals()",
            "vars()",
            "dir()",
            "help()",
            "input()",
            "compile('2+2', '<string>', 'eval')",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_division_by_zero_handling(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test division by zero handling."""
        with pytest.raises(ZeroDivisionError):
            evaluator.evaluate("10 / 0")

    def test_syntax_error_handling(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test syntax error handling."""
        invalid_expressions = [
            "2 +",  # Incomplete expression
            "if True:",  # Statement, not expression
            "for i in range(10):",  # Loop statement
            "def func():",  # Function definition
        ]

        for expr in invalid_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_complex_safe_expressions(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test complex but safe expressions."""
        context = {"a": 5, "b": 10, "c": 3}

        # Complex arithmetic
        result = evaluator.evaluate("(a + b) * c - 2", context)
        assert result == 43  # (5 + 10) * 3 - 2 = 43

        # Complex comparisons
        result = evaluator.evaluate("(a > 3) and (b < 15) and (c == 3)", context)
        assert result is True

    def test_none_and_null_handling(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test None and null value handling."""
        context = {"value": None, "number": 42}

        # None comparisons
        assert evaluator.evaluate("value is None", context) is True
        assert evaluator.evaluate("number is not None", context) is True

    def test_empty_expression(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test empty expression handling."""
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("")

        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("   ")  # Whitespace only

    def test_very_large_numbers(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test handling of very large numbers."""
        # Should handle large numbers without issues
        result = evaluator.evaluate("999999999999 + 1")
        assert result == 1000000000000

    def test_nested_expressions(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test deeply nested expressions."""
        context = {"x": 2}

        # Nested parentheses
        result = evaluator.evaluate("((x + 1) * (x + 2)) + ((x - 1) * (x - 2))", context)
        # ((2+1) * (2+2)) + ((2-1) * (2-2)) = (3*4) + (1*0) = 12 + 0 = 12
        assert result == 12


class TestValidateExpression:
    """Test validate_expression function."""

    def test_validate_safe_expressions(self) -> None:
        """Test validation of safe expressions."""
        safe_expressions = [
            "2 + 3",
            "x > 5",
            "name == 'test'",
            "(a and b) or c",
            "value is None",
        ]

        for expr in safe_expressions:
            # Should not raise exception
            validate_expression_safety(expr)

    def test_validate_unsafe_expressions(self) -> None:
        """Test validation of unsafe expressions."""
        unsafe_expressions = [
            "print('hello')",
            "import os",
            "x.__class__",
            "eval('2+2')",
            "globals()",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                validate_expression_safety(expr)

    def test_validate_empty_expression(self) -> None:
        """Test validation of empty expressions."""
        with pytest.raises(InvalidParameterError):
            validate_expression_safety("")


class TestIsSafeExpression:
    """Test is_safe_expression function."""

    def test_safe_expressions_return_true(self) -> None:
        """Test that safe expressions return True."""
        safe_expressions = [
            "2 + 3",
            "x > 5",
            "name == 'test'",
            "(a and b) or c",
        ]

        for expr in safe_expressions:
            assert is_safe_expression(expr) is True

    def test_unsafe_expressions_return_false(self) -> None:
        """Test that unsafe expressions return False."""
        unsafe_expressions = [
            "print('hello')",
            "import os",
            "x.__class__",
            "eval('2+2')",
            "globals()",
        ]

        for expr in unsafe_expressions:
            assert is_safe_expression(expr) is False

    def test_invalid_syntax_returns_false(self) -> None:
        """Test that invalid syntax returns False."""
        invalid_expressions = [
            "2 +",
            "if True:",
            "",
            "   ",
        ]

        for expr in invalid_expressions:
            assert is_safe_expression(expr) is False


class TestSecurityEdgeCases:
    """Test security edge cases."""

    @pytest.fixture
    def evaluator(self) -> SecureExpressionEvaluator:
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_string_format_attacks(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that string format attacks are blocked."""
        unsafe_expressions = [
            "'{}'.format(globals())",
            "'%s' % globals()",
            "f'{globals()}'",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_list_comprehension_attacks(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that list comprehensions with dangerous code are blocked."""
        unsafe_expressions = [
            "[x for x in globals()]",
            "[print(x) for x in range(3)]",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_lambda_function_attacks(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that lambda functions are blocked."""
        unsafe_expressions = [
            "lambda x: x",
            "(lambda: globals())()",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_escape_sequence_handling(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test handling of escape sequences in strings."""
        # Basic string with escape sequences should be safe
        result = evaluator.evaluate("'hello\\nworld'")
        assert result == "hello\nworld"

        # But ensure no code injection via escape sequences
        with pytest.raises(InvalidParameterError):
            # This should be blocked if it tries to execute code
            evaluator.evaluate("'\\x41\\x41\\x41'.__class__")


class TestSingletonReset:
    """Test singleton reset functionality."""

    def test_reset_secure_expression_evaluator(self) -> None:
        """Test that reset_secure_expression_evaluator resets the singleton."""
        # Get the singleton instance
        evaluator1 = get_secure_expression_evaluator()
        assert evaluator1 is not None

        # Get it again - should be same instance
        evaluator2 = get_secure_expression_evaluator()
        assert evaluator1 is evaluator2

        # Reset the singleton
        reset_secure_expression_evaluator()

        # Get a new instance - should be different from the original
        evaluator3 = get_secure_expression_evaluator()
        assert evaluator3 is not None
        assert evaluator3 is not evaluator1

        # Get it again - should be same as the new instance
        evaluator4 = get_secure_expression_evaluator()
        assert evaluator3 is evaluator4

    def test_reset_functionality_works(self) -> None:
        """Test that reset evaluator still functions correctly."""
        # Get and use evaluator
        evaluator1 = get_secure_expression_evaluator()
        result1 = evaluator1.evaluate("2 + 3")
        assert result1 == 5

        # Reset
        reset_secure_expression_evaluator()

        # Get new evaluator and verify it works
        evaluator2 = get_secure_expression_evaluator()
        result2 = evaluator2.evaluate("10 * 5")
        assert result2 == 50

        # Verify original evaluator still works (not invalidated)
        result3 = evaluator1.evaluate("7 - 2")
        assert result3 == 5


class TestSafeHelperMethods:
    """Test safe helper methods for pandas operations."""

    @pytest.fixture
    def evaluator(self) -> SecureExpressionEvaluator:
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_safe_max_single_argument(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test _safe_max with single argument (array-like)."""
        import numpy as np

        # Test with list
        result = evaluator._safe_max([1, 5, 3, 2])
        assert result == 5

        # Test with numpy array
        result = evaluator._safe_max(np.array([10, 20, 15]))
        assert result == 20

    def test_safe_max_multiple_arguments(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test _safe_max with multiple arguments (element-wise)."""
        import numpy as np

        # Test element-wise maximum
        result = evaluator._safe_max(np.array([1, 5, 3]), np.array([2, 3, 4]))
        np.testing.assert_array_equal(result, [2, 5, 4])

    def test_safe_min_single_argument(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test _safe_min with single argument (array-like)."""
        import numpy as np

        # Test with list
        result = evaluator._safe_min([1, 5, 3, 2])
        assert result == 1

        # Test with numpy array
        result = evaluator._safe_min(np.array([10, 20, 15]))
        assert result == 10

    def test_safe_min_multiple_arguments(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test _safe_min with multiple arguments (element-wise)."""
        import numpy as np

        # Test element-wise minimum
        result = evaluator._safe_min(np.array([1, 5, 3]), np.array([2, 3, 4]))
        np.testing.assert_array_equal(result, [1, 3, 3])


class TestErrorHandling:
    """Test error handling in evaluation methods."""

    @pytest.fixture
    def evaluator(self) -> SecureExpressionEvaluator:
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_evaluate_undefined_variable(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test error when evaluating expression with undefined variable."""
        # Variables are allowed in syntax, but fail at evaluation time
        # Actually fails during validation since 'x' and 'y' aren't defined
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("x + y")

    def test_evaluate_general_exception(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test general exception handling in evaluate."""
        # Type errors during evaluation
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("1 / 'string'")  # Type error

    def test_evaluate_with_context_undefined_variable(
        self, evaluator: SecureExpressionEvaluator
    ) -> None:
        """Test undefined variable error with context."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # undefined_var is valid syntax but not in context
        # This will fail during validation
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate_column_expression("a + undefined_var", df)

    def test_evaluate_with_context_general_error(
        self, evaluator: SecureExpressionEvaluator
    ) -> None:
        """Test general error in column expression evaluation."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        # Division by zero - pandas handles this differently
        result = evaluator.evaluate_column_expression("a / 0", df)
        # Pandas returns inf, not an error
        import numpy as np

        assert all(np.isinf(result))

    def test_context_restoration_after_error(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that context is restored even after evaluation error."""
        # Try to evaluate with context that will fail
        context = {"bad_var": "not_a_number"}
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("bad_var + 5", context)

        # Verify evaluator still works with fresh context
        result = evaluator.evaluate("2 + 3")
        assert result == 5

    def test_invalid_expression_type(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test validation with non-string expression."""
        with pytest.raises(InvalidParameterError):
            evaluator.validate_expression_syntax(123)  # type: ignore[arg-type]

    def test_invalid_expression_none(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test validation with None expression."""
        with pytest.raises(InvalidParameterError):
            evaluator.validate_expression_syntax(None)  # type: ignore[arg-type]

    def test_invalid_function_call_structure(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test validation of complex invalid function calls."""
        # This tests the function call validation edge case
        unsafe_expr = "getattr(x, '__class__')"
        with pytest.raises(InvalidParameterError):
            evaluator.validate_expression_syntax(unsafe_expr)

    def test_attribute_access_non_np(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that non-np attribute access is blocked."""
        with pytest.raises(InvalidParameterError):
            evaluator.validate_expression_syntax("x.__class__")


class TestColumnValidation:
    """Test column validation and context handling."""

    @pytest.fixture
    def evaluator(self) -> SecureExpressionEvaluator:
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_column_not_found_with_context(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test error when column specified in context doesn't exist."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        column_context = {"x": "nonexistent_column"}

        with pytest.raises(InvalidParameterError) as exc_info:
            evaluator.evaluate_column_expression("x + 5", df, column_context)
        # Error message contains the column name
        assert "nonexistent_column" in str(exc_info.value)

    def test_scalar_result_broadcasting(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test that scalar results are broadcast to all rows."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Expression that returns a scalar
        result = evaluator.evaluate_column_expression("10", df)
        assert len(result) == len(df)
        assert all(result == 10)

    def test_array_result_conversion(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test conversion of array results to Series."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})

        # Expression using numpy functions that might return arrays
        result = evaluator.evaluate_column_expression("a * 2", df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_column_context_mapping(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test column context mapping for aliases."""
        import pandas as pd

        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})

        # Use column_context to map friendly names to actual columns
        result = evaluator.evaluate_column_expression("x + y", df, {"x": "col_a", "y": "col_b"})
        expected = pd.Series([5, 7, 9], name=None)
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    def test_evaluate_simple_formula(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test evaluate_simple_formula method."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        result = evaluator.evaluate_simple_formula("a + b", df)
        expected = pd.Series([11, 22, 33])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)


class TestStringMethodEvaluation:
    """Test string method evaluation functionality."""

    @pytest.fixture
    def evaluator(self) -> SecureExpressionEvaluator:
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_string_upper(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string upper() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.upper()", df, {"x": "name"})
        expected = pd.Series(["ALICE", "BOB", "CHARLIE"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_lower(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string lower() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["ALICE", "BOB", "CHARLIE"]})
        result = evaluator.evaluate_string_expression("x.lower()", df, {"x": "name"})
        expected = pd.Series(["alice", "bob", "charlie"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_strip(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string strip() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["  alice  ", " bob ", "charlie  "]})
        result = evaluator.evaluate_string_expression("x.strip()", df, {"x": "name"})
        expected = pd.Series(["alice", "bob", "charlie"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_strip_with_chars(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string strip() with custom characters."""
        import pandas as pd

        df = pd.DataFrame({"name": ["xxalicexx", "xxbobxx", "xxcharliexx"]})
        result = evaluator.evaluate_string_expression("x.strip('x')", df, {"x": "name"})
        expected = pd.Series(["alice", "bob", "charlie"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_title(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string title() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice smith", "bob jones", "charlie brown"]})
        result = evaluator.evaluate_string_expression("x.title()", df, {"x": "name"})
        expected = pd.Series(["Alice Smith", "Bob Jones", "Charlie Brown"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_capitalize(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string capitalize() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.capitalize()", df, {"x": "name"})
        expected = pd.Series(["Alice", "Bob", "Charlie"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_swapcase(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string swapcase() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["Alice", "BOB", "ChArLiE"]})
        result = evaluator.evaluate_string_expression("x.swapcase()", df, {"x": "name"})
        expected = pd.Series(["aLICE", "bob", "cHaRlIe"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_len(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string len() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.len()", df, {"x": "name"})
        expected = pd.Series([5, 3, 7], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_replace(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string replace() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.replace('a', 'A')", df, {"x": "name"})
        expected = pd.Series(["Alice", "bob", "chArlie"], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_contains(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string contains() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.contains('li')", df, {"x": "name"})
        expected = pd.Series([True, False, True], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_startswith(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string startswith() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.startswith('a')", df, {"x": "name"})
        expected = pd.Series([True, False, False], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_endswith(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test string endswith() method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        result = evaluator.evaluate_string_expression("x.endswith('e')", df, {"x": "name"})
        expected = pd.Series([True, False, True], name="name")
        pd.testing.assert_series_equal(result, expected)

    def test_string_method_column_not_found(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test error when string method used with nonexistent column."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob"]})
        with pytest.raises(InvalidParameterError) as exc_info:
            evaluator.evaluate_string_expression("x.upper()", df, {"x": "nonexistent"})
        assert "nonexistent" in str(exc_info.value)

    def test_unsupported_string_method(self, evaluator: SecureExpressionEvaluator) -> None:
        """Test error when using unsupported string method."""
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob"]})
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate_string_expression("x.unsupported_method()", df, {"x": "name"})

    def test_fallback_to_mathematical_expression(
        self, evaluator: SecureExpressionEvaluator
    ) -> None:
        """Test that non-string expressions fall back to math evaluation."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Even with column_context, if no string method, should do math
        result = evaluator.evaluate_string_expression("x + y", df, {"x": "a", "y": "b"})
        expected = pd.Series([5, 7, 9])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)
