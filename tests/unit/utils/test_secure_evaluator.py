"""Unit tests for secure evaluator."""

import pytest

from databeak.exceptions import InvalidParameterError
from databeak.utils.secure_evaluator import (
    SecureExpressionEvaluator,
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
    def evaluator(self):
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_safe_arithmetic_expressions(self, evaluator):
        """Test safe arithmetic expressions."""
        # Basic arithmetic
        assert evaluator.evaluate("2 + 3") == 5
        assert evaluator.evaluate("10 - 4") == 6
        assert evaluator.evaluate("6 * 7") == 42
        assert evaluator.evaluate("15 / 3") == 5.0

    def test_safe_comparison_expressions(self, evaluator):
        """Test safe comparison expressions."""
        assert evaluator.evaluate("5 > 3") is True
        assert evaluator.evaluate("2 < 1") is False
        assert evaluator.evaluate("4 == 4") is True
        assert evaluator.evaluate("3 != 5") is True
        assert evaluator.evaluate("7 >= 7") is True
        assert evaluator.evaluate("2 <= 1") is False

    def test_safe_logical_expressions(self, evaluator):
        """Test safe logical expressions."""
        assert evaluator.evaluate("True and False") is False
        assert evaluator.evaluate("True or False") is True
        assert evaluator.evaluate("not False") is True
        assert evaluator.evaluate("(5 > 3) and (2 < 4)") is True

    def test_safe_variable_access(self, evaluator):
        """Test safe variable access with context."""
        context = {"x": 10, "y": 5, "name": "test"}

        assert evaluator.evaluate("x + y", context) == 15
        assert evaluator.evaluate("x > y", context) is True
        assert evaluator.evaluate("name == 'test'", context) is True

    def test_safe_string_operations(self, evaluator):
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

    def test_blocked_function_calls(self, evaluator):
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

    def test_blocked_attribute_access(self, evaluator):
        """Test that attribute access is blocked."""
        unsafe_expressions = [
            "''.__class__",
            "x.__dict__",
            "object.__subclasses__()",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_blocked_import_statements(self, evaluator):
        """Test that import statements are blocked."""
        unsafe_expressions = [
            "import os",
            "from os import path",
            "__import__('sys')",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_blocked_dangerous_operations(self, evaluator):
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

    def test_division_by_zero_handling(self, evaluator):
        """Test division by zero handling."""
        with pytest.raises(ZeroDivisionError):
            evaluator.evaluate("10 / 0")

    def test_syntax_error_handling(self, evaluator):
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

    def test_complex_safe_expressions(self, evaluator):
        """Test complex but safe expressions."""
        context = {"a": 5, "b": 10, "c": 3}

        # Complex arithmetic
        result = evaluator.evaluate("(a + b) * c - 2", context)
        assert result == 43  # (5 + 10) * 3 - 2 = 43

        # Complex comparisons
        result = evaluator.evaluate("(a > 3) and (b < 15) and (c == 3)", context)
        assert result is True

    def test_none_and_null_handling(self, evaluator):
        """Test None and null value handling."""
        context = {"value": None, "number": 42}

        # None comparisons
        assert evaluator.evaluate("value is None", context) is True
        assert evaluator.evaluate("number is not None", context) is True

    def test_empty_expression(self, evaluator):
        """Test empty expression handling."""
        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("")

        with pytest.raises(InvalidParameterError):
            evaluator.evaluate("   ")  # Whitespace only

    def test_very_large_numbers(self, evaluator):
        """Test handling of very large numbers."""
        # Should handle large numbers without issues
        result = evaluator.evaluate("999999999999 + 1")
        assert result == 1000000000000

    def test_nested_expressions(self, evaluator):
        """Test deeply nested expressions."""
        context = {"x": 2}

        # Nested parentheses
        result = evaluator.evaluate("((x + 1) * (x + 2)) + ((x - 1) * (x - 2))", context)
        # ((2+1) * (2+2)) + ((2-1) * (2-2)) = (3*4) + (1*0) = 12 + 0 = 12
        assert result == 12


class TestValidateExpression:
    """Test validate_expression function."""

    def test_validate_safe_expressions(self):
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

    def test_validate_unsafe_expressions(self):
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

    def test_validate_empty_expression(self):
        """Test validation of empty expressions."""
        with pytest.raises(InvalidParameterError):
            validate_expression_safety("")


class TestIsSafeExpression:
    """Test is_safe_expression function."""

    def test_safe_expressions_return_true(self):
        """Test that safe expressions return True."""
        safe_expressions = [
            "2 + 3",
            "x > 5",
            "name == 'test'",
            "(a and b) or c",
        ]

        for expr in safe_expressions:
            assert is_safe_expression(expr) is True

    def test_unsafe_expressions_return_false(self):
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

    def test_invalid_syntax_returns_false(self):
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
    def evaluator(self):
        """Create secure evaluator instance."""
        return SecureExpressionEvaluator()

    def test_string_format_attacks(self, evaluator):
        """Test that string format attacks are blocked."""
        unsafe_expressions = [
            "'{}'.format(globals())",
            "'%s' % globals()",
            "f'{globals()}'",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_list_comprehension_attacks(self, evaluator):
        """Test that list comprehensions with dangerous code are blocked."""
        unsafe_expressions = [
            "[x for x in globals()]",
            "[print(x) for x in range(3)]",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_lambda_function_attacks(self, evaluator):
        """Test that lambda functions are blocked."""
        unsafe_expressions = [
            "lambda x: x",
            "(lambda: globals())()",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(InvalidParameterError):
                evaluator.evaluate(expr)

    def test_escape_sequence_handling(self, evaluator):
        """Test handling of escape sequences in strings."""
        # Basic string with escape sequences should be safe
        result = evaluator.evaluate("'hello\\nworld'")
        assert result == "hello\nworld"

        # But ensure no code injection via escape sequences
        with pytest.raises(InvalidParameterError):
            # This should be blocked if it tries to execute code
            evaluator.evaluate("'\\x41\\x41\\x41'.__class__")
