"""Unit tests for JSON schema validation with relaxed integer type checking."""

import pytest
from jsonschema import ValidationError, validate
from jsonschema.validators import validator_for

from databeak.core.json_schema_validate import initialize_relaxed_validation


def test_relaxed_integer_validation_string_integers() -> None:
    """Test that string integers are accepted after initialization."""
    initialize_relaxed_validation()

    schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

    # String integer should be valid
    validate(instance={"age": "42"}, schema=schema)

    # Regular integer should still work
    validate(instance={"age": 42}, schema=schema)


def test_relaxed_integer_validation_float_integers() -> None:
    """Test that float integers (like 42.0) are accepted after initialization."""
    initialize_relaxed_validation()

    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}

    # Float with zero decimal should be valid
    validate(instance={"count": 42.0}, schema=schema)

    # Regular integer should still work
    validate(instance={"count": 42}, schema=schema)


def test_relaxed_integer_validation_rejects_non_integers() -> None:
    """Test that non-integer strings and floats are still rejected."""
    initialize_relaxed_validation()

    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}

    # Non-integer string should fail
    with pytest.raises(ValidationError):
        validate(instance={"value": "not_a_number"}, schema=schema)

    # Float with decimal should fail
    with pytest.raises(ValidationError):
        validate(instance={"value": 42.5}, schema=schema)


def test_relaxed_validation_preserves_standard_types() -> None:
    """Test that other JSON schema types still work correctly."""
    initialize_relaxed_validation()

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "active": {"type": "boolean"},
            "score": {"type": "number"},
        },
    }

    # Valid data
    validate(
        instance={"name": "test", "active": True, "score": 95.5},
        schema=schema,
    )

    # Invalid types should still fail
    with pytest.raises(ValidationError):
        validate(instance={"name": 123}, schema=schema)


def test_validator_registration() -> None:
    """Test that the custom validator is properly registered."""
    initialize_relaxed_validation()

    # Get the validator for an empty schema
    validator_class = validator_for({})

    # Verify it has the custom type checker
    type_checker = validator_class.TYPE_CHECKER

    # Test the integer type checker directly
    assert type_checker.is_type("42", "integer")
    assert type_checker.is_type(42, "integer")
    assert type_checker.is_type(42.0, "integer")
    assert not type_checker.is_type(42.5, "integer")
    assert not type_checker.is_type("not_a_number", "integer")
