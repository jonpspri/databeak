"""Patch jsonschema to support integer values represented as strings or floats."""

from typing import Any

from jsonschema import validators
from jsonschema._types import TypeChecker, is_integer
from jsonschema.validators import validator_for


def _is_integer(checker: TypeChecker, instance: Any) -> bool:
    if is_integer(checker, instance):
        return True
    if isinstance(instance, str):
        try:
            int(instance)
            return True
        except ValueError:
            pass
    return isinstance(instance, float) and instance.is_integer()


def _do_it() -> None:
    validator = validator_for({})
    llm_type_checker = validator.TYPE_CHECKER.redefine("integer", _is_integer)

    llm_validator = validators.extend(
        validator,
        type_checker=llm_type_checker,
    )

    # Register the custom validator to handle the specific JSON schema URI used by this application.
    validators.validates(validator.META_SCHEMA["$schema"])(llm_validator)

_do_it()
