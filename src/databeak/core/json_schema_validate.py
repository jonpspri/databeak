"""Patch jsonschema to support integer values represented as strings or floats."""

from typing import TYPE_CHECKING, Any

from jsonschema import validators
from jsonschema._types import TypeChecker, is_integer
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

if TYPE_CHECKING:
    from jsonschema.validators import _Validator
else:
    _Validator = Any  # type: ignore[assignment]


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


_llm_validator: Validator | None = None


def initialize_relaxed_validation() -> None:
    """Initialize a custom JSON schema validator that accepts integers as strings or floats."""
    default_validator = validator_for({})
    llm_type_checker = default_validator.TYPE_CHECKER.redefine("integer", _is_integer)

    global _llm_validator  # noqa: PLW0603
    _llm_validator = validators.extend(default_validator, type_checker=llm_type_checker)

    # Patch _LATEST_VERSION to use our custom validator
    # NOTE: We are accessing the private attribute `_LATEST_VERSION` of the `jsonschema.validators` module.
    # This is necessary because there is currently no public API to globally override the default validator
    # used by `jsonschema`. This patch ensures that our custom validator (which accepts integers as strings or floats)
    # is used throughout the codebase. If a public API becomes available in future versions of `jsonschema`,
    # this code should be updated to use it. Future maintainers: check the upstream library for changes before upgrading.
    validators._LATEST_VERSION = _llm_validator  # type: ignore[attr-defined]  # noqa: SLF001
