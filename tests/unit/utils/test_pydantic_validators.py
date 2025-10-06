"""Tests for shared Pydantic validators."""

import json

import pytest

from databeak.utils.pydantic_validators import (
    parse_json_string_to_dict,
    parse_json_string_to_dict_or_list,
    parse_json_string_to_list,
)


class TestParseJsonStringToDict:
    """Test parse_json_string_to_dict validator."""

    def test_valid_dict_passthrough(self) -> None:
        """Test that valid dictionaries pass through unchanged."""
        test_dict = {"name": "John", "age": 30}
        result = parse_json_string_to_dict(test_dict)
        assert result == test_dict

    def test_valid_json_string_to_dict(self) -> None:
        """Test parsing valid JSON string to dictionary."""
        json_string = '{"name": "Jane", "age": 25}'
        expected = {"name": "Jane", "age": 25}
        result = parse_json_string_to_dict(json_string)
        assert result == expected

    def test_invalid_json_string(self) -> None:
        """Test error handling for invalid JSON string."""
        invalid_json = '{"name": "John", "age":}'
        with pytest.raises(ValueError):
            parse_json_string_to_dict(invalid_json)

    def test_json_string_not_dict(self) -> None:
        """Test error when JSON string parses to non-dict."""
        json_list = '["item1", "item2"]'
        with pytest.raises(TypeError, match="JSON string must parse to dict"):
            parse_json_string_to_dict(json_list)

    def test_complex_dict_json(self) -> None:
        """Test parsing complex dictionary from JSON."""
        complex_dict = {
            "user": {"name": "John", "details": {"age": 30, "active": True}},
            "scores": [95, 87, 92],
            "metadata": None,
        }
        json_string = json.dumps(complex_dict)
        result = parse_json_string_to_dict(json_string)
        assert result == complex_dict


class TestParseJsonStringToDictOrList:
    """Test parse_json_string_to_dict_or_list validator."""

    def test_valid_dict_passthrough(self) -> None:
        """Test that valid dictionaries pass through unchanged."""
        test_dict = {"name": "John", "age": 30}
        result = parse_json_string_to_dict_or_list(test_dict)
        assert result == test_dict

    def test_valid_list_passthrough(self) -> None:
        """Test that valid lists pass through unchanged."""
        test_list = ["John", 30, True]
        result = parse_json_string_to_dict_or_list(test_list)
        assert result == test_list

    def test_json_string_to_dict(self) -> None:
        """Test parsing JSON string to dictionary."""
        json_string = '{"name": "Jane", "age": 25}'
        expected = {"name": "Jane", "age": 25}
        result = parse_json_string_to_dict_or_list(json_string)
        assert result == expected

    def test_json_string_to_list(self) -> None:
        """Test parsing JSON string to list."""
        json_string = '["John", 30, true, null]'
        expected = ["John", 30, True, None]
        result = parse_json_string_to_dict_or_list(json_string)
        assert result == expected

    def test_invalid_json_string(self) -> None:
        """Test error handling for invalid JSON string."""
        invalid_json = '{"name": "John", "age":}'
        with pytest.raises(ValueError):
            parse_json_string_to_dict_or_list(invalid_json)

    def test_json_string_invalid_type(self) -> None:
        """Test error when JSON string parses to unsupported type."""
        json_string = '"just a string"'
        with pytest.raises(TypeError, match="JSON string must parse to dict or list"):
            parse_json_string_to_dict_or_list(json_string)

    def test_json_number_invalid(self) -> None:
        """Test error when JSON string parses to number."""
        json_string = "42"
        with pytest.raises(TypeError, match="JSON string must parse to dict or list"):
            parse_json_string_to_dict_or_list(json_string)


class TestParseJsonStringToList:
    """Test parse_json_string_to_list validator."""

    def test_valid_list_passthrough(self) -> None:
        """Test that valid lists pass through unchanged."""
        test_list = ["John", 30, True, None]
        result = parse_json_string_to_list(test_list)
        assert result == test_list

    def test_json_string_to_list(self) -> None:
        """Test parsing JSON string to list."""
        json_string = '["John", 30, true, null]'
        expected = ["John", 30, True, None]
        result = parse_json_string_to_list(json_string)
        assert result == expected

    def test_invalid_json_string(self) -> None:
        """Test error handling for invalid JSON string."""
        invalid_json = '["John", 30,]'  # Trailing comma
        with pytest.raises(ValueError):
            parse_json_string_to_list(invalid_json)

    def test_json_string_not_list(self) -> None:
        """Test error when JSON string parses to non-list."""
        json_dict = '{"name": "John"}'
        with pytest.raises(TypeError, match="JSON string must parse to list"):
            parse_json_string_to_list(json_dict)

    def test_nested_list_json(self) -> None:
        """Test parsing nested list structure from JSON."""
        nested_list = [["John", 30], ["Jane", 25], ["Bob", 35]]
        json_string = json.dumps(nested_list)
        result = parse_json_string_to_list(json_string)
        assert result == nested_list

    def test_empty_list_json(self) -> None:
        """Test parsing empty list from JSON."""
        json_string = "[]"
        result = parse_json_string_to_list(json_string)
        assert result == []


class TestValidatorEdgeCases:
    """Test edge cases and error conditions for all validators."""

    def test_empty_json_objects(self) -> None:
        """Test handling of empty JSON objects and arrays."""
        # Empty dict
        result = parse_json_string_to_dict("{}")
        assert result == {}

        # Empty list
        result_list = parse_json_string_to_list("[]")
        assert result_list == []

        # Empty for dict_or_list
        result_dict = parse_json_string_to_dict_or_list("{}")
        assert result_dict == {}
        result_list_or_dict = parse_json_string_to_dict_or_list("[]")
        assert result_list_or_dict == []

    def test_null_json_values(self) -> None:
        """Test handling of null values in JSON."""
        # Dict with null values
        json_string = '{"name": null, "age": 30}'
        result = parse_json_string_to_dict(json_string)
        assert result == {"name": None, "age": 30}

        # List with null values
        json_string = '[null, "John", null]'
        result_list = parse_json_string_to_list(json_string)
        assert result_list == [None, "John", None]

    def test_unicode_handling(self) -> None:
        """Test handling of unicode characters in JSON."""
        unicode_dict = {"name": "José García", "city": "São Paulo"}
        json_string = json.dumps(unicode_dict, ensure_ascii=False)
        result = parse_json_string_to_dict(json_string)
        assert result == unicode_dict

    def test_numeric_values_preservation(self) -> None:
        """Test that numeric types are preserved correctly."""
        test_data = {
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "null": None,
            "zero": 0,
            "negative": -100,
        }
        json_string = json.dumps(test_data)
        result = parse_json_string_to_dict(json_string)
        assert result == test_data
        assert isinstance(result["integer"], int)
        assert isinstance(result["float"], float)
        assert isinstance(result["boolean"], bool)
        assert result["null"] is None
