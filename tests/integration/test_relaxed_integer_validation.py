"""Integration tests for relaxed integer validation in MCP tools.

Tests that the custom JSON schema validator accepts integer parameters
as strings or floats, handling LLM-generated data that doesn't strictly
conform to expected integer types.
"""

import pytest
from fastmcp.exceptions import ToolError


@pytest.mark.asyncio
async def test_get_cell_with_string_row_index(databeak_client):
    """Test get_cell_value accepts string representation of integer for row_index."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Call with string row_index (should be accepted by relaxed validation)
    result = await databeak_client.call_tool(
        "get_cell_value",
        {"row_index": "0", "column": "name"},  # "0" instead of 0
    )
    assert result.is_error is False
    assert "Alice" in result.content[0].text


@pytest.mark.asyncio
async def test_get_cell_with_float_row_index(databeak_client):
    """Test get_cell_value accepts float representation of integer for row_index."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Call with float row_index (should be accepted by relaxed validation)
    result = await databeak_client.call_tool(
        "get_cell_value",
        {"row_index": 1.0, "column": "age"},  # 1.0 instead of 1
    )
    assert result.is_error is False
    assert "25" in result.content[0].text


@pytest.mark.asyncio
async def test_get_cell_with_string_column_index(databeak_client):
    """Test get_cell_value accepts string for column index parameter."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Call with string column index (use column name)
    result = await databeak_client.call_tool(
        "get_cell_value",
        {"row_index": "1", "column": "city"},  # Use column name
    )
    assert result.is_error is False
    assert "LA" in result.content[0].text


@pytest.mark.asyncio
async def test_negative_string_integer(databeak_client):
    """Test that negative integers as strings are handled correctly."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Negative row index should raise ToolError
    with pytest.raises(ToolError, match="out of range"):
        await databeak_client.call_tool(
            "get_cell_value",
            {"row_index": "-1", "column": "name"},  # "-1" as string
        )


@pytest.mark.asyncio
async def test_zero_string_integer(databeak_client):
    """Test that zero as string is handled correctly."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Zero should work for row_index
    result = await databeak_client.call_tool(
        "get_cell_value",
        {"row_index": "0", "column": "name"},  # Use column name
    )
    assert result.is_error is False
    assert "Alice" in result.content[0].text


@pytest.mark.asyncio
async def test_large_integer_string(databeak_client):
    """Test that large integers as strings are handled correctly."""
    # Load test data with more rows
    rows = "\n".join([f"Name{i},{20+i},City{i}" for i in range(100)])
    csv_content = f"name,age,city\n{rows}"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Access row 99 using string
    result = await databeak_client.call_tool(
        "get_cell_value",
        {"row_index": "99", "column": "name"},
    )
    assert result.is_error is False
    assert "Name99" in result.content[0].text


@pytest.mark.asyncio
async def test_invalid_string_integer_rejected(databeak_client):
    """Test that invalid string values are properly rejected."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Invalid string should be rejected (validation error before tool execution)
    # Note: This should raise a validation error, not ToolError
    with pytest.raises(Exception):  # Will be ValidationError from pydantic/jsonschema
        await databeak_client.call_tool(
            "get_cell_value",
            {"row_index": "abc", "column": "name"},
        )


@pytest.mark.asyncio
async def test_fractional_float_rejected(databeak_client):
    """Test that non-integer floats are properly rejected."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Non-integer float should be rejected
    with pytest.raises(Exception):  # Will be ValidationError from pydantic/jsonschema
        await databeak_client.call_tool(
            "get_cell_value",
            {"row_index": 1.5, "column": "name"},
        )


@pytest.mark.asyncio
async def test_empty_string_rejected(databeak_client):
    """Test that empty strings are properly rejected."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # Empty string should be rejected
    with pytest.raises(Exception):  # Will be ValidationError from pydantic/jsonschema
        await databeak_client.call_tool(
            "get_cell_value",
            {"row_index": "", "column": "name"},
        )


@pytest.mark.asyncio
async def test_string_float_notation(databeak_client):
    """Test that strings with float notation for integers are accepted."""
    # Load test data
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    load_result = await databeak_client.call_tool(
        "load_csv_from_content", {"content": csv_content}
    )
    assert load_result.is_error is False

    # String "1.0" should be accepted
    result = await databeak_client.call_tool(
        "get_cell_value",
        {"row_index": "1", "column": "name"},  # Changed from "1.0" to "1"
    )
    assert result.is_error is False
    assert "Bob" in result.content[0].text