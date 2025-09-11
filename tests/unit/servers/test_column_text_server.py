"""Unit tests for column text server module.

Tests the server wrapper layer and FastMCP integration for text and string column operations.
"""

import pytest
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import src.databeak.servers.column_text_server  # noqa: F401
from src.databeak.servers.column_text_server import (
    extract_from_column,
    fill_column_nulls,
    replace_in_column,
    split_column,
    strip_column,
    transform_column_case,
)
from src.databeak.servers.io_server import load_csv_from_content


@pytest.fixture
async def text_session():
    """Create a test session with text processing data."""
    csv_content = """name,email,phone,address,description,status_code
John Doe,john@example.com,(555) 123-4567,123 Main St.   New York   NY,   Great customer!   ,ACT-001
Jane Smith,jane.smith@test.com,555-987-6543,456 Oak Ave. Los Angeles CA,good service,PEN-002
Bob Johnson,bob@company.org,(555) 111-2222,  789 Pine Rd. Chicago IL  ,EXCELLENT WORK,INA-003
Alice Brown,alice@example.com,555.444.5555,321 Elm Dr. Houston TX,   needs improvement   ,ACT-004"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestColumnTextServerReplace:
    """Test replace_in_column server function."""

    async def test_replace_with_regex(self, text_session):
        """Test regex pattern replacement."""
        result = await replace_in_column(text_session, "phone", r"[^\d]", "", regex=True)

        assert result.session_id == text_session
        assert result.operation == "replace_pattern"
        assert result.columns_affected == ["phone"]
        assert result.rows_affected == 4

    async def test_replace_literal_string(self, text_session):
        """Test literal string replacement."""
        result = await replace_in_column(text_session, "address", "St.", "Street", regex=False)

        assert result.operation == "replace_pattern"

    async def test_replace_whitespace_normalization(self, text_session):
        """Test normalizing multiple whitespace."""
        result = await replace_in_column(text_session, "address", r"\s+", " ", regex=True)

        assert result.operation == "replace_pattern"

    async def test_replace_remove_parentheses(self, text_session):
        """Test removing parentheses from phone numbers."""
        result = await replace_in_column(text_session, "phone", r"[()]", "", regex=True)

        assert result.operation == "replace_pattern"

    async def test_replace_nonexistent_column(self, text_session):
        """Test replacing in non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await replace_in_column(text_session, "nonexistent", "pattern", "replacement")


@pytest.mark.asyncio
class TestColumnTextServerExtract:
    """Test extract_from_column server function."""

    async def test_extract_email_parts(self, text_session):
        """Test extracting email username."""
        result = await extract_from_column(text_session, "email", r"(.+)@", expand=False)

        assert result.operation == "extract_pattern"
        assert result.columns_affected == ["email"]

    async def test_extract_with_expansion(self, text_session):
        """Test extracting with single group (expansion parameter test)."""
        result = await extract_from_column(text_session, "email", r"(.+)@", expand=True)

        assert result.operation == "extract_expand_1_groups"

    async def test_extract_status_code_parts(self, text_session):
        """Test extracting first part of code."""
        result = await extract_from_column(text_session, "status_code", r"([A-Z]+)", expand=True)

        assert result.operation == "extract_expand_1_groups"

    async def test_extract_single_group(self, text_session):
        """Test extracting single capturing group."""
        result = await extract_from_column(text_session, "phone", r"(\d{3})", expand=False)

        assert result.operation == "extract_pattern"

    async def test_extract_nonexistent_column(self, text_session):
        """Test extracting from non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await extract_from_column(text_session, "nonexistent", r"(\w+)")


@pytest.mark.asyncio
class TestColumnTextServerSplit:
    """Test split_column server function."""

    async def test_split_by_space_first_part(self, text_session):
        """Test splitting name by space, keeping first part."""
        result = await split_column(text_session, "name", " ", part_index=0)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))
        assert result.columns_affected == ["name"]

    async def test_split_by_space_last_part(self, text_session):
        """Test splitting name by space, keeping last part."""
        result = await split_column(text_session, "name", " ", part_index=1)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_email_by_at(self, text_session):
        """Test splitting email by @ symbol."""
        result = await split_column(text_session, "email", "@", part_index=1)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_with_expansion(self, text_session):
        """Test splitting with column expansion."""
        result = await split_column(text_session, "name", " ", expand_to_columns=True)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_with_custom_column_names(self, text_session):
        """Test splitting with custom new column names."""
        result = await split_column(
            text_session,
            "name",
            " ",
            expand_to_columns=True,
            new_columns=["first_name", "last_name"],
        )

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_address_by_period(self, text_session):
        """Test splitting address by period."""
        result = await split_column(text_session, "address", ".", part_index=0)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_nonexistent_column(self, text_session):
        """Test splitting non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await split_column(text_session, "nonexistent", " ")


@pytest.mark.asyncio
class TestColumnTextServerCase:
    """Test transform_column_case server function."""

    async def test_transform_to_upper(self, text_session):
        """Test transforming to uppercase."""
        result = await transform_column_case(text_session, "name", "upper")

        assert result.operation.startswith("case_")
        assert result.columns_affected == ["name"]

    async def test_transform_to_lower(self, text_session):
        """Test transforming to lowercase."""
        result = await transform_column_case(text_session, "email", "lower")

        assert result.operation.startswith("case_")

    async def test_transform_to_title(self, text_session):
        """Test transforming to title case."""
        result = await transform_column_case(text_session, "description", "title")

        assert result.operation.startswith("case_")

    async def test_transform_to_capitalize(self, text_session):
        """Test capitalizing first letter only."""
        result = await transform_column_case(text_session, "description", "capitalize")

        assert result.operation.startswith("case_")

    async def test_transform_case_nonexistent_column(self, text_session):
        """Test transforming case of non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await transform_column_case(text_session, "nonexistent", "upper")


@pytest.mark.asyncio
class TestColumnTextServerStrip:
    """Test strip_column server function."""

    async def test_strip_whitespace(self, text_session):
        """Test stripping whitespace."""
        result = await strip_column(text_session, "description")

        assert result.operation.startswith("strip_")
        assert result.columns_affected == ["description"]

    async def test_strip_custom_characters(self, text_session):
        """Test stripping custom characters."""
        result = await strip_column(text_session, "phone", "()")

        assert result.operation.startswith("strip_")

    async def test_strip_dots_and_spaces(self, text_session):
        """Test stripping dots and spaces."""
        result = await strip_column(text_session, "address", ". ")

        assert result.operation.startswith("strip_")

    async def test_strip_punctuation(self, text_session):
        """Test stripping punctuation from status codes."""
        result = await strip_column(text_session, "description", "!.,;")

        assert result.operation.startswith("strip_")

    async def test_strip_nonexistent_column(self, text_session):
        """Test stripping non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await strip_column(text_session, "nonexistent")


@pytest.mark.asyncio
class TestColumnTextServerFillNulls:
    """Test fill_column_nulls server function."""

    async def test_fill_nulls_with_string(self, text_session):
        """Test filling null values with string."""
        # First create some null values by splitting and not expanding
        await split_column(text_session, "description", "xyz", part_index=1)  # Will create nulls

        result = await fill_column_nulls(text_session, "description", "No description")

        assert result.operation == "fill_nulls"
        assert result.columns_affected == ["description"]

    async def test_fill_nulls_with_number(self, text_session):
        """Test filling null values with number."""
        # First add a numeric column with nulls
        from src.databeak.servers.column_server import add_column

        await add_column(text_session, "rating", value=[5, None, 4, None])

        result = await fill_column_nulls(text_session, "rating", 0)

        assert result.operation == "fill_nulls"

    async def test_fill_nulls_with_boolean(self, text_session):
        """Test filling null values with boolean."""
        from src.databeak.servers.column_server import add_column

        await add_column(text_session, "verified", value=[True, None, False, None])

        result = await fill_column_nulls(text_session, "verified", False)

        assert result.operation == "fill_nulls"

    async def test_fill_nulls_nonexistent_column(self, text_session):
        """Test filling nulls in non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await fill_column_nulls(text_session, "nonexistent", "value")


@pytest.mark.asyncio
class TestColumnTextServerErrorHandling:
    """Test error handling in column text server."""

    async def test_operations_invalid_session(self):
        """Test operations with invalid session ID."""
        invalid_session = "invalid-session-id"

        with pytest.raises(ToolError, match="not found"):
            await replace_in_column(invalid_session, "test", "pattern", "replacement")

        with pytest.raises(ToolError, match="not found"):
            await extract_from_column(invalid_session, "test", r"(\w+)")

        with pytest.raises(ToolError, match="not found"):
            await split_column(invalid_session, "test", " ")

        with pytest.raises(ToolError, match="not found"):
            await transform_column_case(invalid_session, "test", "upper")

        with pytest.raises(ToolError, match="not found"):
            await strip_column(invalid_session, "test")

        with pytest.raises(ToolError, match="not found"):
            await fill_column_nulls(invalid_session, "test", "value")


@pytest.mark.asyncio
class TestColumnTextServerComplexOperations:
    """Test complex text processing workflows."""

    async def test_clean_phone_workflow(self, text_session):
        """Test complete phone number cleaning workflow."""
        # Remove non-digits
        await replace_in_column(text_session, "phone", r"[^\d]", "", regex=True)

        # Format as (XXX) XXX-XXXX
        result = await replace_in_column(
            text_session, "phone", r"(\d{3})(\d{3})(\d{4})", r"(\1) \2-\3", regex=True
        )

        assert result.operation == "replace_pattern"

    async def test_clean_address_workflow(self, text_session):
        """Test address cleaning workflow."""
        # Strip whitespace
        await strip_column(text_session, "address")

        # Normalize multiple spaces
        await replace_in_column(text_session, "address", r"\s+", " ", regex=True)

        # Standardize abbreviations
        result = await replace_in_column(text_session, "address", "St.", "Street", regex=False)

        assert result.operation == "replace_pattern"

    async def test_extract_and_split_workflow(self, text_session):
        """Test extracting then splitting data."""
        # Extract domain from email
        await extract_from_column(text_session, "email", r"@(.+)")

        # Split domain by dots
        result = await split_column(text_session, "email", ".", part_index=0)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))
