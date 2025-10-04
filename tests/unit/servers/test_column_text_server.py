"""Unit tests for column text server module.

Tests the server wrapper layer and FastMCP integration for text and string column operations.
"""

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import databeak.servers.column_text_server  # noqa: F401
from databeak.servers.column_text_server import (
    extract_from_column,
    fill_column_nulls,
    replace_in_column,
    split_column,
    strip_column,
    transform_column_case,
)
from databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context


@pytest.fixture
async def text_session_ctx() -> Context:
    """Create a test session with text processing data."""
    csv_content = """name,email,phone,address,description,status_code
John Doe,john@example.com,(555) 123-4567,123 Main St.   New York   NY,   Great customer!   ,ACT-001
Jane Smith,jane.smith@test.com,555-987-6543,456 Oak Ave. Los Angeles CA,good service,PEN-002
Bob Johnson,bob@company.org,(555) 111-2222,  789 Pine Rd. Chicago IL  ,EXCELLENT WORK,INA-003
Alice Brown,alice@example.com,555.444.5555,321 Elm Dr. Houston TX,   needs improvement   ,ACT-004"""

    ctx = create_mock_context()
    _result = await load_csv_from_content(ctx, csv_content)
    return ctx


@pytest.mark.asyncio
class TestColumnTextServerReplace:
    """Test replace_in_column server function."""

    async def test_replace_with_regex(self, text_session_ctx: Context) -> None:
        """Test regex pattern replacement."""
        ctx = text_session_ctx
        result = await replace_in_column(ctx, "phone", r"[^\d]", "", regex=True)

        assert result.operation == "replace_pattern"
        assert result.columns_affected == ["phone"]
        assert result.rows_affected == 4

    async def test_replace_literal_string(self, text_session_ctx: Context) -> None:
        """Test literal string replacement."""
        ctx = text_session_ctx
        result = await replace_in_column(ctx, "address", "St.", "Street", regex=False)

        assert result.operation == "replace_pattern"

    async def test_replace_whitespace_normalization(self, text_session_ctx: Context) -> None:
        """Test normalizing multiple whitespace."""
        ctx = text_session_ctx
        result = await replace_in_column(ctx, "address", r"\s+", " ", regex=True)

        assert result.operation == "replace_pattern"

    async def test_replace_remove_parentheses(self, text_session_ctx: Context) -> None:
        """Test removing parentheses from phone numbers."""
        ctx = text_session_ctx
        result = await replace_in_column(ctx, "phone", r"[()]", "", regex=True)

        assert result.operation == "replace_pattern"

    async def test_replace_nonexistent_column(self, text_session_ctx: Context) -> None:
        """Test replacing in non-existent column."""
        ctx = text_session_ctx
        with pytest.raises(ToolError, match="Column.*not found"):
            await replace_in_column(ctx, "nonexistent", "pattern", "replacement")


@pytest.mark.asyncio
class TestColumnTextServerExtract:
    """Test extract_from_column server function."""

    async def test_extract_email_parts(self, text_session_ctx: Context) -> None:
        """Test extracting email username."""
        ctx = text_session_ctx
        result = await extract_from_column(ctx, "email", r"(.+)@", expand=False)

        assert result.operation == "extract_pattern"
        assert result.columns_affected == ["email"]

    async def test_extract_with_expansion(self, text_session_ctx: Context) -> None:
        """Test extracting with single group (expansion parameter test)."""
        ctx = text_session_ctx
        result = await extract_from_column(ctx, "email", r"(.+)@", expand=True)

        assert result.operation == "extract_expand_1_groups"

    async def test_extract_status_code_parts(self, text_session_ctx: Context) -> None:
        """Test extracting first part of code."""
        ctx = text_session_ctx
        result = await extract_from_column(ctx, "status_code", r"([A-Z]+)", expand=True)

        assert result.operation == "extract_expand_1_groups"

    async def test_extract_single_group(self, text_session_ctx: Context) -> None:
        """Test extracting single capturing group."""
        ctx = text_session_ctx
        result = await extract_from_column(ctx, "phone", r"(\d{3})", expand=False)

        assert result.operation == "extract_pattern"

    async def test_extract_nonexistent_column(self, text_session_ctx: Context) -> None:
        """Test extracting from non-existent column."""
        ctx = text_session_ctx

        with pytest.raises(ToolError, match="Column.*not found"):
            await extract_from_column(ctx, "nonexistent", r"(\w+)")


@pytest.mark.asyncio
class TestColumnTextServerSplit:
    """Test split_column server function."""

    async def test_split_by_space_first_part(self, text_session_ctx: Context) -> None:
        """Test splitting name by space, keeping first part."""
        ctx = text_session_ctx
        result = await split_column(ctx, "name", " ", part_index=0)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))
        assert result.columns_affected == ["name"]

    async def test_split_by_space_last_part(self, text_session_ctx: Context) -> None:
        """Test splitting name by space, keeping last part."""
        ctx = text_session_ctx
        result = await split_column(ctx, "name", " ", part_index=1)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_email_by_at(self, text_session_ctx: Context) -> None:
        """Test splitting email by @ symbol."""
        ctx = text_session_ctx
        result = await split_column(ctx, "email", "@", part_index=1)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_with_expansion(self, text_session_ctx: Context) -> None:
        """Test splitting with column expansion."""
        ctx = text_session_ctx
        result = await split_column(ctx, "name", " ", expand_to_columns=True)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_with_custom_column_names(self, text_session_ctx: Context) -> None:
        """Test splitting with custom new column names."""
        ctx = text_session_ctx
        result = await split_column(
            ctx,
            "name",
            " ",
            expand_to_columns=True,
            new_columns=["first_name", "last_name"],
        )

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_address_by_period(self, text_session_ctx: Context) -> None:
        """Test splitting address by period."""
        ctx = text_session_ctx
        result = await split_column(ctx, "address", ".", part_index=0)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))

    async def test_split_nonexistent_column(self, text_session_ctx: Context) -> None:
        """Test splitting non-existent column."""
        ctx = text_session_ctx

        with pytest.raises(ToolError, match="Column.*not found"):
            await split_column(ctx, "nonexistent", " ")


@pytest.mark.asyncio
class TestColumnTextServerCase:
    """Test transform_column_case server function."""

    async def test_transform_to_upper(self, text_session_ctx: Context) -> None:
        """Test transforming to uppercase."""
        ctx = text_session_ctx
        result = await transform_column_case(ctx, "name", "upper")

        assert result.operation.startswith("case_")
        assert result.columns_affected == ["name"]

    async def test_transform_to_lower(self, text_session_ctx: Context) -> None:
        """Test transforming to lowercase."""
        ctx = text_session_ctx
        result = await transform_column_case(ctx, "email", "lower")

        assert result.operation.startswith("case_")

    async def test_transform_to_title(self, text_session_ctx: Context) -> None:
        """Test transforming to title case."""
        ctx = text_session_ctx
        result = await transform_column_case(ctx, "description", "title")

        assert result.operation.startswith("case_")

    async def test_transform_to_capitalize(self, text_session_ctx: Context) -> None:
        """Test capitalizing first letter only."""
        ctx = text_session_ctx
        result = await transform_column_case(ctx, "description", "capitalize")

        assert result.operation.startswith("case_")

    async def test_transform_case_nonexistent_column(self, text_session_ctx: Context) -> None:
        """Test transforming case of non-existent column."""
        ctx = text_session_ctx

        with pytest.raises(ToolError, match="Column.*not found"):
            await transform_column_case(ctx, "nonexistent", "upper")


@pytest.mark.asyncio
class TestColumnTextServerStrip:
    """Test strip_column server function."""

    async def test_strip_whitespace(self, text_session_ctx: Context) -> None:
        """Test stripping whitespace."""
        ctx = text_session_ctx
        result = await strip_column(ctx, "description")

        assert result.operation.startswith("strip_")
        assert result.columns_affected == ["description"]

    async def test_strip_custom_characters(self, text_session_ctx: Context) -> None:
        """Test stripping custom characters."""
        ctx = text_session_ctx
        result = await strip_column(ctx, "phone", "()")

        assert result.operation.startswith("strip_")

    async def test_strip_dots_and_spaces(self, text_session_ctx: Context) -> None:
        """Test stripping dots and spaces."""
        ctx = text_session_ctx
        result = await strip_column(ctx, "address", ". ")

        assert result.operation.startswith("strip_")

    async def test_strip_punctuation(self, text_session_ctx: Context) -> None:
        """Test stripping punctuation from status codes."""
        ctx = text_session_ctx
        result = await strip_column(ctx, "description", "!.,;")

        assert result.operation.startswith("strip_")

    async def test_strip_nonexistent_column(self, text_session_ctx: Context) -> None:
        """Test stripping non-existent column."""
        ctx = text_session_ctx

        with pytest.raises(ToolError, match="Column.*not found"):
            await strip_column(ctx, "nonexistent")


@pytest.mark.asyncio
class TestColumnTextServerFillNulls:
    """Test fill_column_nulls server function."""

    async def test_fill_nulls_with_string(self, text_session_ctx: Context) -> None:
        """Test filling null values with string."""
        # First create some null values by splitting and not expanding
        ctx = text_session_ctx
        await split_column(ctx, "description", "xyz", part_index=1)  # Will create nulls

        result = await fill_column_nulls(ctx, "description", "No description")

        assert result.operation == "fill_nulls"
        assert result.columns_affected == ["description"]

    async def test_fill_nulls_with_number(self, text_session_ctx: Context) -> None:
        """Test filling null values with number."""
        # First add a numeric column with nulls
        from databeak.servers.column_server import add_column

        ctx = text_session_ctx
        await add_column(ctx, "rating", value=[5, None, 4, None])

        result = await fill_column_nulls(ctx, "rating", 0)

        assert result.operation == "fill_nulls"

    async def test_fill_nulls_with_boolean(self, text_session_ctx: Context) -> None:
        """Test filling null values with boolean."""
        from databeak.servers.column_server import add_column

        ctx = text_session_ctx
        await add_column(ctx, "verified", value=[True, None, False, None])

        result = await fill_column_nulls(ctx, "verified", value=False)

        assert result.operation == "fill_nulls"

    async def test_fill_nulls_nonexistent_column(self, text_session_ctx: Context) -> None:
        """Test filling nulls in non-existent column."""
        ctx = text_session_ctx

        with pytest.raises(ToolError, match="Column.*not found"):
            await fill_column_nulls(ctx, "nonexistent", "value")


@pytest.mark.asyncio
class TestColumnTextServerErrorHandling:
    """Test error handling in column text server."""

    async def test_operations_invalid_session(self) -> None:
        """Test operations with invalid session ID."""
        invalid_session = "invalid-session-id"

        ctx = create_mock_context(invalid_session)
        with pytest.raises(ToolError, match="No data loaded in session"):
            await replace_in_column(ctx, "test", "pattern", "replacement")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await extract_from_column(ctx, "test", r"(\w+)")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await split_column(ctx, "test", " ")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await transform_column_case(ctx, "test", "upper")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await strip_column(ctx, "test")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await fill_column_nulls(ctx, "test", "value")


@pytest.mark.asyncio
class TestColumnTextServerComplexOperations:
    """Test complex text processing workflows."""

    async def test_clean_phone_workflow(self, text_session_ctx: Context) -> None:
        """Test complete phone number cleaning workflow."""
        # Remove non-digits
        ctx = text_session_ctx
        await replace_in_column(ctx, "phone", r"[^\d]", "", regex=True)

        # Format as (XXX) XXX-XXXX
        result = await replace_in_column(
            ctx,
            "phone",
            r"(\d{3})(\d{3})(\d{4})",
            r"(\1) \2-\3",
            regex=True,
        )

        assert result.operation == "replace_pattern"

    async def test_clean_address_workflow(self, text_session_ctx: Context) -> None:
        """Test address cleaning workflow."""
        # Strip whitespace
        ctx = text_session_ctx
        await strip_column(ctx, "address")

        # Normalize multiple spaces
        await replace_in_column(ctx, "address", r"\s+", " ", regex=True)

        # Standardize abbreviations
        result = await replace_in_column(ctx, "address", "St.", "Street", regex=False)

        assert result.operation == "replace_pattern"

    async def test_extract_and_split_workflow(self, text_session_ctx: Context) -> None:
        """Test extracting then splitting data."""
        # Extract domain from email
        ctx = text_session_ctx
        await extract_from_column(ctx, "email", r"@(.+)")

        # Split domain by dots
        result = await split_column(ctx, "email", ".", part_index=0)

        assert result.operation.startswith(("split_keep_part_", "split_expand_"))
