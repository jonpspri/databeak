# DataBeak Server Refactoring Checklist

This checklist guides the refactoring of DataBeak modules into FastMCP servers
following established patterns and best practices.

## Pre-Refactoring Assessment

- [ ] Identify the module to be refactored
- [ ] List all functions that will become MCP tools
- [ ] Identify shared dependencies (models, exceptions, utilities)
- [ ] Review existing test coverage
- [ ] Document any breaking changes

## 1. FastMCP Tool Registration Pattern

### ✅ Correct Pattern

```python
# Define function directly (not as decorator)
async def filter_rows(
    session_id: str,
    conditions: list[FilterCondition | dict[str, Any]],
    mode: Literal["and", "or"] = "and",
    ctx: Context | None = None,
) -> FilterOperationResult:
    """Filter rows using flexible conditions."""
    # Implementation here
    pass


# Register at the end of file
server = FastMCP("Server Name")
server.tool(name="filter_rows")(filter_rows)
```

### ❌ Incorrect Pattern

```python
# Don't use decorator pattern
@server.tool
async def filter_rows(*args, **kwargs):
    pass
```

**Rationale:** Direct function definition enables unit testing without server
initialization.

## 2. Pydantic Models for Arguments and Returns

### Request Models

- [ ] Create Pydantic models for complex arguments
- [ ] Use `Field` descriptions for documentation
- [ ] Support both Pydantic models and dicts in functions

```python
class FilterCondition(BaseModel):
    """Filter condition for row filtering."""

    column: str = Field(description="Column name to filter on")
    operator: Literal["==", "!=", ">", "<"] = Field(description="Comparison operator")
    value: Any = Field(default=None, description="Value to compare against")


# Function should accept both
async def filter_rows(
    session_id: str,
    conditions: list[FilterCondition | dict[str, Any]],  # Accept both types
    ctx: Context | None = None,
) -> FilterOperationResult:
    # Handle both types
    for cond in conditions:
        if isinstance(cond, FilterCondition):
            condition = cond.model_dump()
        else:
            condition = cond
```

### Response Models

- [ ] Use typed response models from `models.tool_responses`
- [ ] Return model instances, not dicts (unless required for compatibility)

## 3. Refactor Wrapped Functions Into Tool Body

### Before (Wrapper Pattern)

```python
# server file
async def filter_rows(*args, **kwargs) -> FilterOperationResult:
    """MCP tool wrapper."""
    return await transformations.filter_rows(*args, **kwargs)


# separate transformations file
async def filter_rows(*args, **kwargs) -> FilterOperationResult:
    """Actual implementation."""
    # Business logic here
```

### After (Direct Implementation)

```python
# server file - complete implementation
async def filter_rows(
    session_id: str,
    conditions: list[FilterCondition | dict[str, Any]],
    mode: Literal["and", "or"] = "and",
    ctx: Context | None = None,
) -> FilterOperationResult:
    """Filter rows using flexible conditions.

    [Full docstring with examples]
    """
    try:
        session, df = _get_session_data(session_id)

        # Full implementation here
        mask = pd.Series([mode == "and"] * len(df))
        for cond in conditions:
            # Process conditions
            ...

        session.data_session.df = df[mask].reset_index(drop=True)

        return FilterOperationResult(
            session_id=session_id,
            rows_before=rows_before,
            rows_after=rows_after,
        )
    except Exception as e:
        raise ToolError(f"Failed to filter: {e}") from e
```

## 4. Response Model Migration

### Decision Tree

```
Is the model used by multiple servers?
├─ YES → Keep in models.tool_responses
└─ NO → Move to server module
```

### Moving Models to Server

- [ ] Check usage across codebase: `grep -r "ModelName" src/`
- [ ] If single-server use, move to server file
- [ ] Update imports in tests

```python
# In server file
class ColumnOperationResult(BaseModel):
    """Result of a column operation (local to this server)."""

    session_id: str
    operation: str
    rows_affected: int
    columns_affected: list[str]
```

### Keeping Shared Models

```python
# In models/tool_responses.py (shared across servers)
class LoadResult(BaseModel):
    """Result of loading data (used by multiple servers)."""

    session_id: str
    success: bool
    rows_affected: int
```

## 5. Exception Migration

### Decision Tree

```
Is the exception used by multiple servers?
├─ YES → Keep in exceptions.py
└─ NO → Move to server module
```

### Shared Exceptions (Keep in exceptions.py)

- [ ] `SessionNotFoundError` - used across all servers
- [ ] `NoDataLoadedError` - used across all servers
- [ ] `ColumnNotFoundError` - used by multiple column-related servers
- [ ] `InvalidParameterError` - generic validation error

### Server-Specific Exceptions

```python
# In server file
class FilterExpressionError(Exception):
    """Error in filter expression syntax (specific to this server)."""

    pass
```

## 6. Code Cleanup

### Remove Obsolete Files

- [ ] Identify wrapper functions that are no longer needed
- [ ] Check for orphaned imports
- [ ] Remove or stub out old modules for backwards compatibility

```python
# Old wrapper module - convert to stub
def register_data_tools(mcp: FastMCP) -> None:
    """Legacy function kept for backwards compatibility.

    This function is now a no-op as all tools have been migrated to specialized servers.
    """
    pass  # All tools migrated to servers
```

### Update Imports

- [ ] Update server.py to mount new servers
- [ ] Remove old tool registration calls
- [ ] Update any direct imports in examples/tests

```python
# In server.py
from .servers.transformation_server import transformation_server
from .servers.column_server import column_server

# Remove old registration
# register_data_tools(mcp)  # REMOVED

# Add new server mounting
mcp.mount(transformation_server)
mcp.mount(column_server)
```

## 7. Test Coverage Requirements

### Unit Tests

- [ ] Create `tests/unit/servers/test_{server_name}.py`
- [ ] Test all public functions
- [ ] Test parameter validation
- [ ] Test error conditions
- [ ] Test Pydantic model conversion

```python
@pytest.fixture
async def test_session() -> str:
    """Create a test session with sample data."""
    csv_content = """name,age,city
    Alice,30,NYC
    Bob,25,LA"""
    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
async def test_filter_rows_basic(test_session):
    """Test basic filtering functionality."""
    result = await filter_rows(test_session, [{"column": "age", "operator": ">", "value": 25}])
    assert result.rows_after == 1
    assert result.rows_filtered == 1
```

### Coverage Goals

- [ ] Achieve 80%+ line coverage for server module
- [ ] Test all error paths
- [ ] Test edge cases (empty data, nulls, etc.)
- [ ] Test with both Pydantic models and dicts

### Integration Tests

- [ ] Test server mounting
- [ ] Test tool discovery via MCP
- [ ] Test end-to-end workflows

## 8. Documentation

### Docstrings

- [ ] Complete docstring with description
- [ ] Document all parameters
- [ ] Include return type documentation
- [ ] Add usage examples

```python
async def filter_rows(
    session_id: str,
    conditions: list,
    mode: str = "and",
    ctx: Context | None = None,
) -> FilterOperationResult:
    """Filter rows using flexible conditions with comprehensive null value support.

    Provides powerful filtering capabilities optimized for AI-driven data analysis.

    Args:
        session_id: Session identifier for the active CSV data session
        conditions: List of filter conditions with column, operator, and value
        mode: Logic for combining conditions ("and" or "or")
        ctx: FastMCP context (optional)

    Returns:
        FilterOperationResult with filtering statistics

    Examples:
        # Numeric filtering
        filter_rows(session_id, [{"column": "age", "operator": ">", "value": 25}])

        # Text filtering with null handling
        filter_rows(session_id, [
            {"column": "name", "operator": "contains", "value": "Smith"},
            {"column": "email", "operator": "is_not_null"}
        ], mode="and")

    Raises:
        ToolError: If session not found, no data loaded, or invalid parameters
    """
```

## Post-Refactoring Validation

- [ ] All tests pass:
  `uv run -m pytest tests/unit/servers/test_{server_name}.py`
- [ ] Type checking passes: `uv run mypy src/databeak/servers/{server_name}.py`
- [ ] Linting passes: `uv run ruff check src/databeak/servers/{server_name}.py`
- [ ] Server starts successfully: `uv run databeak`
- [ ] Tools are discoverable via MCP
- [ ] No regression in existing functionality

## Example Refactoring Workflow

1. **Create new server file**: `src/databeak/servers/new_server.py`
1. **Define Pydantic models** for complex parameters
1. **Copy implementations** from wrapped functions
1. **Add error handling** with appropriate exceptions
1. **Register tools** at end of file
1. **Create test file**: `tests/unit/servers/test_new_server.py`
1. **Update server.py** to mount new server
1. **Clean up** old wrapper code
1. **Run validation** checks
1. **Commit** with descriptive message

## Common Pitfalls to Avoid

❌ Using decorator pattern for tool registration ❌ Returning dicts instead of
Pydantic models ❌ Moving shared models/exceptions to server files ❌ Forgetting
to handle both Pydantic and dict inputs ❌ Not preserving backwards compatibility
❌ Incomplete error handling migration ❌ Missing test coverage for new code paths
❌ Not updating imports in examples/documentation

## Success Criteria

✅ Server follows FastMCP patterns ✅ Functions are directly testable ✅ Pydantic
models provide type safety ✅ Implementation is self-contained ✅ Shared code
remains shared ✅ Tests achieve 80%+ coverage ✅ No regression in functionality ✅
Code is cleaner and more maintainable
