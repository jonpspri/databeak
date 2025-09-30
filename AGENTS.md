# Claude Code Instructions for DataBeak Development

## Project Context

DataBeak is a Model Context Protocol (MCP) server providing 40+ tools for CSV
data manipulation. Built with FastMCP, Pandas, and modern Python tooling.

**Key Technologies**: FastMCP, Pandas, Pydantic, uv package manager

## Critical Development Rules

### Git Workflow (ENFORCED BY PRE-COMMIT)

**NEVER commit directly to `main`** - Pre-commit hooks will reject.

- Always create feature branches: `feature/`, `fix/`, `docs/`, `test/`,
  `refactor/`
- All changes to `main` must go through Pull Requests
- Standard workflow:
  1. `git checkout -b feature/name`
  1. Make changes and commit
  1. `git push -u origin feature/name`
  1. `gh pr create --title "..." --body "..."`
  1. After PR merge: cleanup branches

**⚠️ CRITICAL**: Only cleanup branches AFTER confirming PR merge via GitHub UI.

### Package Management

**Use `uv` exclusively** (not pip or poetry):

- `uv sync` - Install dependencies
- `uv run -m pytest` - Run tests
- `uv add <package>` or `uv add --dev <package>` - Add dependencies

### Code Quality (Zero Tolerance)

All enforced via pre-commit hooks - see docs/quality.md for details:

- **Zero ruff violations** - 46 rules enabled
- **100% MyPy compliance** - Strict type checking
- **No Args sections** in MCP tool docstrings (use Field descriptions)
- **80%+ test coverage** - Required minimum

**Quick check**: `uv run pre-commit run --all-files`

**Auto-fix**:
`uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/`

## Core Coding Patterns

### Defensive Programming: Session Access

**ALWAYS use centralized helpers** from `session_utils`:

```python
from databeak.utils.session_utils import get_session_data


def my_mcp_tool(ctx: Context) -> Result:
    session_id = ctx.session_id
    session, df = get_session_data(session_id)  # Safe, validated access
    # ... process df safely
```

**Available helpers**:

- `get_session_data(session_id)` - Returns (session, df) with validation
- `get_session_only(session_id)` - Returns session without requiring data
- `validate_session_has_data(session, session_id)` - Validates existing session

**❌ NEVER**:

- Direct `session.df` access without validation
- Manual null checks instead of using helpers
- Assert statements for validation

### MCP Tool Development

**Server Composition Pattern** - Create domain-specific servers:

```python
from fastmcp import Context, FastMCP
from pydantic import BaseModel, ConfigDict, Field


# 1. Define Pydantic models
class DomainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    success: bool = True


# 2. Implement synchronous logic
def process_operation(session_id: str, ctx: Context | None = None) -> DomainResult:
    # Implementation
    return DomainResult(session_id=session_id)


# 3. Register as MCP server
domain_server = FastMCP("DataBeak-Domain")
domain_server.tool(name="process_operation")(process_operation)

# 4. Mount in main server (src/databeak/server.py)
# mcp.mount(domain_server)
```

**Modern Pydantic**: Use discriminated unions for type conversion:

```python
class BaseRule(BaseModel):
    type: str


RuleType = Annotated[TypeARule | TypeBRule, Field(discriminator="type")]
```

See validation_server.py for complete example.

### Type Safety

**Minimize `Any` usage** - Use structured types:

```python
# ❌ Avoid
def operation() -> dict[str, Any]: ...


# ✅ Prefer
class OperationSuccess(TypedDict):
    success: Literal[True]
    data: dict[str, CellValue]


class OperationError(TypedDict):
    success: Literal[False]
    error: str


OperationResult = OperationSuccess | OperationError
```

**Common MyPy fixes**:

- DataFrame | None access: Add null checks
- TYPE_CHECKING imports: Use `if TYPE_CHECKING:` block for type-only imports

### Logging

**Use Context-based logging** when FastMCP Context available:

```python
# ✅ In MCP tools
await ctx.info("Message")
await ctx.error("Error")

# ✅ In non-MCP functions
logger.info("Message")  # Standard Python logger
```

### Configuration

**No magic numbers** - Use DataBeakSettings:

```python
# ❌ Avoid
if memory_usage > 1024 * 0.75:
    ...

# ✅ Use settings
settings = get_csv_settings()
if memory_usage > threshold * settings.memory_warning_threshold:
    ...
```

All environment variables use `DATABEAK_` prefix. Configuration centralized in
`csv_session.py`.

### Testing Patterns

See tests/README.md and docs/testing.md for comprehensive guidance.

**Session fixture pattern**:

```python
@pytest.fixture
async def test_session():
    """Standard session fixture."""
    csv_content = """col1,col2\nval1,val2"""
    result = await load_csv_from_content(csv_content)
    yield result.session_id
    manager = get_session_manager()
    await manager.remove_session(result.session_id)
```

**Run tests**: `uv run pytest -n auto tests/unit/` (primary focus)

### Version Management

- Primary source: `pyproject.toml`
- Sync: `uv run sync-versions`
- Code uses dynamic loading via `importlib.metadata`

## Quick Reference Commands

```bash
# Development
uv sync                              # Install dependencies
uv run databeak                      # Run MCP server (stdio)
uv run databeak --transport http --host 0.0.0.0 --port 8000  # HTTP mode

# Quality checks
uv run pre-commit run --all-files    # All quality checks
uv run ruff check --fix src/ tests/  # Fix linting
uv run ruff format src/ tests/       # Format code
uv run mypy src/databeak/            # Type check
uv run pytest -n auto tests/unit/    # Run unit tests
scripts/check_docstring_args.py      # MCP Args compliance
scripts/check_mcp_field_descriptions.py  # MCP Field compliance

# Version sync
uv run sync-versions                 # After version changes

# Documentation
uv run mkdocs serve                  # Serve docs locally
uv run mdformat docs/                # Format markdown
uv run pymarkdownlnt scan docs/      # Lint markdown
```

## Architecture Notes

- **Stateless MCP design** with external context management
- **Session-based processing** with automatic cleanup
- **Modular server composition** for domain separation
- **Type-safe operations** using Pydantic validation

**File structure**:
`src/databeak/{server.py, models/, servers/, services/, utils/}`

## Additional Resources

- @./.claude/Claude_Code_Style_Guide.md - Communication style
- docs/quality.md - Detailed quality standards and tools
- docs/testing.md - Comprehensive testing guide
- docs/architecture.md - Architecture details
- CONTRIBUTING.md - Git workflow and contributing guide
