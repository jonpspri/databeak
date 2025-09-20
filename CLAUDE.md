# Claude Code Instructions for DataBeak Development

## Project Context

DataBeak is a Model Context Protocol (MCP) server that provides AI assistants
with 40+ tools for CSV data manipulation. Built with FastMCP, Pandas, and modern
Python tooling.

## Development Guidelines

### Git Workflow Requirements

**IMPORTANT**: All development must follow a branch-based workflow.

**Use the `git-repository-manager` agent** for all Git and GitHub repository
management tasks including:

- Synchronizing main branch after PR merges
- Creating and managing feature branches
- Cleaning up merged branches (local and remote)
- Validating Git workflow compliance
- Repository maintenance and hygiene

See `.claude/agents/git-repository-manager.md` for comprehensive Git and GitHub
repository management guidance.

### Package Management

- Use `uv` for all Python operations (not pip or poetry)
- Run tests with `uv run -m pytest`
- Use `uv run python` instead of direct `python` commands
- Install dependencies with `uv add <package>` or `uv add --dev <package>`

### Code Quality Standards

DataBeak maintains strict code quality standards:

- **Zero ruff violations** - Clean linting compliance across all categories
- **100% mypy compliance** - Strong type safety with minimal Any usage
- **High test coverage** - 1100+ unit tests with good coverage targets
- **Clear API design** - No boolean traps, keyword-only parameters for clarity
- **Defensive practices** - No silent exception handling, proper validation
- **No magic numbers** - Use configurable settings with defaults instead of
  hardcoded values

**Use the `quality-gate-runner` agent** for comprehensive quality pipeline
execution including linting, formatting, type checking, testing, and coverage
analysis.

See `.claude/agents/quality-gate-runner.md` for detailed quality assurance
guidance.

### Markdown Standards

- **Line length**: Keep lines under 80 characters for readability
- **HTML in Markdown**: Only `<details>` and `<summary>` tags allowed for
  collapsible sections; use markdown syntax for all other formatting
- **Linting**: Use `uv run pymarkdownlnt scan docs/` for markdown validation
- **Formatting**: Use `uv run mdformat docs/` to auto-format markdown files
- **Pre-commit**: Both tools integrated in pre-commit hooks for automatic
  checking
- **README files**: Provide concise context and structure information only, not
  project status or tracking

### Testing Approach

DataBeak currently focuses on comprehensive unit testing with future plans for
integration and E2E testing:

1. **Unit Tests** (`tests/unit/`) - Fast, isolated module tests (current focus)
1. **Integration Tests** (`tests/integration/`) - Future: FastMCP Client-based
   testing
1. **E2E Tests** (`tests/e2e/`) - Future: Complete workflow validation

**Current Test Execution:**

```bash
uv run pytest -n auto tests/unit/          # Run unit tests (primary)
uv run pytest -n auto --cov=src/databeak   # Run with coverage analysis
```

**Future Integration Testing:** Planned implementation using FastMCP Client for
realistic MCP protocol testing (tracked in GitHub issues).

**Use the `test-coverage-analyzer` agent** for systematic test coverage analysis
and gap identification to achieve the 80%+ coverage requirement through
comprehensive unit testing.

See `.claude/agents/test-coverage-analyzer.md` and `tests/README.md` for
comprehensive testing guidance.

### Version Management

- Primary version source: `pyproject.toml`
- Sync versions with `uv run sync-versions` after version changes
- Code uses dynamic version loading via `importlib.metadata`

### MCP Tool Development

**Use the `mcp-tool-generator` agent** for creating new MCP tools following
established DataBeak patterns with proper error handling, type annotations,
docstrings, and test files.

See `.claude/agents/mcp-tool-generator.md` for comprehensive MCP tool
development guidance.

### Type Checking Guidelines

**Use the `python-type-optimizer` agent** for systematic type safety
improvements including reducing `Any` usage and implementing specific TypedDict
definitions.

See `.claude/agents/python-type-optimizer.md` for comprehensive type annotation
guidance.

### Environment Configuration

- All environment variables use `DATABEAK_` prefix
- Configuration centralized in `DataBeakSettings` class in `csv_session.py`
- Default values defined in the Settings class, not scattered `os.getenv()`
  calls

### Logging Guidelines

**Context-Based Logging Rule**: When a FastMCP `Context` object is available
(e.g., in MCP tool functions), use the Context for all logging operations
instead of standard Python loggers.

- **Use**: `await ctx.info("Message")`, `await ctx.error("Error")`
- **Not**: `logger.info("Message")`, `logger.error("Error")`
- **Benefit**: Better traceability, MCP protocol integration, client visibility
- **Standard loggers**: Only use in non-MCP functions (utilities, internal
  logic)

### Configuration and Magic Numbers

**Avoid Magic Numbers Rule**: Replace hardcoded values with configurable
settings that have sensible defaults and can be overridden via environment
variables.

**Examples of proper configuration:**

```python
# ❌ Avoid: Magic numbers hardcoded
if memory_usage > 1024 * 0.75:  # What's this threshold?
    status = "warning"

# ✅ Prefer: Configurable settings with defaults
settings = get_csv_settings()
if memory_usage > threshold * settings.memory_warning_threshold:
    status = "warning"
```

**Configuration Guidelines:**

- **Use DataBeakSettings class** for all configurable values
- **Environment variable support** with `DATABEAK_` prefix
- **Sensible defaults** that work out of the box
- **Clear documentation** of what each setting controls
- **Type safety** with Pydantic Field validation

**Common Configuration Categories:**

- **Thresholds**: Memory limits, capacity warnings, timeouts
- **Limits**: History operations, file sizes, session counts
- **Behavior**: Auto-save settings, cleanup intervals, retry attempts
- **Performance**: Cache TTLs, chunk sizes, batch limits

### Architecture Notes

- Session-based design with automatic cleanup
- Auto-save functionality enabled by default
- Persistent history with undo/redo capabilities
- Type-safe operations using Pydantic validation
- Modular tool organization for maintainability

## Common Commands

```bash
# Setup and development
uv sync                 # Install all dependencies
uv run databeak      # Run the MCP server

# Version management
uv run sync-versions   # Sync version numbers across files

# Markdown quality
uv run mdformat docs/              # Auto-format markdown files
uv run pymarkdownlnt scan docs/    # Lint markdown files

# Documentation
uv run mkdocs build       # Build MkDocs site
uv run mkdocs serve       # Serve docs locally
```

## File Structure Context

- `src/databeak/` - Main package code
- `tests/` - Test suite mirroring source structure
- `examples/` - Usage examples and demos
- `docs/` - MkDocs documentation site
- `scripts/` - Maintenance and utility scripts

## Key Implementation Details

- FastMCP framework for MCP protocol
- Pydantic models for data validation
- Pandas for data operations
- Session management with configurable timeouts
- Error handling and logging

## Documentation Tone Standards

DataBeak documentation maintains professional, factual tone:

### Avoid Self-Aggrandizing Language

**Prohibited terms**:

- "exceptional", "perfect", "amazing", "outstanding", "superior"
- "revolutionary", "cutting-edge", "world-class", "best-in-class"
- "unparalleled", "state-of-the-art", "industry-leading", "premium", "elite"
- "ultimate", "maximum", "optimal", "flawless"

**Use factual alternatives**:

- "exceptional standards" → "strict standards"
- "perfect compliance" → "clean compliance"
- "comprehensive coverage" → "high coverage"
- "API design excellence" → "clear API design"
- "security best practices" → "defensive practices"

### Measurable Claims Only

**Acceptable** (measurable):

- "Zero ruff violations" (verifiable metric)
- "100% mypy compliance" (measurable result)
- "1100+ unit tests" (concrete count)

**Prohibited** (subjective claims):

- "production quality" (marketing speak)
- "advanced analytics" (vague superlative)
- "sophisticated architecture" (self-congratulatory)

### Professional Descriptors

Use measured, technical language:

- "provides" not "delivers amazing"
- "supports" not "offers comprehensive"
- "implements" not "features advanced"
- "handles" not "excels at"

## Additional Resources

- @./.claude/Claude_Code_Style_Guide.md
