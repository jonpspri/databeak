# Claude Code Instructions for DataBeak Development

## Project Context

DataBeak is a Model Context Protocol (MCP) server that provides AI assistants
with 40+ tools for CSV data manipulation. Built with FastMCP, Pandas, and
modern Python tooling.

## Development Guidelines

### Git Workflow Requirements

**IMPORTANT**: All development must follow a branch-based workflow:

- **NEVER commit directly to `main` branch**
- **Always create feature branches** for any changes (use `git checkout -b feature/description`)
- **All changes to `main` must go through Pull Requests**
- **Pre-commit hooks enforce this policy** and will reject direct commits to main
- **Branch naming**: Use descriptive prefixes like `feature/`, `fix/`, `docs/`, `test/`

#### Typical Development Flow

```bash
# Create feature branch
git checkout -b feature/add-new-tool

# Make changes and commit to branch
git add .
git commit -m "Add new MCP tool for data filtering"

# Push branch and create PR
git push -u origin feature/add-new-tool
gh pr create --title "Add data filtering tool" --body "Description..."

# After PR approval, merge via GitHub UI
# Delete local branch after merge
git checkout main && git pull origin main
git branch -D feature/add-new-tool
```

### Package Management

- Use `uv` for all Python operations (not pip or poetry)
- Run tests with `uv run -m pytest`
- Use `uv run python` instead of direct `python` commands
- Install dependencies with `uv add <package>` or `uv add --dev <package>`

### Code Quality Standards

- Run `uv run all-checks` before committing (includes lint, format, type-check, test)
- Use `uv run ruff check` for linting
- Use `uv run ruff format` for formatting
- Use `uv run mypy src/` for type checking
- Maintain test coverage above 80%
- **Markdown line length**: Keep lines under 80 characters for readability
- **HTML in Markdown**: Only `<details>` and `<summary>` tags allowed for
  collapsible sections; use markdown syntax for all other formatting

### Testing Approach

- Write comprehensive tests for all new MCP tools
- Use pytest fixtures for session management
- Test both success and error cases
- Include integration tests for tool workflows
- Run `uv run test-cov` to check coverage

### Version Management

- Primary version source: `pyproject.toml`
- Sync versions with `uv run sync-versions` after version changes
- Code uses dynamic version loading via `importlib.metadata`

### MCP Tool Development

- Place new tools in appropriate `tools/` modules
- Follow existing patterns for error handling and response structure
- Include comprehensive docstrings with examples
- Use type hints for all parameters and return values
- Handle null values correctly (JSON null → Python None → pandas NaN)

### Type Checking Guidelines

- Avoid using `Any` when a more specific type can be declared
- Be especially sparing in using `Any` as a function return type or dict datatype
- Prefer specific types like `dict[str, str]` over `dict[str, Any]`
- Use union types when multiple specific types are acceptable
- Consider using TypedDict for structured dictionary returns

### Environment Configuration

- All environment variables use `DATABEAK_` prefix
- Configuration centralized in `DataBeakSettings` class in `csv_session.py`
- Default values defined in the Settings class, not scattered `os.getenv()` calls

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
uv run test            # Run test suite
uv run all-checks      # Full quality check pipeline

# Version management
uv run sync-versions   # Sync version numbers across files

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
- Comprehensive error handling and logging

## Additional Resources

- @./.claude/Claude_Code_Style_Guide.md
