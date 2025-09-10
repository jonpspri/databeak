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

**Use the `test-coverage-analyzer` agent** for systematic test coverage analysis
and gap identification to achieve the 80%+ coverage requirement.

See `.claude/agents/test-coverage-analyzer.md` for comprehensive testing
guidance.

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
- Comprehensive error handling and logging

## Additional Resources

- @./.claude/Claude_Code_Style_Guide.md
