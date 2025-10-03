# DataBeak

[![Tests](https://github.com/jonpspri/databeak/actions/workflows/test.yml/badge.svg)](https://github.com/jonpspri/databeak/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jonpspri/databeak/branch/main/graph/badge.svg)](https://codecov.io/gh/jonpspri/databeak)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## AI-Powered CSV Processing via Model Context Protocol

Transform how AI assistants work with CSV data. DataBeak provides 40+
specialized tools for data manipulation, analysis, and validation through the
Model Context Protocol (MCP).

## Features

- 🔄 **Complete Data Operations** - Load, transform, analyze, and export CSV data
- 📊 **Advanced Analytics** - Statistics, correlations, outlier detection, data
  profiling
- ✅ **Data Validation** - Schema validation, quality scoring, anomaly detection
- 🎯 **Stateless Design** - Clean MCP architecture with external context
  management
- ⚡ **High Performance** - Handles large datasets with streaming and chunking
- 🔒 **Session Management** - Multi-user support with isolated sessions
- 🌟 **Code Quality** - Zero ruff violations, 100% mypy compliance, perfect MCP
  documentation standards, comprehensive test coverage

## Getting Started

The fastest way to use DataBeak is with `uvx` (no installation required):

### For Claude Desktop

Add this to your MCP Settings file:

```json
{
  "mcpServers": {
    "databeak": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/jonpspri/databeak.git",
        "databeak"
      ]
    }
  }
}
```

### For Other AI Clients

DataBeak works with Continue, Cline, Windsurf, and Zed. See the
[installation guide](https://jonpspri.github.io/databeak/installation) for
specific configuration examples.

### HTTP Mode (Advanced)

For HTTP-based AI clients or custom deployments:

```bash
# Run in HTTP mode
uv run databeak --transport http --host 0.0.0.0 --port 8000

# Access server at http://localhost:8000/mcp
# Health check at http://localhost:8000/health
```

### Quick Test

Once configured, ask your AI assistant:

```text
"Load a CSV file and show me basic statistics"
"Remove duplicate rows and export as Excel"
"Find outliers in the price column"
```

## Documentation

📚 **[Complete Documentation](https://jonpspri.github.io/databeak/)**

- [Installation Guide](https://jonpspri.github.io/databeak/installation) - Setup
  for all AI clients
- [Quick Start Tutorial](https://jonpspri.github.io/databeak/tutorials/quickstart)
  \- Learn in 10 minutes
- [API Reference](https://jonpspri.github.io/databeak/api/overview) - All 40+
  tools documented
- [Architecture](https://jonpspri.github.io/databeak/architecture) - Technical
  details

## Environment Variables

| Variable                    | Default | Description               |
| --------------------------- | ------- | ------------------------- |
| `DATABEAK_MAX_FILE_SIZE_MB` | 1024    | Maximum file size         |
| `DATABEAK_CSV_HISTORY_DIR`  | "."     | History storage location  |
| `DATABEAK_SESSION_TIMEOUT`  | 3600    | Session timeout (seconds) |

## Contributing

We welcome contributions! Please:

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Make your changes with tests
1. Run quality checks: `uv run -m pytest`
1. Submit a pull request

**Note**: All changes must go through pull requests. Direct commits to `main`
are blocked by pre-commit hooks.

## Development

```bash
# Setup development environment
git clone https://github.com/jonpspri/databeak.git
cd databeak
uv sync

# Run the server locally
uv run databeak

# Run tests
uv run -m pytest tests/unit/          # Unit tests (primary)
uv run -m pytest                      # All tests

# Run quality checks
uv run ruff check
uv run mypy src/databeak/
```

### Testing Structure

DataBeak implements comprehensive unit and integration testing:

- **Unit Tests** (`tests/unit/`) - 983 fast, isolated module tests
- **Integration Tests** (`tests/integration/`) - 43 FastMCP Client-based
  protocol tests across 7 test files
- **E2E Tests** (`tests/e2e/`) - Planned: Complete workflow validation

**Test Execution:**

```bash
uv run pytest -n auto tests/unit/          # Run unit tests (983 tests)
uv run pytest -n auto tests/integration/   # Run integration tests (43 tests)
uv run pytest -n auto --cov=src/databeak   # Run with coverage analysis
```

See [Testing Guide](tests/README.md) for comprehensive testing details.

## License

Apache 2.0 - see [LICENSE](LICENSE) file.

## Support

- **Issues**: [GitHub Issues](https://github.com/jonpspri/databeak/issues)
- **Discussions**:
  [GitHub Discussions](https://github.com/jonpspri/databeak/discussions)
- **Documentation**:
  [jonpspri.github.io/databeak](https://jonpspri.github.io/databeak/)
