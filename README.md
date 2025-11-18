# DataBeak

[![Tests](https://github.com/jonpspri/databeak/actions/workflows/test.yml/badge.svg)](https://github.com/jonpspri/databeak/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jonpspri/databeak/branch/main/graph/badge.svg)](https://codecov.io/gh/jonpspri/databeak)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## AI-Powered CSV Processing via Model Context Protocol

Transform how AI assistants work with CSV data. DataBeak provides 40+
specialized tools for data manipulation, analysis, and validation through the
Model Context Protocol (MCP).

<a href="https://glama.ai/mcp/servers/@jonpspri/databeak">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@jonpspri/databeak/badge" alt="DataBeak MCP server" />
</a>

## Features

- 🔄 **Complete Data Operations** - Load, transform, and analyze CSV data from
  URLs and string content
- 📊 **Advanced Analytics** - Statistics, correlations, outlier detection, data
  profiling
- ✅ **Data Validation** - Schema validation, quality scoring, anomaly detection
- 🎯 **Stateless Design** - Clean MCP architecture with external context
  management
- ⚡ **High Performance** - Async I/O, streaming downloads, chunked processing
- 🔒 **Session Management** - Multi-user support with isolated sessions
- 🛡️ **Web-Safe** - No file system access; designed for secure web hosting
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
"Load this CSV data: name,price\nWidget,10.99\nGadget,25.50"
"Load CSV from URL: https://example.com/data.csv"
"Remove duplicate rows and show me the statistics"
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

Configure DataBeak behavior with environment variables (all use `DATABEAK_`
prefix):

| Variable                              | Default   | Description                        |
| ------------------------------------- | --------- | ---------------------------------- |
| `DATABEAK_SESSION_TIMEOUT`            | 3600      | Session timeout (seconds)          |
| `DATABEAK_MAX_DOWNLOAD_SIZE_MB`       | 100       | Maximum URL download size (MB)     |
| `DATABEAK_MAX_MEMORY_USAGE_MB`        | 1000      | Max DataFrame memory (MB)          |
| `DATABEAK_MAX_ROWS`                   | 1,000,000 | Max DataFrame rows                 |
| `DATABEAK_URL_TIMEOUT_SECONDS`        | 30        | URL download timeout               |
| `DATABEAK_HEALTH_MEMORY_THRESHOLD_MB` | 2048      | Health monitoring memory threshold |

See [settings.py](src/databeak/core/settings.py) for complete configuration
options.

## Known Limitations

DataBeak is designed for interactive CSV processing with AI assistants. Be aware
of these constraints:

- **Data Loading**: URLs and string content only (no local file system access
  for web hosting security)
- **Download Size**: Maximum 100MB per URL download (configurable via
  `DATABEAK_MAX_DOWNLOAD_SIZE_MB`)
- **DataFrame Size**: Maximum 1GB memory and 1M rows per DataFrame
  (configurable)
- **Session Management**: Maximum 100 concurrent sessions, 1-hour timeout
  (configurable)
- **Memory**: Large datasets may require significant memory; monitor with
  `health_check` tool
- **CSV Dialects**: Assumes standard CSV format; complex dialects may require
  pre-processing
- **Concurrency**: Async I/O for concurrent URL downloads; parallel sessions
  supported
- **Data Types**: Automatic type inference; complex types may need explicit
  conversion
- **URL Loading**: HTTPS only; blocks private networks (127.0.0.1, 192.168.x.x,
  10.x.x.x) for security

For production deployments with larger datasets, adjust environment variables
and monitor resource usage with `health_check` and `get_server_info` tools.

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

- **Unit Tests** (`tests/unit/`) - 940+ fast, isolated module tests
- **Integration Tests** (`tests/integration/`) - 43 FastMCP Client-based
  protocol tests across 7 test files
- **E2E Tests** (`tests/e2e/`) - Planned: Complete workflow validation

**Test Execution:**

```bash
uv run pytest -n auto tests/unit/          # Run unit tests (940+ tests)
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