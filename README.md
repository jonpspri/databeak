# DataBeak

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

### Docker Deployment

For production deployments or HTTP-based AI clients:

```bash
# Quick start with Docker Compose
docker-compose up -d

# Access server at http://localhost:8000/mcp
# Health check at http://localhost:8000/health
```

See the [Docker deployment guide](docs/docker-deployment.md) for production
configuration, scaling, and security considerations.

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
uv run --directory src mypy .
```

### Testing Structure

DataBeak currently focuses on comprehensive unit testing with future plans for
integration and E2E testing:

- **Unit Tests** (`tests/unit/`) - Fast, isolated module tests (current focus)
- **Integration Tests** (`tests/integration/`) - Future: FastMCP Client-based
  testing
- **E2E Tests** (`tests/e2e/`) - Future: Complete workflow validation

**Current Test Execution:**

```bash
uv run pytest -n auto tests/unit/          # Run unit tests (primary)
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
