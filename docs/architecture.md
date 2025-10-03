---
sidebar_position: 4
title: Architecture
---

## Architecture Overview

DataBeak is built as a Model Context Protocol (MCP) server that provides AI
assistants with comprehensive CSV data manipulation capabilities. This document
explains the technical architecture and design decisions.

## Technology Stack

- **Framework**: FastMCP 2.11.3+ (Model Context Protocol)
- **Data Processing**: Pandas 2.2.3+, NumPy 2.1.3+
- **Package Manager**: uv (ultra-fast Python package management)
- **Build System**: Hatchling
- **Code Quality**: Ruff (linting and formatting), MyPy (type checking)
- **Configuration**: Pydantic Settings for environment management

## Core Components

```text
src/databeak/
├── server.py                 # FastMCP server composition & routing
├── models/                   # Data models and session management
│   ├── csv_session.py          # Session management & settings
│   ├── data_models.py          # Core data types & enums
│   ├── data_session.py         # Data operations
│   ├── pandera_schemas.py      # Pandera schema integration for validation
│   ├── typed_dicts.py          # TypedDict definitions for type safety
│   └── tool_responses.py       # Pydantic response models
├── servers/                  # Specialized MCP servers (server composition)
│   ├── io_server.py            # Load/export operations
│   ├── transformation_server.py # Data transformation
│   ├── statistics_server.py    # Statistical analysis
│   ├── discovery_server.py     # Data profiling & discovery
│   ├── validation_server.py    # Schema validation & quality
│   ├── column_server.py        # Column operations
│   ├── column_text_server.py   # Text manipulation
│   ├── row_operations_server.py # Row-level operations
│   └── system_server.py        # Health & system info
├── services/                 # Business logic services
├── utils/                    # Utility functions
├── exceptions.py             # Custom error handling
└── _version.py              # Dynamic version loading
```

## Key Features

### Session Management

- **Multi-session support** with automatic cleanup
- **Configurable timeouts** and resource limits
- **Session isolation** for concurrent users

### Data Operations

- **40+ tools** covering I/O, manipulation, analysis, and validation
- **Multiple format support**: CSV, JSON, Excel, Parquet, HTML, Markdown
- **Streaming processing** for large files
- **Type-safe operations** with Pydantic validation

### Auto-Save & History

- **Automatic saving** after each operation
- **Undo/redo functionality** with operation tracking
- **Persistent history** with JSON storage
- **Configurable strategies**: overwrite, backup, versioned

### Configuration Management

- **Environment-based settings** using Pydantic Settings
- **Centralized configuration** in CSVSettings class
- **Runtime version detection** via importlib.metadata

### Code Quality & Architecture

- **Zero static analysis violations** - Clean ruff compliance across all
  categories
- **Strong type safety** - 100% mypy compliance with minimal Any usage
- **High test coverage** - 983 unit tests + 43 integration tests with good
  coverage targets
- **Server composition pattern** - Modular FastMCP servers for different domains
- **Context-based logging** - MCP-integrated logging for better traceability
- **Clear API design** - Keyword-only boolean parameters, no boolean traps
- **Defensive practices** - Proper exception handling, input validation

## Environment Variables

All configuration uses the `DATABEAK_` prefix:

| Variable                             | Default | Purpose                                |
| ------------------------------------ | ------- | -------------------------------------- |
| `DATABEAK_MAX_FILE_SIZE_MB`          | 1024    | Maximum file size limit                |
| `DATABEAK_SESSION_TIMEOUT`           | 3600    | Session timeout (seconds)              |
| `DATABEAK_CHUNK_SIZE`                | 10000   | Processing chunk size                  |
| `DATABEAK_MEMORY_THRESHOLD_MB`       | 2048    | Memory threshold for health monitoring |
| `DATABEAK_MAX_VALIDATION_VIOLATIONS` | 1000    | Max validation violations to report    |
| `DATABEAK_MAX_ANOMALY_SAMPLE_SIZE`   | 10000   | Max sample size for anomaly detection  |

## MCP Integration

The server implements the Model Context Protocol standard:

- **Tools**: 40+ data manipulation functions
- **Resources**: Session and data access
- **Prompts**: Data analysis templates
- **Error Handling**: Structured error responses

### Tool Categories

1. **I/O Operations** - Load/export data in multiple formats
1. **Data Manipulation** - Transform, filter, sort, and modify data
1. **Data Analysis** - Statistics, correlations, outliers, profiling
1. **Data Validation** - Schema validation, quality checking, anomaly detection
1. **Session Management** - Stateless data processing with external context
   management
1. **System Tools** - Health monitoring and server information

## Design Principles

1. **Type Safety**: Full type annotations with Pydantic validation
1. **Modularity**: Clear separation of concerns across modules
1. **Performance**: Streaming operations for large datasets
1. **Reliability**: Comprehensive error handling and logging
1. **Usability**: Simple installation and configuration
1. **Maintainability**: Modern tooling and clear documentation

## Quality Standards

DataBeak maintains strict code quality standards with automated enforcement:

### Code Quality Metrics

- **Zero ruff violations** - Perfect linting compliance across 46 rules
- **100% MyPy compliance** - Complete type safety with minimal Any usage
- **Perfect MCP documentation** - Comprehensive Field descriptions, no Args
  sections
- **High test coverage** - 983 unit tests + 43 integration tests validating all
  functionality
- **Clean architecture** - Stateless MCP design with eliminated complexity

### Quality Enforcement Tools

- **Ruff** - Comprehensive linting and formatting (46 rules enabled)
- **MyPy** - Static type checking with strict configuration
- **Pre-commit hooks** - Automated quality gates preventing regressions
- **Custom MCP checkers** - Specialized tools for MCP documentation standards:
  - `check_docstring_args.py` - Ensures no Args sections in MCP tool docstrings
  - `check_mcp_field_descriptions.py` - Validates comprehensive Field
    descriptions

### Quality Commands

```bash
# Run all quality checks
uv run pre-commit run --all-files

# Individual checks
uv run ruff check src/ tests/           # Linting
uv run mypy src/databeak/                        # Type checking
uv run pytest tests/unit/               # Unit tests
scripts/check_docstring_args.py         # MCP Args compliance
scripts/check_mcp_field_descriptions.py # MCP Field compliance
```

## Development Workflow

### Package Management

```bash
uv sync              # Install dependencies
uv run databeak      # Run server
uv run -m pytest    # Run tests
uv run ruff check && uv run ruff format --check && uv run mypy src/databeak/ && uv run -m pytest
```

### Version Management

- **Single source of truth**: pyproject.toml
- **Automatic synchronization**: `uv run sync-versions`
- **Dynamic loading**: via importlib.metadata

### Quality Assurance

- **Linting**: Ruff with comprehensive rule set
- **Formatting**: Ruff with 100-character lines
- **Type checking**: MyPy with strict configuration
- **Testing**: Three-tier testing strategy
  - **Unit tests** (`tests/unit/`): Fast, isolated module testing
  - **Integration tests** (`tests/integration/`): Component interaction
    validation
  - **E2E tests** (`tests/e2e/`): Complete workflow verification
  - **Coverage target**: 80%+ with pytest-cov
  - See [Testing Guide](testing.md) for best practices

## Future Considerations

- **Advanced transformation interface** for complex operations
- **Real-time collaboration** features
- **Machine learning integrations** for data insights
- **Cloud storage support** for remote data sources
- **Advanced visualization tools** for data exploration

______________________________________________________________________

**For implementation details and contributing guidelines, see
[CONTRIBUTING.md](https://github.com/jonpspri/databeak/blob/main/CONTRIBUTING.md)**
