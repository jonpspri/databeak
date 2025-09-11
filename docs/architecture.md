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
├── server.py           # FastMCP server entry point
├── models/            # Data models and session management
│   ├── csv_session.py    # Session management & settings
│   ├── data_models.py    # Core data types
│   └── data_session.py   # Data operations
├── tools/             # MCP tool implementations
│   ├── data_io.py       # Load/export operations
│   ├── data_manipulation.py  # Transform operations
│   ├── data_analysis.py     # Statistics & analysis
│   └── data_validation.py   # Schema validation
├── exceptions.py      # Custom error handling
└── _version.py        # Dynamic version loading
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

## Environment Variables

All configuration uses the `DATABEAK_` prefix:

| Variable                    | Default | Purpose                   |
| --------------------------- | ------- | ------------------------- |
| `DATABEAK_MAX_FILE_SIZE_MB` | 1024    | Maximum file size         |
| `DATABEAK_CSV_HISTORY_DIR`  | "."     | History storage location  |
| `DATABEAK_SESSION_TIMEOUT`  | 3600    | Session timeout (seconds) |
| `DATABEAK_CHUNK_SIZE`       | 10000   | Processing chunk size     |
| `DATABEAK_AUTO_SAVE`        | true    | Enable auto-save          |

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
1. **Session Management** - Auto-save, history, undo/redo operations
1. **System Tools** - Health monitoring and server information

## Design Principles

1. **Type Safety**: Full type annotations with Pydantic validation
1. **Modularity**: Clear separation of concerns across modules
1. **Performance**: Streaming operations for large datasets
1. **Reliability**: Comprehensive error handling and logging
1. **Usability**: Simple installation and configuration
1. **Maintainability**: Modern tooling and clear documentation

## Development Workflow

### Package Management

```bash
uv sync              # Install dependencies
uv run databeak      # Run server
uv run -m pytest    # Run tests
uv run ruff check && uv run ruff format --check && uv run mypy src/ && uv run -m pytest
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
