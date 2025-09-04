---
sidebar_position: 4
title: Architecture
---

## Architecture Overview

CSV Editor is built as a Model Context Protocol (MCP) server that provides
AI assistants with comprehensive CSV data manipulation capabilities. This
document explains the technical architecture and design decisions.

## Technology Stack

- **Framework**: FastMCP 2.11.3+ (Model Context Protocol)
- **Data Processing**: Pandas 2.2.3+, NumPy 2.1.3+
- **Package Manager**: uv (ultra-fast Python package management)
- **Build System**: Hatchling
- **Code Quality**: Ruff (linting and formatting), MyPy (type
  checking)
- **Configuration**: Pydantic Settings for environment management

## Core Components

```text
src/csv_editor/
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
- **Multiple format support**: CSV, JSON, Excel, Parquet, HTML,
  Markdown
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

All configuration uses the `CSV_EDITOR_` prefix:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CSV_EDITOR_MAX_FILE_SIZE_MB` | 1024 | Maximum file size |
| `CSV_EDITOR_CSV_HISTORY_DIR` | "." | History storage location |
| `CSV_EDITOR_SESSION_TIMEOUT` | 3600 | Session timeout (seconds) |
| `CSV_EDITOR_CHUNK_SIZE` | 10000 | Processing chunk size |
| `CSV_EDITOR_AUTO_SAVE` | true | Enable auto-save |

## MCP Integration

The server implements the Model Context Protocol standard:

- **Tools**: 40+ data manipulation functions
- **Resources**: Session and data access
- **Prompts**: Data analysis templates
- **Error Handling**: Structured error responses

### Tool Categories

1. **I/O Operations** - Load/export data in multiple formats
2. **Data Manipulation** - Transform, filter, sort, and modify data
3. **Data Analysis** - Statistics, correlations, outliers, profiling
4. **Data Validation** - Schema validation, quality checking, anomaly
   detection
5. **Session Management** - Auto-save, history, undo/redo operations
6. **System Tools** - Health monitoring and server information

## Design Principles

1. **Type Safety**: Full type annotations with Pydantic validation
2. **Modularity**: Clear separation of concerns across modules
3. **Performance**: Streaming operations for large datasets
4. **Reliability**: Comprehensive error handling and logging
5. **Usability**: Simple installation and configuration
6. **Maintainability**: Modern tooling and clear documentation

## Development Workflow

### Package Management

```bash
uv sync              # Install dependencies
uv run csv-editor    # Run server
uv run test          # Run tests
uv run all-checks    # Lint, format, type-check, test
```

### Version Management

- **Single source of truth**: pyproject.toml
- **Automatic synchronization**: `uv run sync-versions`
- **Dynamic loading**: via importlib.metadata

### Quality Assurance

- **Linting**: Ruff with comprehensive rule set
- **Formatting**: Ruff with 100-character lines
- **Type checking**: MyPy with strict configuration
- **Testing**: pytest with asyncio support and coverage
  reporting

## Future Considerations

- **SQL query interface** for advanced operations
- **Real-time collaboration** features
- **Machine learning integrations** for data insights
- **Cloud storage support** for remote data sources
- **Advanced visualization tools** for data exploration

---

**For implementation details and contributing guidelines, see
[CONTRIBUTING.md](https://github.com/jonpspri/csv-editor/blob/main/CONTRIBUTING.md)**
