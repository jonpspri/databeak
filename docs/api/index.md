---
sidebar_position: 1
title: API Overview
---

## API Reference Overview

DataBeak provides 40+ tools for comprehensive CSV manipulation through the Model
Context Protocol (MCP). All tools return structured responses and include
comprehensive error handling.

## Tool Categories

### 📁 I/O Operations

Tools for loading CSV data from web sources:

- **`load_csv_from_url`** - Load CSV from HTTP/HTTPS URL
- **`load_csv_from_content`** - Load CSV from string content
- **`get_session_info`** - Get current session details and statistics
- **`list_sessions`** - List all active sessions
- **`close_session`** - Close and cleanup a session

### 🔧 Data Manipulation

Tools for transforming and modifying CSV data:

- **`filter_rows`** - Filter rows with complex conditions (AND/OR logic)
- **`sort_data`** - Sort by single or multiple columns
- **`select_columns`** - Select specific columns by name or pattern
- **`rename_columns`** - Rename columns with mapping
- **`add_column`** - Add computed columns with formulas
- **`remove_columns`** - Remove unwanted columns
- **`update_column`** - Update column values with transformations
- **`change_column_type`** - Convert column data types
- **`fill_missing_values`** - Handle null/NaN values with strategies
- **`remove_duplicates`** - Remove duplicate rows with optional key columns

### 📊 Data Analysis

Tools for statistical analysis and insights:

- **`get_statistics`** - Descriptive statistics for numeric columns
- **`get_column_statistics`** - Detailed stats for specific columns
- **`get_correlation_matrix`** - Pearson, Spearman, and Kendall correlations
- **`group_by_aggregate`** - Group data with aggregation functions
- **`get_value_counts`** - Frequency counts for categorical data
- **`detect_outliers`** - Find outliers using IQR, Z-score, or custom methods
- **`profile_data`** - Comprehensive data profiling report

### ✅ Data Validation

Tools for schema validation and quality checking:

- **`validate_schema`** - Validate data against schema definitions
- **`check_data_quality`** - Overall data quality scoring
- **`find_anomalies`** - Detect statistical and pattern anomalies

### 🔄 Session Management

Tools for managing data sessions:

- **`list_sessions`** - List all active sessions
- **`close_session`** - Close and cleanup a session
- **`get_session_info`** - Get session metadata and statistics

### ⚙️ System Tools

System information and health monitoring:

- **`health_check`** - Server health and status
- **`get_server_info`** - Server capabilities and configuration

## Common Patterns

### Error Handling

All tools return consistent response format:

```json
{
  "success": true,
  "data": {...},
  "session_id": "uuid-here"
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error description",
  "session_id": "uuid-here"
}
```

### Session Management

Most tools require a `session_id` parameter. Sessions are automatically created
and managed with configurable timeouts.

### Data Types

DataBeak supports rich data types including:

- **Strings**, **Numbers**, **Booleans**
- **Dates** and **DateTime** objects
- **Null values** (JSON `null` → Python `None` → pandas `NaN`)

### Filtering Conditions

Filter operations support complex conditions:

```json
{
  "conditions": [
    {"column": "age", "operator": ">", "value": 18},
    {"column": "status", "operator": "==", "value": "active"}
  ],
  "logic": "AND"  // or "OR"
}
```

### Environment Configuration

All tools respect these environment variables (all use `DATABEAK_` prefix):

| Variable                              | Default   | Purpose                          |
| ------------------------------------- | --------- | -------------------------------- |
| `DATABEAK_SESSION_TIMEOUT`            | 3600      | Session timeout (seconds)        |
| `DATABEAK_MAX_DOWNLOAD_SIZE_MB`       | 100       | Maximum URL download size (MB)   |
| `DATABEAK_MAX_MEMORY_USAGE_MB`        | 1000      | Max DataFrame memory (MB)        |
| `DATABEAK_MAX_ROWS`                   | 1,000,000 | Max DataFrame rows               |
| `DATABEAK_URL_TIMEOUT_SECONDS`        | 30        | URL download timeout (seconds)   |
| `DATABEAK_HEALTH_MEMORY_THRESHOLD_MB` | 2048      | Health monitoring threshold (MB) |

See
[DatabeakSettings](https://github.com/jonpspri/databeak/blob/main/src/databeak/core/settings.py)
for all configuration options.

## Advanced Features

### Null Value Support

Full support for null values across all operations:

- JSON `null` values are preserved and handled correctly
- Python `None` and pandas `NaN` compatibility
- Filtering and operations work seamlessly with nulls

### Stateless Architecture

Clean MCP server design:

- **Session-based processing** - Data operations without internal state
- **External persistence** - Context handles data persistence as needed
- **Resource efficient** - No overhead from history or auto-save tracking
- **MCP-aligned** - Follows Model Context Protocol server patterns

______________________________________________________________________

**For detailed examples and tutorials, see the
[Quick Start Guide](../tutorials/quickstart.md)**
