---
sidebar_position: 1
title: API Overview
---

## API Reference Overview

DataBeak provides 40+ tools for comprehensive CSV manipulation through the
Model Context Protocol (MCP). All tools return structured responses and
include comprehensive error handling.

## Tool Categories

### 📁 I/O Operations

Tools for loading and exporting CSV data in various formats:

- **`load_csv`** - Load CSV from file path
- **`load_csv_from_url`** - Load CSV from HTTP/HTTPS URL
- **`load_csv_from_content`** - Load CSV from string content
- **`export_csv`** - Export to CSV, JSON, Excel, Parquet, HTML, Markdown
- **`get_session_info`** - Get current session details and statistics
- **`list_sessions`** - List all active sessions
- **`close_session`** - Close and cleanup a session

### 🔧 Data Manipulation

Tools for transforming and modifying CSV data:

- **`filter_rows`** - Filter rows with complex conditions (AND/OR
  logic)
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
- **`get_correlation_matrix`** - Pearson, Spearman, and Kendall
  correlations
- **`group_by_aggregate`** - Group data with aggregation functions
- **`get_value_counts`** - Frequency counts for categorical data
- **`detect_outliers`** - Find outliers using IQR, Z-score, or custom
  methods
- **`profile_data`** - Comprehensive data profiling report

### ✅ Data Validation

Tools for schema validation and quality checking:

- **`validate_schema`** - Validate data against schema definitions
- **`check_data_quality`** - Overall data quality scoring
- **`find_anomalies`** - Detect statistical and pattern anomalies

### 🔄 Session Management

Tools for managing data sessions and workflow:

- **`configure_auto_save`** - Set up automatic saving strategies
- **`get_auto_save_status`** - Check current auto-save configuration
- **`undo`** - Undo the last operation
- **`redo`** - Redo previously undone operation
- **`get_history`** - View operation history
- **`restore_to_operation`** - Restore to specific point in history

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

Most tools require a `session_id` parameter. Sessions are automatically
created and managed with configurable timeouts.

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

All tools respect these environment variables:

| Variable                    | Default | Purpose                   |
| --------------------------- | ------- | ------------------------- |
| `DATABEAK_MAX_FILE_SIZE_MB` | 1024    | Maximum file size         |
| `DATABEAK_CSV_HISTORY_DIR`  | "."     | History storage location  |
| `DATABEAK_SESSION_TIMEOUT`  | 3600    | Session timeout (seconds) |
| `DATABEAK_CHUNK_SIZE`       | 10000   | Processing chunk size     |
| `DATABEAK_AUTO_SAVE`        | true    | Enable auto-save          |

## Advanced Features

### Null Value Support

Full support for null values across all operations:

- JSON `null` values are preserved and handled correctly
- Python `None` and pandas `NaN` compatibility
- Filtering and operations work seamlessly with nulls

### Auto-Save Strategies

Configurable auto-save with multiple strategies:

- **Overwrite** - Update original file
- **Backup** - Create timestamped backups
- **Versioned** - Maintain version history
- **Custom** - Save to specified location

### History and Undo/Redo

Complete operation tracking:

- Persistent history storage
- Snapshot-based undo/redo
- Operation metadata and timestamps
- Restore to any point in history

______________________________________________________________________

**For detailed examples and tutorials, see the
[Quick Start Guide](../tutorials/quickstart.md)**
