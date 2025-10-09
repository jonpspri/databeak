# DataBeak MCP Server - AI-Powered CSV Data Platform

A high-performance MCP server providing 40+ specialized tools for CSV data
manipulation, analysis, and validation. Optimized for AI assistants with precise
coordinate-based operations and comprehensive null value support.

## 🎯 Core Philosophy

**AI-First Design**: Every tool is designed for natural language interaction
with AI assistants, providing precise coordinates, detailed metadata, and
comprehensive error handling.

**Modular Architecture**: Tools are organized into logical categories:

- **System**: Health checks and server information
- **I/O**: Loading data from web sources (URLs, string content)
- **Data**: Filtering, sorting, transformations, and column operations
- **Row**: Precise row-level and cell-level access and manipulation
- **Analytics**: Statistical analysis and data profiling
- **Validation**: Data quality checks and schema validation
- **Session**: Session management and health monitoring

## 📐 Coordinate System (Critical for AI Success)

**🎯 Zero-based indexing** for maximum precision:

- **Rows**: 0 to N-1 (where N = total rows)
- **Columns**: Use column names (strings) OR 0-based indices (integers)
- **Cells**: Address as (row_index, column) - e.g., (0, "name") or (2, 1)

### Data Structure Reference

```text
    Column indices:  0      1      2
                   name   age   city
Row 0:             John   30    NYC      <- get_cell_value(session, 0, "name")
Row 1:             Jane   25    LA       <- get_cell_value(session, 1, 1) → 25
Row 2:             Bob    35    Chicago  <- get_row_data(session, 2) →
                                            {"name": "Bob", "age": 35, "city": "Chicago"}
```

**⚠️ Critical**: All operations use 0-based indexing. Row 1 in a spreadsheet =
Row 0 in DataBeak.

### 🎯 **Stateless Design**

Clean MCP architecture for AI assistants

## Getting Started (AI Workflow)

### Step 1: Load Data

```python
# Load from content (most common for AI)
load_csv_from_content("name,age,city\nJohn,30,NYC\nJane,25,LA")
# Returns: session_id + enhanced preview with __row_index__ field
```

### Step 2: Explore and Transform

```python
# Inspect data structure and content
get_data_summary(session_id)  # Comprehensive overview
get_row_data(session_id, 0)  # Inspect specific row
get_cell_value(session_id, 0, "name")  # Read individual cell

# Transform data with precise operations
filter_rows(session_id, [{"column": "age", "operator": ">", "value": 25}])
sort_data(session_id, ["name"])  # Sort by column
insert_row(session_id, -1, {"name": "Alice", "email": null})  # Add with nulls
```

### Step 3: Analyze Results

```python
get_statistics(session_id, ["age"])  # Statistical analysis
# Note: DataBeak processes data in memory for web-based hosting security
```

## Enhanced Resource Endpoints

- `csv://{session_id}/data` - Full data with enhanced indexing
- `csv://{session_id}/cell/{row}/{col}` - Direct cell access
- `csv://{session_id}/row/{index}` - Row-specific data
- `csv://{session_id}/preview` - Enhanced preview with coordinate system docs
- `csv://{session_id}/schema` - Column information and data types

## Key Features

• **Session-based**: Multiple independent data sessions with automatic cleanup •
**History tracking**: Full operation history with snapshots for undo/redo •
**Coordinate precision**: Every operation includes row/column coordinate
information • **AI-optimized returns**: All data includes indexing for precise
reference • **Clear method names**: No confusing operation parameters - method
names express intent • **Enhanced error messages**: Include valid coordinate
ranges in error responses • **Progress reporting**: Real-time feedback for long
operations • **Type safety**: Proper handling of pandas/numpy types for JSON
serialization • **Null value support**: Full JSON null → Python None → pandas
NaN compatibility • **Claude Code compatible**: Automatic JSON string
deserialization

## 🎯 AI Assistant Guidelines

**Key Principles:**

- Always start with `get_data_summary()` to understand data structure and
  coordinate system
- Use precise coordinates (0-based indexing) for all operations
- Check tool docstrings for comprehensive usage examples and workflow patterns
- All operations include detailed metadata for continued AI analysis

**Session Management:**

- Each CSV dataset gets an isolated session with unique session_id
- Sessions support multiple concurrent operations with full history tracking
- Use `list_sessions()` to manage multiple active datasets

**Data Quality:**

- All tools preserve data types and handle null values seamlessly
- JSON `null` → Python `None` → pandas `NaN` conversion is automatic
- Use validation tools for data quality assessment before major transformations
