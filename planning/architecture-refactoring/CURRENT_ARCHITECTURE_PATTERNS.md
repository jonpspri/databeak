# Current Architecture Patterns Analysis

## Session Management Patterns

### **CSVSession Architecture**

**File**: `src/databeak/models/csv_session.py:CSVSession` (415 lines)

**Current Responsibilities**:

- Session lifecycle management (creation, TTL, cleanup)
- Data loading and session state management
- Auto-save configuration and triggering
- History tracking and undo/redo operations
- Operation recording and metadata updates
- Settings integration and validation

**Pattern**: **Facade + Aggregator**

```python
class CSVSession:
    def __init__(self):
        self.lifecycle = SessionLifecycle()      # Composition
        self.data_session = DataSession()       # Composition
        self.auto_save_manager = AutoSaveManager()  # Composition
        self.history_manager = HistoryManager()     # Composition
```

**Issues**:

- **God Object**: Too many responsibilities in single class
- **Tight Coupling**: Direct pandas DataFrame manipulation
- **CSV Assumptions**: Naming and error messages assume CSV format

### **SessionManager Pattern**

**File**: `src/databeak/models/csv_session.py:SessionManager` (lines 300+)

**Current Pattern**: **Singleton + Registry**

```python
_session_manager: SessionManager | None = None

def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
```

**Strengths**:

- Global session coordination
- Automatic cleanup of expired sessions
- Thread-safe session operations

**Issues**:

- **Global state**: Testing complications
- **Hidden dependencies**: Hard to inject for testing

## Tool Organization Patterns

### **MCP Tool Registration Pattern**

**Files**: All `tools/mcp_*_tools.py` files

**Current Pattern**: **Manual Registration with Boilerplate**

```python
# Repeated pattern across 7 MCP tool files:
@mcp.tool
async def tool_name(
    ctx: Context,
    session_id: str,
    # ... parameters ...
) -> dict[str, Any]:
    try:
        session = get_session(session_id)
        validate_session_has_data(session)
        result = await actual_operation(session, ...)
        session.record_operation(OperationType.FILTER, {...})
        return {"success": True, "data": result, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error in {tool_name}: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}
```

**Analysis**:

- **~200 lines of repetitive code** across modules
- **Identical error handling** in each tool
- **Manual session management** in every tool
- **Inconsistent operation recording**

### **Business Logic Separation**

**Files**: All `tools/*_operations.py` files

**Pattern**: **Pure Functions + Parameter Objects**

```python
# Good separation in files like data_operations.py:
async def filter_rows_operation(
    session: CSVSession,
    conditions: list[dict[str, Any]],
    logic: str
) -> dict[str, Any]:
    # Pure business logic without MCP concerns
```

**Strengths**:

- Clean separation from MCP protocol
- Testable business logic
- Reusable across different interfaces

## Data Model Patterns

### **Type Safety Approach**

**File**: `src/databeak/models/data_models.py` (400+ lines)

**Pattern**: **Rich Domain Models with Pydantic**

```python
class FilterCondition(BaseModel):
    column: str
    operator: ComparisonOperator
    value: CellValue

class OperationResult(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    session_id: str
```

**Strengths**:

- Comprehensive type validation
- Self-documenting interfaces
- JSON serialization built-in

**Issues**:

- **Large single file**: Could be modularized by domain
- **CSV assumptions**: Types like `CellValue` assume CSV structure
- **Limited format support**: No models for Excel sheets, database schemas

### **Configuration Pattern**

**File**: `src/databeak/models/csv_session.py:DataBeakSettings`

**Pattern**: **Pydantic Settings with Environment Variables**

```python
class DataBeakSettings(BaseSettings):
    max_file_size_mb: int = 1024
    csv_history_dir: str = "."
    session_timeout: int = 3600

    class Config:
        env_prefix = "DATABEAK_"
```

**Strengths**:

- Environment-based configuration
- Type validation and defaults
- Centralized settings management

## Error Handling Patterns

### **Exception Hierarchy**

**File**: `src/databeak/exceptions.py`

**Pattern**: **Structured Exception Hierarchy**

```python
class DataBeakError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.details = details or {}
        super().__init__(message)

class SessionError(DataBeakError): ...
class DataValidationError(DataBeakError): ...
class FileOperationError(DataBeakError): ...
```

**Strengths**:

- Structured error information
- Hierarchical error types
- Consistent error formatting

### **Error Response Pattern**

**Usage**: Consistent across all MCP tools

**Pattern**: **Standardized Error Responses**

```python
# Success response:
{"success": True, "data": result, "session_id": session_id}

# Error response:
{"success": False, "error": error_message, "session_id": session_id}
```

**Strengths**: Predictable client interface
**Issues**: Could be more structured (error codes, categories)

## Data Processing Patterns

### **DataFrame Operations**

**Files**: Throughout `tools/*_operations.py`

**Pattern**: **Direct Pandas Manipulation**

```python
# Example from data_operations.py:
def filter_dataframe(df: pd.DataFrame, conditions: list[FilterCondition]) -> pd.DataFrame:
    filtered_df = df.copy()
    for condition in conditions:
        mask = create_filter_mask(filtered_df, condition)
        filtered_df = filtered_df[mask]
    return filtered_df
```

**Strengths**:

- Leverages pandas functionality fully
- Efficient for CSV data
- Well-tested patterns

**Limitations**:

- **Format assumptions**: Assumes rectangular data
- **Memory bound**: Loads full dataset
- **No streaming**: Can't handle files larger than memory

### **Validation Pattern**

**File**: `src/databeak/tools/validation.py`

**Pattern**: **Schema-based Validation**

```python
def validate_data_schema(df: pd.DataFrame, schema: dict) -> ValidationResult:
    results = []
    for column, rules in schema.items():
        result = validate_column(df[column], rules)
        results.append(result)
    return ValidationResult(results)
```

**Strengths**: Extensible validation framework
**Opportunity**: Could be extended for format-specific validation

## Current Dependencies & Coupling Analysis

### **High Coupling Areas**

1. **Session → pandas**: CSVSession directly manipulates DataFrames
2. **Tools → CSVSession**: All tools assume CSV session type
3. **Operations → pandas**: Direct DataFrame API usage throughout

### **Low Coupling Areas**

1. **MCP layer**: Well isolated from business logic
2. **Error handling**: Consistent patterns across modules
3. **Configuration**: Centralized settings management

### **Abstraction Opportunities**

1. **Data source abstraction**: Hide format details from sessions
2. **Operation interfaces**: Abstract DataFrame operations for other formats
3. **Result standardization**: Consistent return types across all operations

---

This analysis reveals a well-architected system with clear separation of concerns
but limited by CSV-centric assumptions. The proposed refactoring will transform
it into a flexible, multi-format data platform while preserving existing
strengths.
