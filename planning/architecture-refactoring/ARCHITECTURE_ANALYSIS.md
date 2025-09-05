# DataBeak Architecture Analysis & Refactoring Plan

## Executive Summary

**Objective**: Transform DataBeak from a CSV-focused tool into a comprehensive
data manipulation platform supporting multiple formats (XLSX, databases) while
maintaining session-based design and MCP integration.

**Current State**: Well-architected MCP server with strong type safety and
session management, but limited by CSV-centric design and some structural
redundancies.

**Recommendation**: Implement data source abstraction layer and modular
architecture patterns to enable seamless multi-format support without
breaking existing functionality.

## Current Architecture Assessment

### 📊 **Codebase Overview**

- **Total**: ~8,789 lines across 34 Python files
- **Session-based design**: Stateful operations with lifecycle management
- **MCP integration**: 40+ tools across 7 functional modules
- **Type safety**: Comprehensive Pydantic models and type hints
- **Testing**: ~60% coverage with focus on critical paths

### 🏗️ **Core Components Analysis**

#### **1. Session Management (`models/`)**

**Strengths**:

- Well-designed lifecycle management with TTL and cleanup
- Comprehensive auto-save strategies (overwrite, backup, versioned)
- History tracking with undo/redo capabilities
- Multi-session coordination with isolation

**Issues Identified**:

- **Monolithic `CSVSession`**: 415 lines with multiple responsibilities
- **Tight pandas coupling**: Direct DataFrame dependencies throughout
- **Format assumptions**: CSV-specific naming and error messages
- **Configuration complexity**: Settings spread across multiple classes

#### **2. Tool Layer (`tools/`)**

**Current Organization**:

```
tools/
├── mcp_*_tools.py      # FastMCP wrapper modules (7 files)
├── *_operations.py     # Business logic modules (7 files)
├── data_operations.py  # Core utility functions
└── registry.py         # Manual tool registration
```

**Strengths**:

- Clear separation between MCP interface and business logic
- Modular organization by functionality (IO, analytics, validation)
- Consistent error handling patterns
- Comprehensive parameter validation

**Redundancies Found**:

- **Repetitive MCP registration**: ~200 lines of boilerplate across modules
- **Similar error handling**: Identical try/catch patterns in each tool
- **Parameter validation**: Duplicate session retrieval and validation logic
- **Response formatting**: Repeated success/error response construction

#### **3. Data Models (`models/data_models.py`)**

**Strengths**:

- Comprehensive type definitions with 20+ Pydantic models
- Consistent use of enums for controlled values
- Good separation between data types and business logic

**Issues**:

- **Large single file**: 400+ lines could be better organized
- **CSV-centric types**: `CellValue`, `RowData` assume CSV structure
- **Limited format support**: No models for Excel sheets, database schemas

## Refactoring Opportunities

### 🎯 **High-Impact, Low-Risk Improvements**

#### **1. Tool Registration Abstraction**

*Effort: 2-3 days | Impact: High*

**Current Problem**: Manual registration boilerplate

```python
# Repeated in every mcp_*_tools.py file
@mcp.tool
async def filter_rows(
    ctx: Context,
    session_id: str,
    conditions: list[dict[str, Any]],
    logic: str = "AND"
) -> dict[str, Any]:
    try:
        session = get_session(session_id)
        # ... validation logic ...
        result = await filter_rows_operation(session, conditions, logic)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Proposed Solution**: Decorator-based auto-registration

```python
# New: tools/decorators.py
@register_mcp_tool("data", description="Filter DataFrame rows")
async def filter_rows_operation(
    session_id: str,
    conditions: list[FilterCondition],
    logic: LogicOperator = "AND"
) -> FilterResult:
    session = get_session(session_id)
    # Pure business logic here
    return await _filter_rows_impl(session, conditions, logic)

# Auto-generates MCP wrapper with:
# - Session retrieval and validation
# - Standardized error handling
# - Response formatting
# - Parameter validation
```

**Benefits**:

- Eliminate ~200 lines of boilerplate
- Consistent error handling across all tools
- Easier to add new tools
- Better separation of MCP concerns from business logic

#### **2. Format Constants Consolidation**

*Effort: 1 day | Impact: Medium*

**Issue**: Format enums scattered across files

```python
# Found in multiple files:
class ExportFormat(str, Enum): ...
class ValidationFormat(str, Enum): ...
# Different enums with overlapping values
```

**Solution**: Unified format system

```python
# New: src/databeak/formats/constants.py
class DataFormat(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    JSON = "json"
    DATABASE = "database"

class FormatCapabilities(BaseModel):
    can_read: bool
    can_write: bool
    supports_sheets: bool
    supports_streaming: bool
```

### 🏗️ **Structural Improvements**

#### **3. Data Source Abstraction Layer**

*Effort: 2-3 weeks | Impact: Very High*

**Current Limitation**: Hard-coded pandas operations

```python
# Throughout codebase:
df = pd.read_csv(file_path)  # Only CSV supported for input
session.data_session.df = df  # Direct DataFrame assignment
```

**Proposed Architecture**:

```python
# New: src/databeak/sources/base.py
class DataSource(Protocol):
    """Abstract interface for all data sources"""

    async def load(self, config: SourceConfig) -> pd.DataFrame:
        """Load data from source into DataFrame"""

    async def save(self, df: pd.DataFrame, config: TargetConfig) -> SaveResult:
        """Save DataFrame to target location"""

    async def get_schema(self, config: SourceConfig) -> DataSchema:
        """Get schema information without loading data"""

    def get_capabilities(self) -> FormatCapabilities:
        """Return what operations this source supports"""

    def supports_streaming(self) -> bool:
        """Whether source supports chunked processing"""

# Implementations:
class CSVDataSource(DataSource): ...
class ExcelDataSource(DataSource): ...
class DatabaseDataSource(DataSource): ...
class ParquetDataSource(DataSource): ...
```

**Session Refactoring**:

```python
# Refactored: models/data_session.py
class DataSession:
    def __init__(self, source: DataSource, session_id: str):
        self.source = source
        self.session_id = session_id
        self.df: pd.DataFrame | None = None
        self.metadata: dict[str, Any] = {}

    async def load_data(self, config: SourceConfig) -> LoadResult:
        self.df = await self.source.load(config)
        return LoadResult(success=True, rows=len(self.df), ...)

# Format-specific sessions extend base:
class ExcelSession(DataSession):
    def __init__(self, source: ExcelDataSource, session_id: str):
        super().__init__(source, session_id)
        self.active_sheet: str | None = None
        self.available_sheets: list[str] = []
```

## Multi-Format Support Design

### 📊 **XLSX Integration Plan**

#### **Phase 1: Basic Excel Support**

```python
# New: tools/mcp_excel_tools.py
@register_mcp_tool("io")
async def load_excel(
    file_path: str,
    sheet_name: str | None = None,
    header: int = 0
) -> ExcelLoadResult:
    """Load Excel file with sheet selection"""

@register_mcp_tool("io")
async def list_excel_sheets(file_path: str) -> SheetsResult:
    """List available sheets in Excel file"""

@register_mcp_tool("io")
async def switch_excel_sheet(
    session_id: str,
    sheet_name: str
) -> SheetSwitchResult:
    """Switch active sheet in Excel session"""
```

#### **Phase 2: Advanced Excel Features**

```python
@register_mcp_tool("excel")
async def get_excel_metadata(file_path: str) -> ExcelMetadata:
    """Get workbook metadata, sheet info, named ranges"""

@register_mcp_tool("excel")
async def create_excel_workbook(
    session_id: str,
    sheets: dict[str, pd.DataFrame]
) -> WorkbookResult:
    """Create multi-sheet Excel workbook from session data"""
```

### 🗄️ **Database Integration Design**

#### **Connection Management**

```python
# New: src/databeak/sources/database/
class DatabaseConfig(BaseModel):
    connection_string: str
    engine_type: DatabaseEngine
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

class DatabaseSource(DataSource):
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_async_engine(config.connection_string)

    async def load(self, query: str, params: dict = None) -> pd.DataFrame:
        async with self.engine.begin() as conn:
            return await asyncio.to_thread(
                pd.read_sql, query, conn, params=params
            )
```

#### **Database Tools**

```python
# New: tools/mcp_database_tools.py
@register_mcp_tool("database")
async def connect_database(
    connection_string: str,
    alias: str = "default"
) -> ConnectionResult:
    """Establish database connection"""

@register_mcp_tool("database")
async def query_database(
    connection_alias: str,
    query: str,
    params: dict[str, Any] | None = None
) -> QueryResult:
    """Execute SQL query and load results"""

@register_mcp_tool("database")
async def list_tables(connection_alias: str) -> TablesResult:
    """List available tables in database"""

@register_mcp_tool("database")
async def get_table_schema(
    connection_alias: str,
    table_name: str
) -> SchemaResult:
    """Get table column information and constraints"""
```

## Implementation Roadmap

### **Phase 1: Foundation (2-3 weeks)**

#### **Week 1: Code Organization**

- [ ] Extract tool registration decorator system
- [ ] Consolidate format constants and enums
- [ ] Refactor data models into logical modules
- [ ] Add comprehensive interface documentation

#### **Week 2: Data Source Abstraction**

- [ ] Implement `DataSource` protocol and base classes
- [ ] Refactor `CSVDataSource` from existing code
- [ ] Update session management to use data sources
- [ ] Add plugin discovery mechanism

#### **Week 3: Testing & Validation**

- [ ] Comprehensive tests for new abstractions
- [ ] Backward compatibility validation
- [ ] Performance regression testing
- [ ] Update documentation and examples

### **Phase 2: Format Extensions (3-4 weeks)**

#### **Week 4-5: Excel Integration**

- [ ] Implement `ExcelDataSource` with sheet support
- [ ] Add Excel-specific MCP tools (sheet listing, switching)
- [ ] Extend session management for multi-sheet operations
- [ ] Add Excel validation and metadata tools

#### **Week 6-7: Database Connectivity**

- [ ] Implement database connection management
- [ ] Add SQL query execution with parameter binding
- [ ] Implement table listing and schema introspection
- [ ] Add database-specific error handling and validation

### **Phase 3: Advanced Features (2-3 weeks)**

#### **Week 8-9: Streaming & Performance**

- [ ] Implement chunked processing for large datasets
- [ ] Add progress tracking for long-running operations
- [ ] Memory optimization and async I/O improvements
- [ ] Connection pooling for database sources

#### **Week 10: Integration & Polish**

- [ ] Cross-format operations (CSV → Excel → Database workflows)
- [ ] Enhanced configuration management
- [ ] Performance monitoring and logging
- [ ] Final testing and documentation

## Success Metrics

### **Technical Metrics**

- [ ] Support XLSX files as first-class input format
- [ ] Database connectivity with 3+ engines (PostgreSQL, SQLite, MySQL)
- [ ] Memory usage <50% of current for equivalent operations
- [ ] Build time remains <1 second
- [ ] Test coverage maintains >80%

### **User Experience Metrics**

- [ ] Same or fewer commands needed for common operations
- [ ] Backward compatibility with all existing CSV workflows
- [ ] Clear error messages for format-specific issues
- [ ] Seamless switching between data sources in same session

### **Code Quality Metrics**

- [ ] Reduced code duplication by >30%
- [ ] Improved modularity (lower coupling scores)
- [ ] Enhanced testability (isolated components)
- [ ] Simplified maintenance (fewer files to touch for changes)

## Regression Testing Strategy

### **⚠️ MANDATORY: Pre-Refactoring Test Requirements**

**Every refactoring phase MUST begin with establishing comprehensive regression
tests to ensure existing functionality remains intact.**

#### **Required Test Coverage Before Refactoring**

```bash
# Must achieve before starting any refactoring work:
uv run pytest --cov=src/databeak --cov-fail-under=90
uv run pytest tests/ -v --tb=short  # All tests passing
uv run pytest tests/integration/ -v # End-to-end workflows verified
```

#### **Specific Test Requirements by Component**

##### **Tool Registration Refactoring (Week 1)**

- [ ] **All 40+ MCP tools**: Input/output signature tests
- [ ] **Error handling**: Consistent error response format validation
- [ ] **Session interaction**: Tool→session→operation flow testing
- [ ] **Parameter validation**: All tool parameter combinations

##### **Session Management Refactoring (Week 2)**

- [ ] **Session lifecycle**: Creation, TTL, cleanup functionality
- [ ] **Data loading**: All current CSV loading scenarios
- [ ] **Auto-save system**: All strategies (overwrite, backup, versioned)
- [ ] **History management**: Undo/redo operations and operation recording
- [ ] **Multi-session**: Concurrent session isolation and management

##### **Data Operations Refactoring**

- [ ] **Core operations**: Filter, sort, transform, aggregate functions
- [ ] **Data validation**: Schema validation and quality checking
- [ ] **Export functions**: All current export formats and options
- [ ] **Analytics**: Statistics, correlations, outlier detection

#### **Integration Test Requirements**

```python
# Required end-to-end workflow tests:
class TestRegressionWorkflows:
    async def test_complete_csv_workflow(self):
        """Load CSV → filter → analyze → export workflow unchanged"""

    async def test_session_management_workflow(self):
        """Multi-session creation, operations, and cleanup unchanged"""

    async def test_history_workflow(self):
        """Auto-save, undo/redo, restore operations unchanged"""

    async def test_error_handling_workflow(self):
        """Error conditions produce consistent responses"""
```

#### **Performance Baseline Requirements**

```python
# Required performance baselines before refactoring:
class PerformanceBaseline:
    # Tool execution times (must not regress >10%)
    TOOL_EXECUTION_LIMITS = {
        "load_csv": 100,        # ms for 1MB file
        "filter_rows": 50,      # ms for 10K rows
        "get_statistics": 200,  # ms for 100K rows
        "export_csv": 150,      # ms for 10K rows
    }

    # Memory usage limits (must not increase >20%)
    MEMORY_USAGE_LIMITS = {
        "session_creation": 10,     # MB baseline
        "data_loading": 2.0,        # multiplier of file size
        "processing_overhead": 1.5   # multiplier of data size
    }
```

## Risk Mitigation Strategies

### **High-Risk Areas**

1. **Session Management Changes**: Use adapter pattern to maintain compatibility
2. **Tool Interface Updates**: Maintain existing signatures, add new overloads
3. **Performance Regressions**: Continuous benchmarking during refactoring

### **Rollback Plans**

- **Feature flags**: Enable/disable new data sources
- **Gradual rollout**: Phase in new formats without removing CSV support
- **Comprehensive testing**: Automated tests for all format combinations
- **Version compatibility**: Maintain v1.x API during transition

### **🔒 Regression Prevention Protocol**

#### **Mandatory Steps for Each Refactoring**

1. **Pre-work**: Establish >90% test coverage for target area
2. **During work**: Run tests continuously (`uv run pytest --lf`)
3. **Post-work**: Full regression suite + performance validation
4. **Documentation**: Update tests to reflect new patterns
5. **Review**: Code review focusing on backward compatibility

#### **Continuous Validation**

```bash
# Required validation pipeline:
uv run pytest                    # All functionality tests
uv run pytest --benchmark-only  # Performance regression detection
uv run pytest tests/integration/ # End-to-end workflow validation
uv run mypy src/                 # Type safety maintained
uv run all-checks               # Code quality standards
```

## Next Steps

1. **Get stakeholder approval** for refactoring approach and timeline
2. **Create feature branch** for architecture refactoring work
3. **Begin Phase 1** with tool registration decorator implementation
4. **Set up feature flags** for gradual rollout of new capabilities
5. **Establish performance baselines** for regression detection

---

**Conclusion**: This refactoring plan provides a clear path to transform DataBeak
into a multi-format data platform while preserving its strengths in session
management, type safety, and MCP integration. The phased approach minimizes
risk while delivering incremental value to users.
