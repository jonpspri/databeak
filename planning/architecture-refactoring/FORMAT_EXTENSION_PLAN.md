# XLSX and Database Integration Plan

## Strategic Approach

Transform DataBeak from CSV-focused to multi-format data platform through
incremental enhancements that maintain backward compatibility while adding
first-class support for Excel files and relational databases.

## XLSX Integration Strategy

### **Phase 1: Basic Excel Support**

*Timeline: 1-2 weeks*

#### **Core Excel Operations**

```python
# New MCP tools for Excel files
load_excel(file_path, sheet_name=None, header=0) -> session_id
list_excel_sheets(file_path) -> list[sheet_names]
switch_excel_sheet(session_id, sheet_name) -> result
export_to_excel(session_id, file_path, sheet_name=None) -> result
```

#### **Implementation Details**

```python
# src/databeak/sources/excel.py
class ExcelDataSource:
    async def load(self, config: ExcelSourceConfig) -> pd.DataFrame:
        """Load Excel file with openpyxl engine for full compatibility"""
        return pd.read_excel(
            config.file_path,
            sheet_name=config.sheet_name,
            header=config.header,
            skiprows=config.skip_rows,
            engine="openpyxl",  # Full .xlsx support
            dtype_backend="pyarrow"  # Better type inference
        )

    async def get_workbook_info(self, file_path: str) -> WorkbookInfo:
        """Extract workbook metadata without loading data"""
        with pd.ExcelFile(file_path, engine="openpyxl") as xls:
            sheets_info = []
            for sheet_name in xls.sheet_names:
                # Peek at each sheet for metadata
                sample = pd.read_excel(xls, sheet_name=sheet_name, nrows=5)
                sheets_info.append(SheetInfo(
                    name=sheet_name,
                    columns=list(sample.columns),
                    estimated_rows=self._estimate_sheet_rows(xls, sheet_name),
                    data_types=sample.dtypes.to_dict()
                ))

            return WorkbookInfo(
                file_path=file_path,
                file_size_mb=os.path.getsize(file_path) / (1024 * 1024),
                sheets=sheets_info,
                created_date=datetime.fromtimestamp(os.path.getctime(file_path))
            )
```

### **Phase 2: Advanced Excel Features**

*Timeline: 2-3 weeks*

#### **Multi-Sheet Operations**

```python
# Advanced Excel tools
merge_excel_sheets(session_id, sheet_names, how="concat") -> result
create_excel_workbook(data_sessions: dict[str, session_id], file_path) -> result
copy_sheet_formatting(source_file, target_session_id) -> result
```

#### **Excel-Specific Analysis**

```python
# Excel format awareness in existing tools
@register_mcp_tool("analytics", supported_formats=[DataFormat.EXCEL, DataFormat.CSV])
async def get_statistics(session_id: str) -> StatisticsResult:
    session = get_session(session_id)

    # Excel-specific enhancements
    if isinstance(session, ExcelSession):
        stats = calculate_basic_stats(session.df)
        stats.excel_metadata = ExcelStatsMetadata(
            active_sheet=session.active_sheet,
            workbook_path=session.workbook_metadata.file_path,
            sheet_count=len(session.workbook_metadata.sheets)
        )
        return stats
    else:
        return calculate_basic_stats(session.df)
```

## Database Integration Strategy

### **Phase 1: SQLite Foundation**

*Timeline: 2-3 weeks*

#### **Connection Management**

```python
# src/databeak/sources/database/connection.py
class DatabaseConnectionManager:
    """Manages database connections with pooling"""

    def __init__(self):
        self._connections: dict[str, AsyncEngine] = {}
        self._connection_configs: dict[str, DatabaseConfig] = {}

    async def create_connection(self,
                               alias: str,
                               config: DatabaseConfig) -> ConnectionResult:
        """Create new database connection"""
        try:
            engine = create_async_engine(
                config.connection_string,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                echo=config.debug_sql
            )

            # Test connection
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            self._connections[alias] = engine
            self._connection_configs[alias] = config

            return ConnectionResult(
                success=True,
                alias=alias,
                engine_type=config.engine_type,
                database_name=self._extract_database_name(config.connection_string)
            )

        except Exception as e:
            return ConnectionResult(
                success=False,
                error=f"Failed to connect: {str(e)}"
            )
```

#### **Core Database Tools**

```python
# New: src/databeak/tools/mcp_database_tools.py
@register_mcp_tool("database", "Connect to database", [DataFormat.POSTGRESQL, DataFormat.SQLITE])
async def connect_database(
    connection_string: str,
    alias: str = "default",
    pool_size: int = 5
) -> ConnectionResult:
    """Establish database connection with connection pooling"""

@register_mcp_tool("database", "Execute SQL query")
async def query_database(
    connection_alias: str,
    query: str,
    params: dict[str, Any] | None = None,
    create_session: bool = True
) -> QueryResult:
    """Execute SQL query and optionally create session with results"""

@register_mcp_tool("database", "List database tables")
async def list_database_tables(
    connection_alias: str,
    schema: str | None = None
) -> TablesResult:
    """List all tables in database schema"""

@register_mcp_tool("database", "Get table schema information")
async def describe_table(
    connection_alias: str,
    table_name: str,
    schema: str | None = None
) -> TableSchemaResult:
    """Get detailed table schema with column types and constraints"""
```

### **Phase 2: Advanced Database Features**

*Timeline: 2-3 weeks*

#### **Query Builder Integration**

```python
# New: src/databeak/tools/sql_builder.py
class SQLQueryBuilder:
    """Helper for building safe SQL queries"""

    def __init__(self, dialect: str):
        self.dialect = dialect

    def build_select(self,
                    table: str,
                    columns: list[str] | None = None,
                    where_conditions: list[FilterCondition] | None = None,
                    limit: int | None = None) -> str:
        """Build SELECT query with proper escaping"""

    def build_insert(self,
                    table: str,
                    data: pd.DataFrame,
                    on_conflict: str = "error") -> str:
        """Build INSERT query from DataFrame"""

    def validate_query_safety(self, query: str) -> ValidationResult:
        """Validate query for safety (prevent DROP, DELETE without WHERE, etc.)"""
```

#### **Data Export to Database**

```python
@register_mcp_tool("database", "Export session data to database table")
async def export_to_database(
    session_id: str,
    connection_alias: str,
    table_name: str,
    if_exists: str = "append",  # append, replace, fail
    schema: str | None = None
) -> ExportResult:
    """Export current session DataFrame to database table"""

@register_mcp_tool("database", "Execute bulk insert")
async def bulk_insert_database(
    connection_alias: str,
    table_name: str,
    data: list[dict[str, Any]],
    batch_size: int = 1000
) -> BulkInsertResult:
    """Efficient bulk insert of data into database"""
```

### **Phase 3: Cross-Format Workflows**

*Timeline: 1-2 weeks*

#### **Format Bridge Operations**

```python
@register_mcp_tool("workflow", "Transfer data between sessions")
async def transfer_data(
    source_session_id: str,
    target_session_id: str,
    transformation: str | None = None
) -> TransferResult:
    """Transfer data from one session format to another"""

@register_mcp_tool("workflow", "Create analysis pipeline")
async def create_analysis_pipeline(
    input_sources: list[SourceConfig],
    output_target: TargetConfig,
    operations: list[PipelineStep]
) -> PipelineResult:
    """Execute multi-step analysis across different data sources"""
```

## Configuration Enhancement

### **Multi-Format Settings**

```python
# Enhanced: src/databeak/config.py
class ExcelSettings(BaseModel):
    default_engine: str = "openpyxl"
    max_sheets_per_workbook: int = 100
    default_header_row: int = 0
    auto_detect_types: bool = True

class DatabaseSettings(BaseModel):
    default_pool_size: int = 5
    max_pool_overflow: int = 10
    connection_timeout: int = 30
    query_timeout: int = 300
    enable_query_logging: bool = False

class DataBeakSettings(BaseSettings):
    # Existing CSV settings
    csv_history_dir: str = "."
    max_file_size_mb: int = 1024

    # New format settings
    excel: ExcelSettings = ExcelSettings()
    database: DatabaseSettings = DatabaseSettings()

    # Multi-format settings
    default_chunk_size: int = 10000
    enable_cross_format_operations: bool = True

    class Config:
        env_prefix = "DATABEAK_"
        env_nested_delimiter = "__"
        # Enables DATABEAK_EXCEL__MAX_SHEETS_PER_WORKBOOK=50
```

## Error Handling Enhancement

### **Format-Specific Exceptions**

```python
# Enhanced: src/databeak/exceptions.py
class FormatError(DataBeakError):
    """Base class for format-specific errors"""
    def __init__(self, format_type: DataFormat, message: str):
        self.format_type = format_type
        super().__init__(f"[{format_type.upper()}] {message}")

class ExcelError(FormatError):
    """Excel-specific errors"""
    def __init__(self, message: str):
        super().__init__(DataFormat.EXCEL, message)

class SheetNotFoundError(ExcelError):
    def __init__(self, sheet_name: str, available_sheets: list[str]):
        super().__init__(
            f"Sheet '{sheet_name}' not found. Available: {', '.join(available_sheets)}"
        )
        self.sheet_name = sheet_name
        self.available_sheets = available_sheets

class DatabaseError(FormatError):
    """Database-specific errors"""
    def __init__(self, message: str):
        super().__init__(DataFormat.DATABASE, message)

class SQLExecutionError(DatabaseError):
    def __init__(self, query: str, original_error: Exception):
        super().__init__(f"SQL execution failed: {original_error}")
        self.query = query
        self.original_error = original_error
```

## Performance Optimization Plans

### **Memory Efficiency for Large Files**

```python
# New: src/databeak/sources/streaming.py
class StreamingMixin:
    """Mixin for data sources that support streaming"""

    async def process_in_chunks(self,
                               config: SourceConfig,
                               operation: Callable[[pd.DataFrame], pd.DataFrame],
                               output_config: TargetConfig,
                               chunk_size: int = 10000) -> StreamingResult:
        """Process large datasets in chunks without loading everything"""

        total_rows = 0
        chunk_count = 0

        async for chunk in self.load_chunked(config, chunk_size):
            processed_chunk = operation(chunk)
            await self.save_chunk(processed_chunk, output_config, chunk_count)
            total_rows += len(processed_chunk)
            chunk_count += 1

        return StreamingResult(
            success=True,
            total_rows=total_rows,
            chunks_processed=chunk_count,
            memory_efficient=True
        )

# Usage in tools:
@register_mcp_tool("data", "Process large file with streaming")
async def stream_process_file(
    input_file: str,
    output_file: str,
    operation: str,  # "filter", "transform", etc.
    chunk_size: int = 10000
) -> StreamingResult:
    """Process files larger than memory using streaming"""
```

### **Database Query Optimization**

```python
# Database-specific optimizations
class DatabaseDataSource:
    async def load_with_pagination(self,
                                  base_query: str,
                                  page_size: int = 10000,
                                  offset: int = 0) -> pd.DataFrame:
        """Load database results with pagination for large tables"""

    async def get_table_sample(self,
                              table_name: str,
                              sample_size: int = 1000,
                              random: bool = True) -> pd.DataFrame:
        """Get representative sample of large table"""

    async def execute_explain_plan(self, query: str) -> QueryPlan:
        """Analyze query performance before execution"""
```

## Testing Strategy for Multi-Format Support

### **Test Data Management**

```python
# New: tests/fixtures/multi_format_data.py
class MultiFormatTestData:
    """Test data available in multiple formats for consistency testing"""

    @classmethod
    def get_sales_data(cls) -> dict[DataFormat, SourceConfig]:
        """Same sales dataset in different formats"""
        return {
            DataFormat.CSV: CSVSourceConfig(file_path="tests/data/sales.csv"),
            DataFormat.EXCEL: ExcelSourceConfig(
                file_path="tests/data/sales.xlsx",
                sheet_name="Sales"
            ),
            DataFormat.SQLITE: DatabaseSourceConfig(
                connection_string="sqlite:///tests/data/test.db",
                table_name="sales"
            )
        }

    @classmethod
    async def ensure_test_data_consistency(cls) -> None:
        """Verify all format versions contain identical data"""
        # Implementation to validate test data consistency
```

### **Cross-Format Testing**

```python
# tests/test_cross_format_operations.py
class TestCrossFormatOperations:
    """Test that operations produce identical results across formats"""

    @pytest.mark.parametrize("format_type", [DataFormat.CSV, DataFormat.EXCEL])
    async def test_filter_operations_consistent(self, format_type: DataFormat):
        """Verify filtering produces same results regardless of source format"""

    async def test_csv_to_excel_workflow(self):
        """Test complete workflow: CSV input → processing → Excel output"""

    async def test_database_to_csv_export(self):
        """Test database query results exported to CSV"""
```

## Migration Path for Existing Users

### **Backward Compatibility Guarantees**

```python
# All existing CSV-focused tools maintain identical signatures
load_csv(file_path, ...) -> session_id           # ✅ Unchanged
filter_rows(session_id, conditions, ...) -> ...   # ✅ Unchanged
export_csv(session_id, format, ...) -> ...       # ✅ Unchanged

# New tools are additive, not replacements
load_excel(file_path, sheet_name, ...) -> ...    # ✅ New, optional
query_database(connection, query, ...) -> ...    # ✅ New, optional
```

### **Gradual Adoption Path**

1. **Phase 1**: Users can continue using existing CSV workflows
1. **Phase 2**: Excel tools available as optional enhancement
1. **Phase 3**: Database connectivity for advanced users
1. **Phase 4**: Cross-format workflows for power users

## Environment Variable Extensions

### **New Configuration Options**

```bash
# Excel settings
DATABEAK_EXCEL__DEFAULT_ENGINE=openpyxl
DATABEAK_EXCEL__MAX_SHEETS_PER_WORKBOOK=100
DATABEAK_EXCEL__AUTO_DETECT_TYPES=true

# Database settings
DATABEAK_DATABASE__DEFAULT_POOL_SIZE=5
DATABEAK_DATABASE__CONNECTION_TIMEOUT=30
DATABEAK_DATABASE__QUERY_TIMEOUT=300
DATABEAK_DATABASE__ENABLE_QUERY_LOGGING=false

# Cross-format settings
DATABEAK_DEFAULT_CHUNK_SIZE=10000
DATABEAK_ENABLE_CROSS_FORMAT_OPERATIONS=true
```

## Security Considerations

### **Database Security**

```python
# src/databeak/sources/database/security.py
class DatabaseSecurityManager:
    """Security controls for database operations"""

    def validate_connection_string(self, conn_str: str) -> ValidationResult:
        """Ensure connection string doesn't contain credentials in plain text"""

    def validate_query_safety(self, query: str) -> QuerySafetyResult:
        """Prevent dangerous SQL operations"""
        dangerous_patterns = [
            r"DROP\s+(TABLE|DATABASE|SCHEMA)",
            r"DELETE\s+FROM\s+\w+\s*(?!WHERE)",  # DELETE without WHERE
            r"TRUNCATE\s+TABLE",
            r"ALTER\s+TABLE.*DROP"
        ]

    async def check_permissions(self,
                               connection: AsyncEngine,
                               operation: str,
                               table_name: str) -> PermissionResult:
        """Verify user has required permissions for operation"""
```

### **File Security**

```python
# Enhanced file validation for Excel
class ExcelSecurityValidator:
    def validate_excel_file(self, file_path: str) -> SecurityResult:
        """Check Excel file for security issues"""
        # Check file size limits
        # Validate file is actual Excel format
        # Scan for macros or external links
        # Verify no embedded objects
```

## Performance Benchmarking Plan

### **Baseline Metrics**

Current CSV performance baselines to maintain:

- **Small files** (\<1MB): \<100ms processing
- **Medium files** (1-100MB): \<5s processing
- **Large files** (100MB-1GB): \<30s with chunking
- **Memory usage**: \<2x file size for processing

### **Multi-Format Performance Targets**

- **Excel files**: \<150ms for small files (50% overhead acceptable)
- **Database queries**: \<500ms for typical queries (\<10K rows)
- **Cross-format operations**: \<20% overhead vs single-format
- **Streaming operations**: Process 10GB+ files with \<1GB memory

### **Monitoring and Metrics**

```python
# New: src/databeak/monitoring/performance.py
class PerformanceMonitor:
    """Track operation performance across formats"""

    async def track_operation(self,
                             operation_name: str,
                             format_type: DataFormat,
                             file_size_mb: float,
                             execution_time_ms: float,
                             memory_usage_mb: float) -> None:
        """Record performance metrics for analysis"""

    def get_performance_report(self,
                              operation: str | None = None,
                              format_type: DataFormat | None = None) -> PerformanceReport:
        """Generate performance analysis report"""
```

This comprehensive format extension plan ensures DataBeak can evolve into a
multi-format data platform while maintaining its current strengths and providing
clear migration paths for existing users.
