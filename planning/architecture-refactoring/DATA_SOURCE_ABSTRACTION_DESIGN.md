# Data Source Abstraction Layer Design

## Overview

Design for a flexible data source abstraction layer that enables DataBeak to
support multiple input/output formats while maintaining the current
session-based architecture and MCP tool interface.

## Core Abstraction Design

### **DataSource Protocol**

```python
# New: src/databeak/sources/base.py
from abc import ABC, abstractmethod
from typing import Protocol, AsyncIterator
import pandas as pd

class DataSource(Protocol):
    """Abstract interface for all data sources in DataBeak"""

    @property
    def format_type(self) -> DataFormat:
        """Format type identifier"""
        ...

    @property
    def capabilities(self) -> FormatCapabilities:
        """What operations this source supports"""
        ...

    async def load(self, config: SourceConfig) -> pd.DataFrame:
        """Load data from source into DataFrame"""
        ...

    async def save(self, df: pd.DataFrame, config: TargetConfig) -> SaveResult:
        """Save DataFrame to target location"""
        ...

    async def get_metadata(self, config: SourceConfig) -> SourceMetadata:
        """Get source metadata without loading data"""
        ...

    async def validate_source(self, config: SourceConfig) -> ValidationResult:
        """Validate source accessibility and format"""
        ...

    def supports_streaming(self) -> bool:
        """Whether source supports chunked processing"""
        ...

    async def load_chunked(self,
                          config: SourceConfig,
                          chunk_size: int) -> AsyncIterator[pd.DataFrame]:
        """Load data in chunks for memory efficiency"""
        ...
```

### **Supporting Types**

```python
# New: src/databeak/sources/types.py
class DataFormat(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    JSON = "json"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"

class FormatCapabilities(BaseModel):
    can_read: bool
    can_write: bool
    supports_sheets: bool           # Excel workbooks
    supports_schemas: bool          # Database schemas/tables
    supports_streaming: bool        # Chunked processing
    supports_transactions: bool     # Database transactions
    max_file_size_mb: int | None   # Size limits

class SourceConfig(BaseModel):
    """Base configuration for data sources"""
    source_type: DataFormat

class FileSourceConfig(SourceConfig):
    """Configuration for file-based sources"""
    file_path: str
    encoding: str = "utf-8"

class ExcelSourceConfig(FileSourceConfig):
    """Excel-specific configuration"""
    sheet_name: str | None = None
    header: int | list[int] | None = 0
    skip_rows: int = 0

class DatabaseSourceConfig(SourceConfig):
    """Database-specific configuration"""
    connection_string: str
    query: str | None = None
    table_name: str | None = None
    schema: str | None = None
```

## Format-Specific Implementations

### **1. Enhanced CSV Source**

```python
# Refactored: src/databeak/sources/csv.py
class CSVDataSource:
    """Enhanced CSV data source with streaming support"""

    format_type = DataFormat.CSV

    def __init__(self):
        self.capabilities = FormatCapabilities(
            can_read=True,
            can_write=True,
            supports_sheets=False,
            supports_schemas=False,
            supports_streaming=True,
            supports_transactions=False,
            max_file_size_mb=None  # No limit with streaming
        )

    async def load(self, config: CSVSourceConfig) -> pd.DataFrame:
        return pd.read_csv(
            config.file_path,
            encoding=config.encoding,
            dtype_backend="pyarrow"  # Better null handling
        )

    async def load_chunked(self, config: CSVSourceConfig,
                          chunk_size: int) -> AsyncIterator[pd.DataFrame]:
        for chunk in pd.read_csv(config.file_path, chunksize=chunk_size):
            yield chunk

    async def get_metadata(self, config: CSVSourceConfig) -> CSVMetadata:
        # Use pandas to peek at file info without full load
        sample = pd.read_csv(config.file_path, nrows=100)
        return CSVMetadata(
            columns=sample.columns.tolist(),
            dtypes=sample.dtypes.to_dict(),
            estimated_rows=self._estimate_row_count(config.file_path),
            file_size_mb=os.path.getsize(config.file_path) / 1024 / 1024
        )
```

### **2. Excel Source Implementation**

```python
# New: src/databeak/sources/excel.py
class ExcelDataSource:
    """Excel/XLSX data source with multi-sheet support"""

    format_type = DataFormat.EXCEL

    def __init__(self):
        self.capabilities = FormatCapabilities(
            can_read=True,
            can_write=True,
            supports_sheets=True,   # Multi-sheet workbooks
            supports_schemas=False,
            supports_streaming=False,  # Excel doesn't support streaming
            supports_transactions=False,
            max_file_size_mb=2048  # Reasonable Excel limit
        )

    async def load(self, config: ExcelSourceConfig) -> pd.DataFrame:
        """Load Excel sheet into DataFrame"""
        return pd.read_excel(
            config.file_path,
            sheet_name=config.sheet_name,
            header=config.header,
            skiprows=config.skip_rows,
            engine="openpyxl"
        )

    async def list_sheets(self, config: ExcelSourceConfig) -> list[str]:
        """List all sheets in Excel workbook"""
        with pd.ExcelFile(config.file_path) as xls:
            return xls.sheet_names

    async def get_sheet_metadata(self,
                                config: ExcelSourceConfig) -> ExcelSheetMetadata:
        """Get metadata for specific sheet"""
        sample = pd.read_excel(config.file_path,
                              sheet_name=config.sheet_name, nrows=10)
        return ExcelSheetMetadata(
            sheet_name=config.sheet_name,
            columns=sample.columns.tolist(),
            dtypes=sample.dtypes.to_dict(),
            estimated_rows=len(sample) * 10  # Rough estimate
        )

    async def get_workbook_metadata(self,
                                   config: ExcelSourceConfig) -> ExcelWorkbookMetadata:
        """Get complete workbook information"""
        sheets = await self.list_sheets(config)
        sheet_metadata = []
        for sheet in sheets:
            sheet_config = config.model_copy(update={"sheet_name": sheet})
            metadata = await self.get_sheet_metadata(sheet_config)
            sheet_metadata.append(metadata)

        return ExcelWorkbookMetadata(
            file_path=config.file_path,
            sheets=sheet_metadata,
            file_size_mb=os.path.getsize(config.file_path) / 1024 / 1024
        )
```

### **3. Database Source Implementation**

```python
# New: src/databeak/sources/database.py
class DatabaseDataSource:
    """Relational database data source with SQL support"""

    def __init__(self, engine_type: DatabaseEngine):
        self.engine_type = engine_type
        self.capabilities = FormatCapabilities(
            can_read=True,
            can_write=True,
            supports_sheets=False,
            supports_schemas=True,  # Database schemas/tables
            supports_streaming=True,  # Chunked queries
            supports_transactions=True,
            max_file_size_mb=None  # No file size limit
        )

    async def load(self, config: DatabaseSourceConfig) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        if config.query:
            return await self._execute_query(config.query, config.params)
        elif config.table_name:
            return await self._load_table(config.table_name, config.schema)
        else:
            raise ValueError("Must specify either query or table_name")

    async def _execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute SQL query with parameter binding"""
        async with self._get_connection() as conn:
            return await asyncio.to_thread(pd.read_sql, query, conn, params=params)

    async def get_tables(self, schema: str = None) -> list[TableInfo]:
        """List available tables in database"""
        query = self._build_tables_query(schema)
        result_df = await self._execute_query(query)
        return [TableInfo.from_dict(row) for row in result_df.to_dict('records')]

    async def get_table_schema(self, table_name: str,
                              schema: str = None) -> DatabaseTableSchema:
        """Get detailed table schema information"""
        query = self._build_schema_query(table_name, schema)
        schema_df = await self._execute_query(query)
        return DatabaseTableSchema.from_dataframe(schema_df)

    async def save(self, df: pd.DataFrame, config: DatabaseTargetConfig) -> SaveResult:
        """Save DataFrame to database table"""
        async with self._get_connection() as conn:
            await asyncio.to_thread(
                df.to_sql,
                config.table_name,
                conn,
                if_exists=config.if_exists,
                index=config.include_index,
                schema=config.schema
            )
        return SaveResult(success=True, rows_affected=len(df))
```

## Session Refactoring Design

### **Format-Agnostic Base Session**

```python
# Refactored: src/databeak/models/base_session.py
class DataSession(ABC):
    """Base class for all data sessions"""

    def __init__(self, source: DataSource, session_id: str):
        self.source = source
        self.session_id = session_id
        self.df: pd.DataFrame | None = None
        self.metadata: SessionMetadata = SessionMetadata()
        self.lifecycle = SessionLifecycle()

    async def load_data(self, config: SourceConfig) -> LoadResult:
        """Load data using the configured source"""
        self.df = await self.source.load(config)
        self.metadata.update_load_info(config, len(self.df))
        self.lifecycle.update_access_time()
        return LoadResult(success=True, rows=len(self.df),
                         columns=list(self.df.columns))

    @abstractmethod
    def get_format_specific_info(self) -> dict[str, Any]:
        """Return format-specific session information"""
        pass
```

### **Format-Specific Session Extensions**

```python
# New: src/databeak/models/excel_session.py
class ExcelSession(DataSession):
    """Excel-specific session with sheet management"""

    def __init__(self, source: ExcelDataSource, session_id: str):
        super().__init__(source, session_id)
        self.workbook_metadata: ExcelWorkbookMetadata | None = None
        self.active_sheet: str | None = None

    async def switch_sheet(self, sheet_name: str) -> SheetSwitchResult:
        """Switch to different sheet in same workbook"""
        if not self.workbook_metadata:
            raise SessionError("No workbook loaded")

        if sheet_name not in self.workbook_metadata.sheet_names:
            raise SessionError(f"Sheet '{sheet_name}' not found")

        # Load new sheet data
        config = ExcelSourceConfig(
            file_path=self.workbook_metadata.file_path,
            sheet_name=sheet_name
        )
        await self.load_data(config)
        self.active_sheet = sheet_name

        return SheetSwitchResult(
            success=True,
            active_sheet=sheet_name,
            rows=len(self.df),
            columns=list(self.df.columns)
        )

    def get_format_specific_info(self) -> dict[str, Any]:
        return {
            "format": "excel",
            "active_sheet": self.active_sheet,
            "available_sheets": self.workbook_metadata.sheet_names if self.workbook_metadata else [],
            "workbook_path": self.workbook_metadata.file_path if self.workbook_metadata else None
        }

# New: src/databeak/models/database_session.py
class DatabaseSession(DataSession):
    """Database-specific session with connection management"""

    def __init__(self, source: DatabaseDataSource, session_id: str):
        super().__init__(source, session_id)
        self.connection_alias: str | None = None
        self.active_table: str | None = None
        self.query_history: list[QueryRecord] = []

    async def execute_query(self, query: str,
                           params: dict = None) -> QueryResult:
        """Execute custom SQL query"""
        config = DatabaseSourceConfig(
            connection_string=self.source.connection_string,
            query=query,
            params=params
        )

        self.df = await self.source.load(config)

        # Record query in session history
        self.query_history.append(QueryRecord(
            query=query,
            params=params,
            timestamp=datetime.utcnow(),
            rows_returned=len(self.df)
        ))

        return QueryResult(
            success=True,
            rows=len(self.df),
            columns=list(self.df.columns),
            query=query
        )

    def get_format_specific_info(self) -> dict[str, Any]:
        return {
            "format": "database",
            "connection_alias": self.connection_alias,
            "active_table": self.active_table,
            "query_history_count": len(self.query_history),
            "database_type": self.source.engine_type.value
        }
```

## Tool Layer Abstraction

### **Generic Tool Registration System**

```python
# New: src/databeak/tools/decorators.py
from typing import Callable, TypeVar, ParamSpec
from functools import wraps

P = ParamSpec('P')
T = TypeVar('T')

def register_mcp_tool(
    category: str,
    description: str | None = None,
    supported_formats: list[DataFormat] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to automatically register MCP tools with standardized wrapper"""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Store metadata for auto-registration
        func._mcp_metadata = MCPToolMetadata(
            category=category,
            description=description or func.__doc__,
            supported_formats=supported_formats or [DataFormat.CSV],
            function_name=func.__name__
        )

        @wraps(func)
        async def mcp_wrapper(ctx: Context, *args, **kwargs) -> dict[str, Any]:
            """Standardized MCP wrapper with error handling"""
            try:
                # Validate session and format compatibility
                if "session_id" in kwargs:
                    session = get_session(kwargs["session_id"])
                    if supported_formats and session.source.format_type not in supported_formats:
                        raise UnsupportedFormatError(
                            f"Tool {func.__name__} doesn't support {session.source.format_type}"
                        )

                # Execute business logic
                result = await func(*args, **kwargs)

                # Record operation if session involved
                if "session_id" in kwargs:
                    session.record_operation(func.__name__, kwargs)

                return {"success": True, "data": result}

            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", extra={"session_id": kwargs.get("session_id")})
                return {"success": False, "error": str(e)}

        return mcp_wrapper
    return decorator

# Usage example:
@register_mcp_tool("data", "Filter rows based on conditions")
async def filter_rows(session_id: str,
                     conditions: list[FilterCondition],
                     logic: LogicOperator = "AND") -> FilterResult:
    """Pure business logic - no MCP boilerplate needed"""
    session = get_session(session_id)
    filtered_df = apply_filters(session.df, conditions, logic)
    session.df = filtered_df
    return FilterResult(rows=len(filtered_df), conditions_applied=len(conditions))
```

### **Auto-Discovery and Registration**

```python
# Enhanced: src/databeak/tools/registry.py
class ToolRegistry:
    """Centralized tool registration and discovery"""

    def __init__(self):
        self._tools_by_category: dict[str, list[MCPToolMetadata]] = defaultdict(list)
        self._tools_by_format: dict[DataFormat, list[MCPToolMetadata]] = defaultdict(list)

    def discover_tools(self, package_name: str = "databeak.tools") -> None:
        """Auto-discover tools with @register_mcp_tool decorator"""
        for module_info in pkgutil.iter_modules(databeak.tools.__path__):
            module = importlib.import_module(f"{package_name}.{module_info.name}")

            for name in dir(module):
                obj = getattr(module, name)
                if hasattr(obj, '_mcp_metadata'):
                    self._register_tool(obj._mcp_metadata, obj)

    def get_tools_for_format(self, format_type: DataFormat) -> list[MCPToolMetadata]:
        """Get all tools compatible with given format"""
        return self._tools_by_format[format_type]

    def validate_tool_compatibility(self, tool_name: str,
                                   session: DataSession) -> bool:
        """Check if tool supports session's data format"""
        # Implementation details...
```

## Multi-Format Session Management

### **Session Factory Pattern**

```python
# New: src/databeak/factories/session_factory.py
class SessionFactory:
    """Factory for creating format-appropriate sessions"""

    def __init__(self, container: DataBeakContainer):
        self.container = container
        self._source_registry = container.get_source_registry()

    async def create_session(self,
                           source_config: SourceConfig,
                           session_id: str | None = None) -> DataSession:
        """Create appropriate session type based on source format"""

        session_id = session_id or generate_session_id()
        source = self._source_registry.create_source(source_config.source_type)

        # Validate source compatibility
        validation = await source.validate_source(source_config)
        if not validation.is_valid:
            raise SourceValidationError(f"Invalid source: {validation.error}")

        # Create format-specific session
        match source_config.source_type:
            case DataFormat.CSV:
                return CSVSession(source, session_id)
            case DataFormat.EXCEL:
                session = ExcelSession(source, session_id)
                # Pre-load workbook metadata for sheet operations
                session.workbook_metadata = await source.get_workbook_metadata(source_config)
                return session
            case DataFormat.POSTGRESQL | DataFormat.SQLITE | DataFormat.MYSQL:
                return DatabaseSession(source, session_id)
            case _:
                # Fallback to generic session
                return DataSession(source, session_id)
```

### **Enhanced Session Manager**

```python
# Refactored: src/databeak/models/session_manager.py
class SessionManager:
    """Manages multiple sessions of different formats"""

    def __init__(self, session_factory: SessionFactory):
        self.factory = session_factory
        self._sessions: dict[str, DataSession] = {}
        self._sessions_by_format: dict[DataFormat, list[str]] = defaultdict(list)

    async def create_session(self, source_config: SourceConfig) -> str:
        """Create new session with automatic format detection"""
        session = await self.factory.create_session(source_config)
        self._sessions[session.session_id] = session
        self._sessions_by_format[session.source.format_type].append(session.session_id)
        return session.session_id

    def get_sessions_by_format(self, format_type: DataFormat) -> list[DataSession]:
        """Get all sessions of specific format"""
        session_ids = self._sessions_by_format[format_type]
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def cross_format_operation(self,
                                    source_session_id: str,
                                    target_format: DataFormat,
                                    operation: str) -> CrossFormatResult:
        """Execute operations between different format sessions"""
        # Implementation for cross-format workflows
        pass
```

## Extension Point Design

### **Plugin Architecture for New Formats**

```python
# New: src/databeak/plugins/base.py
class DataSourcePlugin(ABC):
    """Base class for data source plugins"""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Unique format identifier"""
        pass

    @abstractmethod
    def create_source(self, config: dict) -> DataSource:
        """Create data source instance"""
        pass

    @abstractmethod
    def get_config_schema(self) -> type[BaseModel]:
        """Return Pydantic model for configuration validation"""
        pass

# Entry point registration in pyproject.toml:
[project.entry-points."databeak.sources"]
csv = "databeak.sources.csv:CSVSourcePlugin"
excel = "databeak.sources.excel:ExcelSourcePlugin"
postgresql = "databeak.plugins.postgres:PostgreSQLPlugin"  # External plugin
```

### **Format-Specific Tool Extensions**

```python
# Pattern for format-specific tools
class FormatSpecificTool(ABC):
    """Base for tools that only work with specific formats"""

    @property
    @abstractmethod
    def supported_formats(self) -> list[DataFormat]:
        pass

    def validate_session_format(self, session: DataSession) -> None:
        if session.source.format_type not in self.supported_formats:
            raise UnsupportedFormatError(
                f"Tool requires {self.supported_formats}, got {session.source.format_type}"
            )

# Example usage:
@register_mcp_tool("excel", supported_formats=[DataFormat.EXCEL])
async def merge_excel_sheets(session_id: str,
                           sheet_names: list[str]) -> MergeResult:
    """Merge multiple sheets - Excel only"""
    session = get_session(session_id)
    # Excel-specific logic here
```

This abstraction design provides a solid foundation for supporting multiple data
formats while maintaining DataBeak's current strengths in session management,
type safety, and MCP integration.
