# Architecture Refactoring Quick Reference

## 📋 Executive Summary

**Goal**: Transform DataBeak from CSV-focused tool to multi-format data platform
**Timeline**: 10-12 weeks in 3 phases
**Approach**: Incremental refactoring with mandatory regression testing

## 🏗️ Current Architecture (Issues)

```
❌ Current Problems:
├── CSVSession (415 lines) - Too many responsibilities
├── Tool Registration (~200 lines boilerplate) - Repetitive patterns
├── Pandas Coupling - Hard dependencies throughout
├── CSV-Only Input - Limited to CSV files for data loading
└── Manual Configuration - Scattered settings management

✅ Current Strengths:
├── Session Management - Lifecycle, auto-save, history tracking
├── Type Safety - Comprehensive Pydantic models
├── MCP Integration - Clean separation of concerns
├── Error Handling - Structured exception hierarchy
└── Modular Tools - 40+ tools across 7 functional areas
```

## 🎯 Target Architecture (Solutions)

```
✅ Proposed Solutions:
├── DataSource Abstraction - Protocol-based multi-format support
├── Tool Decorator System - Auto-registration with @register_mcp_tool
├── Session Hierarchy - Format-specific sessions with shared base
├── Plugin Architecture - Extensible format support system
└── Unified Configuration - pyproject.toml-based settings

🚀 New Capabilities:
├── Excel/XLSX - First-class multi-sheet workbook support
├── Databases - SQLite, PostgreSQL, MySQL connectivity
├── Streaming - Process 10GB+ files with <2GB memory
├── Cross-Format - Transfer data between session types
└── Auto-Discovery - Plugin system for third-party formats
```

## ⚡ Implementation Phases

### **Phase 1: Foundation (Weeks 1-3)**

```
Week 1: Data Source Abstraction (CRITICAL PATH)
├── DataSource protocol design
├── CSVDataSource refactoring
├── Session management updates
└── Plugin discovery mechanism

Week 2: Tool Registration Automation
├── @register_mcp_tool decorator
├── Auto-discovery system
├── Boilerplate elimination
└── Error handling standardization

Week 3: Testing & Validation
├── Comprehensive regression tests
├── Performance baseline validation
├── Integration test updates
└── Documentation refresh
```

### **Phase 2: Format Extensions (Weeks 4-7)**

```
Week 4-5: Excel Integration
├── ExcelDataSource implementation
├── Multi-sheet support (list, switch, merge)
├── Excel-specific session type
└── Workbook metadata extraction

Week 6-7: Database Connectivity
├── SQLite foundation + PostgreSQL/MySQL
├── Connection management with pooling
├── SQL query execution with safety validation
└── Schema introspection and table operations
```

### **Phase 3: Advanced Features (Weeks 8-10)**

```
Week 8: Streaming & Performance
├── Chunked processing for large datasets
├── Memory optimization improvements
├── Progress tracking for long operations
└── Async I/O enhancements

Week 9: Cross-Format Workflows
├── Data transfer between session types
├── Multi-source analysis pipelines
├── Format conversion utilities
└── Workflow orchestration tools

Week 10: Integration & Polish
├── Configuration management enhancements
├── Performance monitoring and diagnostics
├── Error handling improvements
└── Final documentation and examples
```

## 🔒 Mandatory Testing Requirements

### **⚠️ BEFORE ANY REFACTORING**

```bash
# Required before starting each week:
uv run pytest --cov=src/databeak --cov-fail-under=90
uv run pytest tests/integration/ -v
uv run pytest tests/benchmarks/ --benchmark-only

# Create baseline measurements:
Performance baselines → Store in baseline.json
Memory usage limits → Document current usage patterns
Test coverage report → Identify coverage gaps
```

### **Test Coverage Requirements**

- **>90% coverage** for areas being refactored
- **Integration tests** for end-to-end workflows
- **Performance baselines** with \<10% regression tolerance
- **Error condition tests** for all failure modes
- **Backward compatibility** validation for existing tools

## 🛡️ Safety & Risk Mitigation

### **Rollback Strategy**

```
Safety Measures:
├── Feature Flags - Enable/disable new formats individually
├── Parallel Testing - Run old and new systems during transition
├── Git History - All original code preserved in commits
├── Performance Monitoring - Detect regressions immediately
└── Gradual Rollout - Phased deployment with user feedback
```

### **High-Risk Areas**

- **Session Management**: Core to all operations (use adapter patterns)
- **Tool Interfaces**: Breaking changes affect all MCP tools
- **Database Security**: SQL injection, unauthorized access risks
- **Performance**: Abstraction overhead could slow operations

## 💼 Business Value

### **Immediate Wins (Phase 1)**

- **30% code reduction** through boilerplate elimination
- **Simplified maintenance** with standardized patterns
- **Foundation for growth** enabling format extensions
- **Improved testability** with better abstractions

### **Format Expansion (Phase 2)**

- **Excel support** addresses major user requests
- **Database connectivity** enables enterprise adoption
- **Competitive advantage** over CSV-only tools
- **Expanded user base** through multi-format capabilities

### **Scalability (Phase 3)**

- **10GB+ file support** through streaming processing
- **Enterprise performance** with connection pooling
- **Advanced workflows** for power users
- **Production readiness** with monitoring and diagnostics

## 🔧 Technical Implementation

### **Key Abstractions**

```python
# Core abstraction pattern:
class DataSource(Protocol):
    async def load(self, config: SourceConfig) -> DataFrame
    async def save(self, df: DataFrame, config: TargetConfig) -> SaveResult
    def get_capabilities(self) -> FormatCapabilities

# Format-specific implementations:
CSVDataSource    - Enhanced with streaming support
ExcelDataSource  - Multi-sheet workbook operations
DatabaseSource   - SQL execution with connection pooling

# Session hierarchy:
DataSession (base) → CSVSession, ExcelSession, DatabaseSession
```

### **Tool Registration Automation**

```python
# Replace manual boilerplate:
@mcp.tool
async def filter_rows(ctx, session_id, conditions, logic) -> dict:
    # 20+ lines of boilerplate per tool

# With auto-registration:
@register_mcp_tool("data", "Filter rows based on conditions")
async def filter_rows(session_id: str, conditions: list[FilterCondition]) -> FilterResult:
    # Pure business logic only
```

## 📊 Success Metrics

### **Technical Metrics**

- [ ] **Multi-format support**: CSV, Excel, SQLite, PostgreSQL
- [ ] **Code reduction**: 30% less boilerplate code
- [ ] **Performance**: Maintain speed, add streaming for >1GB files
- [ ] **Memory efficiency**: Process 10x larger files with same memory
- [ ] **Test coverage**: >90% across all implementations

### **User Experience Metrics**

- [ ] **Zero breaking changes**: All existing workflows unchanged
- [ ] **Enhanced capabilities**: Excel and database tools available
- [ ] **Clear error messages**: Categorized, actionable feedback
- [ ] **Seamless adoption**: New formats work with existing tools

### **Development Metrics**

- [ ] **Reduced maintenance**: Fewer files to modify for changes
- [ ] **Faster development**: New tools easier to create
- [ ] **Better testing**: Isolated components with dependency injection
- [ ] **Enhanced modularity**: Lower coupling between components

## 🚀 Getting Started

### **Pre-Implementation Checklist**

1. **Get stakeholder approval** for refactoring approach
1. **Create feature branch** for Phase 1 work
1. **Set up regression test suite** (>90% coverage requirement)
1. **Establish performance baselines** for regression detection
1. **Configure feature flags** for gradual rollout

### **Week 1 Kickoff**

```bash
# Start with data source abstraction (critical path):
git checkout -b refactor/data-source-abstraction

# Establish test coverage:
uv run pytest --cov=src/databeak/models/ --cov-fail-under=90

# Begin implementation:
mkdir src/databeak/sources/
touch src/databeak/sources/{base.py,types.py,csv.py}
```

______________________________________________________________________

**This quick reference provides immediate access to key information from the
comprehensive planning documents, enabling developers to quickly understand
the refactoring strategy and implementation approach.**
