# Regression Testing Requirements for Architecture Refactoring

## ⚠️ CRITICAL REQUIREMENT

**NO REFACTORING WORK SHALL BEGIN** without first establishing comprehensive
regression tests that verify existing functionality remains intact.

## Test-First Refactoring Protocol

### **Mandatory Pre-Refactoring Steps**

#### **1. Current State Validation**

```bash
# MUST pass before any refactoring work:
uv run pytest --cov=src/databeak --cov-fail-under=90
uv run pytest tests/ -v --tb=short
uv run pytest tests/integration/ -v
uv run all-checks
```

#### **2. Performance Baseline Establishment**

```bash
# Document current performance metrics:
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=baseline.json
# Store baseline metrics for comparison during refactoring
```

#### **3. Behavior Documentation**

```bash
# Create comprehensive behavior tests:
uv run pytest tests/regression/ -v --capture=no
# Must cover all tool interfaces and session behaviors
```

## Component-Specific Test Requirements

### **Tool Registration System (Week 1 Prep)**

#### **Required Test Coverage**

- [ ] **All 40+ MCP tools tested** with various input combinations
- [ ] **Error response consistency** across all tools
- [ ] **Session validation** in every tool
- [ ] **Parameter type checking** and edge cases
- [ ] **Async operation handling** and timeouts

#### **Regression Test Template**

```python
# tests/regression/test_tool_registration.py
class TestToolRegistrationRegression:
    """Ensure tool registration refactoring doesn't break existing tools"""

    @pytest.mark.parametrize("tool_name", ALL_CURRENT_TOOLS)
    async def test_tool_signature_unchanged(self, tool_name: str):
        """Verify all tool signatures remain identical after refactoring"""
        original_tool = get_original_tool(tool_name)
        refactored_tool = get_refactored_tool(tool_name)

        assert original_tool.__signature__ == refactored_tool.__signature__

    @pytest.mark.parametrize("tool_name", ALL_CURRENT_TOOLS)
    async def test_tool_response_format_unchanged(self, tool_name: str):
        """Verify response format consistency after decorator application"""
        # Test with valid inputs
        response = await execute_tool(tool_name, VALID_INPUTS[tool_name])
        assert response["success"] in [True, False]
        assert "session_id" in response

        # Test with invalid inputs
        response = await execute_tool(tool_name, INVALID_INPUTS[tool_name])
        assert response["success"] is False
        assert "error" in response

    async def test_error_handling_consistency(self):
        """Ensure error handling remains consistent across refactored tools"""
        # Test SessionNotFoundError
        # Test DataValidationError
        # Test FileOperationError
        # Verify consistent error response structure
```

### **Session Management (Week 2 Prep)**

#### **Required Test Coverage**

- [ ] **Session creation** with all configuration combinations
- [ ] **Data loading** from various CSV sources (files, URLs, content)
- [ ] **Session lifecycle** including TTL expiration and cleanup
- [ ] **Auto-save functionality** with all strategies and error conditions
- [ ] **History management** including undo/redo and operation recording
- [ ] **Multi-session operations** and isolation testing

#### **Session Regression Tests**

```python
# tests/regression/test_session_management.py
class TestSessionManagementRegression:
    """Comprehensive session behavior validation before refactoring"""

    async def test_csv_session_complete_lifecycle(self):
        """Validate entire session lifecycle remains unchanged"""
        # Create session
        session = CSVSession()
        assert session.session_id is not None
        assert not session.is_expired()

        # Load data
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        session.load_data(df, "test.csv")
        assert session.data_session.df is not None
        assert session.auto_save_manager.original_file_path == "test.csv"

        # Record operation
        session.record_operation(OperationType.FILTER, {"test": "data"})
        assert len(session.operations_history) > 0
        assert session.data_session.metadata["needs_autosave"] is True

        # Auto-save
        result = await session.trigger_auto_save_if_needed()
        # Verify auto-save behavior unchanged

    async def test_session_manager_behavior_unchanged(self):
        """Verify SessionManager behavior identical after refactoring"""
        manager = get_session_manager()

        # Test session creation
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        assert session is not None

        # Test cleanup
        session.lifecycle.is_expired = Mock(return_value=True)
        manager._cleanup_expired()
        assert manager.get_session(session_id) is None
```

### **Data Operations (Week 2-3 Prep)**

#### **Operations Test Matrix**

```python
# tests/regression/test_data_operations.py
class TestDataOperationsRegression:
    """Ensure all data operations produce identical results"""

    # Test data for consistent validation
    TEST_DATA = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "salary": [50000, 60000, 70000, 55000],
        "department": ["Engineering", "Sales", "Engineering", "Marketing"]
    })

    async def test_filter_operations_unchanged(self):
        """Filter operations must produce identical results"""
        session = CSVSession()
        session.load_data(self.TEST_DATA.copy(), "test.csv")

        # Test various filter conditions
        conditions = [{"column": "age", "operator": ">", "value": 28}]
        result = await filter_rows_operation(session, conditions, "AND")

        # Verify exact same filtering behavior
        expected_names = ["Bob", "Charlie"]
        assert result["data"]["rows"] == 2
        assert set(session.df["name"].tolist()) == set(expected_names)

    async def test_statistics_calculations_unchanged(self):
        """Statistical calculations must remain identical"""
        session = CSVSession()
        session.load_data(self.TEST_DATA.copy(), "test.csv")

        stats = await get_statistics_operation(session)

        # Verify exact statistical values
        assert stats["data"]["mean_age"] == 29.5
        assert stats["data"]["total_rows"] == 4
        # ... additional statistical validations
```

## Integration Test Requirements

### **End-to-End Workflow Validation**

```python
# tests/integration/test_complete_workflows.py
class TestCompleteWorkflowRegression:
    """Validate complete user workflows remain unchanged"""

    async def test_data_analysis_pipeline(self):
        """Complete analysis pipeline: load → clean → analyze → export"""
        # Load sample CSV
        session_id = await load_csv_from_content(CSV_SAMPLE_DATA)

        # Clean data
        await remove_duplicates(session_id)
        await fill_missing_values(session_id, strategy="mean", columns=["age"])

        # Analyze
        stats = await get_statistics(session_id, columns=["age", "salary"])
        correlations = await get_correlation_matrix(session_id)

        # Export
        export_result = await export_csv(session_id, format="excel", file_path="output.xlsx")

        # Verify entire pipeline works identically
        assert stats["success"] is True
        assert correlations["success"] is True
        assert export_result["success"] is True

    async def test_error_handling_pipeline(self):
        """Error conditions must produce consistent behavior"""
        # Test invalid session ID
        result = await filter_rows("invalid-session", [])
        assert result["success"] is False
        assert "error" in result

        # Test invalid data operations
        session_id = await load_csv_from_content("invalid,csv\ndata")
        result = await get_statistics(session_id, columns=["nonexistent"])
        assert result["success"] is False
```

### **Cross-Tool Integration Tests**

```python
# tests/integration/test_tool_interactions.py
class TestToolInteractionRegression:
    """Validate tool interactions remain consistent"""

    async def test_session_state_consistency(self):
        """Operations must maintain proper session state"""
        session_id = await load_csv_from_content(SAMPLE_DATA)

        # Multiple operations on same session
        await filter_rows(session_id, [{"column": "age", "operator": ">", "value": 25}])
        await sort_data(session_id, [{"column": "name", "ascending": True}])
        await add_column(session_id, "category", "adult")

        # Verify session state consistency
        session = get_session(session_id)
        assert len(session.operations_history) == 4  # load + 3 operations
        assert "category" in session.df.columns
        assert all(session.df["age"] > 25)
```

## Performance Regression Prevention

### **Benchmark Test Requirements**

```python
# tests/benchmarks/test_performance_regression.py
class TestPerformanceRegression:
    """Prevent performance regressions during refactoring"""

    @pytest.mark.benchmark(group="data_loading")
    def test_csv_loading_performance(self, benchmark):
        """CSV loading must not regress >10%"""
        def load_operation():
            return pd.read_csv(LARGE_CSV_PATH)  # 10MB test file

        result = benchmark(load_operation)
        # Benchmark framework compares with baseline automatically

    @pytest.mark.benchmark(group="tool_execution")
    async def test_tool_execution_overhead(self, benchmark):
        """Tool registration decorator must not add significant overhead"""
        session_id = await load_csv_from_content(MEDIUM_SAMPLE)

        async def filter_operation():
            return await filter_rows(session_id, STANDARD_FILTER_CONDITIONS)

        result = await benchmark(filter_operation)
        # Must remain within 10% of baseline
```

### **Memory Usage Monitoring**

```python
# tests/memory/test_memory_regression.py
import tracemalloc
import psutil

class TestMemoryRegression:
    """Monitor memory usage during refactoring"""

    async def test_session_memory_usage(self):
        """Session creation must not increase memory significantly"""
        tracemalloc.start()
        process = psutil.Process()

        baseline_memory = process.memory_info().rss

        # Create multiple sessions
        sessions = []
        for i in range(10):
            session = CSVSession()
            await session.load_data(STANDARD_TEST_DATA.copy(), f"test_{i}.csv")
            sessions.append(session)

        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - baseline_memory) / (1024 * 1024)  # MB

        # Must not exceed reasonable limits
        assert memory_increase < 100  # 10MB per session max

        tracemalloc.stop()
```

## Refactoring Validation Checklist

### **Before Starting Each Week**

- [ ] **Current tests passing**: `uv run pytest` shows all green
- [ ] **Coverage sufficient**: Target areas have >90% coverage
- [ ] **Performance baseline**: Benchmark metrics documented
- [ ] **Integration tests**: End-to-end workflows validated
- [ ] **Error conditions**: All failure modes tested

### **During Refactoring Work**

- [ ] **Continuous testing**: Run `uv run pytest --lf` after each change
- [ ] **Incremental commits**: Small, testable changes with clear commit messages
- [ ] **Feature flags**: New functionality behind configuration flags
- [ ] **Documentation updates**: Keep tests and docs in sync

### **After Completing Each Week**

- [ ] **Full test suite**: All tests pass including new ones
- [ ] **Performance validation**: No >10% regressions
- [ ] **Integration verification**: Complete workflows still functional
- [ ] **Documentation review**: All changes documented
- [ ] **Code review**: Focus on backward compatibility

## Test Infrastructure Improvements

### **Enhanced Test Utilities**

```python
# New: tests/utils/regression_helpers.py
class RegressionTestHelper:
    """Utilities for validating refactoring doesn't break functionality"""

    @staticmethod
    def compare_tool_outputs(tool_name: str,
                           original_output: dict,
                           refactored_output: dict) -> bool:
        """Compare tool outputs for identical behavior"""

    @staticmethod
    def validate_session_state(session_before: CSVSession,
                             session_after: CSVSession) -> ValidationResult:
        """Ensure session state changes are intentional"""

    @staticmethod
    async def run_complete_workflow(workflow_name: str) -> WorkflowResult:
        """Execute predefined workflows for regression testing"""
```

### **Automated Regression Detection**

```python
# New: tests/automation/regression_detector.py
class RegressionDetector:
    """Automated detection of behavior changes during refactoring"""

    def __init__(self, baseline_file: str):
        self.baseline = load_baseline_data(baseline_file)

    async def detect_regressions(self, test_suite: str) -> RegressionReport:
        """Run tests and compare with baseline behavior"""
        current_results = await run_test_suite(test_suite)

        regressions = []
        for test_name, result in current_results.items():
            baseline_result = self.baseline.get(test_name)
            if not self._results_equivalent(baseline_result, result):
                regressions.append(RegressionDetection(
                    test_name=test_name,
                    expected=baseline_result,
                    actual=result,
                    severity=self._assess_severity(baseline_result, result)
                ))

        return RegressionReport(regressions=regressions)
```

## Specific Test Requirements by Refactoring Phase

### **Phase 1: Foundation (Weeks 1-3)**

#### **Tool Registration Decorator (Week 1)**

```python
# Required before starting Week 1:
class TestToolRegistrationBaseline:

    async def test_all_tools_current_behavior(self):
        """Document current behavior of all tools before decorator change"""
        for tool_name in CURRENT_MCP_TOOLS:
            # Test valid inputs
            result = await execute_tool_with_valid_inputs(tool_name)
            store_baseline_result(tool_name, "valid", result)

            # Test invalid inputs
            result = await execute_tool_with_invalid_inputs(tool_name)
            store_baseline_result(tool_name, "invalid", result)

            # Test edge cases
            result = await execute_tool_with_edge_cases(tool_name)
            store_baseline_result(tool_name, "edge", result)

    async def test_error_response_format_consistency(self):
        """All tools must return identical error response format"""
        expected_error_format = {
            "success": False,
            "error": str,  # Error message
            "session_id": str  # Session ID if provided
        }

        for tool_name in CURRENT_MCP_TOOLS:
            error_response = await trigger_tool_error(tool_name)
            validate_response_structure(error_response, expected_error_format)
```

#### **Data Source Abstraction (Week 2)**

```python
# Required before starting Week 2:
class TestSessionAbstractionBaseline:

    async def test_csv_loading_behavior_matrix(self):
        """Test all CSV loading scenarios before abstraction"""
        test_scenarios = [
            {"source": "file_path", "encoding": "utf-8", "header": 0},
            {"source": "url", "encoding": "latin-1", "header": None},
            {"source": "content", "encoding": "utf-8", "delimiter": ";"},
            # ... comprehensive scenario matrix
        ]

        for scenario in test_scenarios:
            result = await test_csv_loading_scenario(scenario)
            store_csv_loading_baseline(scenario, result)

    async def test_session_state_management(self):
        """Validate session state changes during operations"""
        session = CSVSession()
        initial_state = capture_session_state(session)

        # Load data
        await session.load_data(TEST_DATAFRAME, "test.csv")
        after_load_state = capture_session_state(session)

        # Verify expected state changes
        assert after_load_state.df is not None
        assert after_load_state.metadata["file_path"] == "test.csv"
        # ... comprehensive state validation
```

### **Phase 2: Format Extensions (Weeks 4-7)**

#### **Excel Integration Prep (Week 4)**

```python
# Required before Excel development:
class TestExcelBaselinePreparation:

    def test_create_excel_test_data(self):
        """Create comprehensive Excel test workbooks"""
        # Single sheet workbook
        # Multi-sheet workbook with different data types
        # Workbook with complex formatting
        # Large workbook for performance testing

    async def test_current_excel_export_behavior(self):
        """Document how Excel export currently works"""
        session_id = await load_csv_from_content(SAMPLE_DATA)
        result = await export_csv(session_id, format="excel", file_path="test.xlsx")

        # Validate current Excel export behavior
        assert result["success"] is True
        workbook_info = analyze_generated_excel_file("test.xlsx")
        store_excel_export_baseline(workbook_info)
```

#### **Database Integration Prep (Week 6)**

```python
# Required before database development:
class TestDatabasePreparation:

    async def test_setup_database_test_infrastructure(self):
        """Prepare test databases with known data"""
        # SQLite test database
        # PostgreSQL test container
        # MySQL test container
        # Test data identical across all formats

    async def test_current_system_with_mock_database(self):
        """Validate how system would behave with database-like operations"""
        # Simulate loading large datasets
        # Test memory constraints
        # Validate session behavior with large data
```

## Continuous Regression Monitoring

### **Automated Regression Detection**

```bash
# Add to CI/CD pipeline:
name: Regression Detection
on: [push, pull_request]
jobs:
  regression-check:
    steps:
      - name: Run Regression Test Suite
        run: |
          uv run pytest tests/regression/ --baseline=master
          uv run pytest tests/benchmarks/ --benchmark-compare=baseline

      - name: Generate Regression Report
        run: |
          python scripts/generate_regression_report.py

      - name: Fail on Regressions
        run: |
          if [ -s regression_failures.txt ]; then
            echo "❌ REGRESSIONS DETECTED"
            cat regression_failures.txt
            exit 1
          fi
```

### **Developer Workflow Integration**

```bash
# Pre-commit hook for refactoring work:
# .pre-commit-config.yaml addition:
- repo: local
  hooks:
    - id: regression-check
      name: Run regression tests before commit
      entry: uv run pytest tests/regression/ --tb=short
      language: system
      pass_filenames: false
      stages: [commit]
```

## Documentation Requirements

### **Regression Test Documentation**

Each refactoring phase must include:

1. **Test plan document** outlining what will be tested
2. **Baseline metrics** for performance and behavior
3. **Test results** before and after refactoring
4. **Regression analysis** documenting any behavior changes
5. **Sign-off checklist** confirming no unintended regressions

---

**SUMMARY**: This regression testing strategy ensures that DataBeak's
architecture refactoring maintains 100% backward compatibility while enabling
new capabilities. No refactoring work should proceed without first establishing
comprehensive test coverage to validate that existing functionality remains
intact.
