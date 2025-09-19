# Testing Best Practices

## Overview

This document outlines testing best practices for DataBeak development. Our goal
is to maintain high code quality, catch bugs early, and ensure reliable
functionality across all components.

## Testing Philosophy

### Current Testing Approach

DataBeak currently focuses on comprehensive unit testing with future plans for
integration and E2E testing:

```
        /\
       /E2E\      <- Future: Complete workflow validation
      /------\
     /Integr. \   <- Future: FastMCP Client-based testing
    /----------\
   /   Unit     \ <- Current focus: 1100+ comprehensive tests
  /--------------\
```

- **Unit Tests (Current)**: 1100+ fast, isolated module tests with high coverage
- **Integration Tests (Planned)**: FastMCP Client-based realistic protocol
  testing
- **E2E Tests (Planned)**: Complete workflow validation

### Key Principles

1. **Test Behavior, Not Implementation**

   - Focus on what the code does, not how it does it
   - Tests should survive refactoring

1. **Fast Feedback**

   - Unit tests should run in milliseconds
   - Developers should run tests frequently

1. **Isolation**

   - Tests should not depend on each other
   - Use mocking to isolate components

1. **Clarity**

   - Test names should describe what is being tested
   - Failures should clearly indicate what went wrong

## Test Organization

### Directory Structure

```
tests/
├── unit/              # Mirrors src/ structure
│   ├── models/
│   ├── prompts/
│   ├── resources/
│   ├── servers/
│   ├── services/
│   └── utils/
├── integration/       # Cross-component tests
└── e2e/              # Full workflow tests
```

### Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<what_is_being_tested>`
- Fixtures: `<resource>_fixture` or descriptive name

### Example Test Structure

```python
"""Unit tests for statistics_server module."""

import pytest
from unittest.mock import Mock, patch
from src.databeak.servers.statistics_server import get_statistics


class TestGetStatistics:
    """Tests for get_statistics function."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock session with test data."""
        session = Mock()
        session.data = pd.DataFrame({"col1": [1, 2, 3]})
        return session

    def test_get_statistics_success(self, mock_session):
        """Test successful statistics calculation."""
        # Arrange
        session_id = "test-123"

        # Act
        with patch('get_session_manager') as mock_manager:
            mock_manager.return_value.get_session.return_value = mock_session
            result = get_statistics(session_id)

        # Assert
        assert result.success is True
        assert "col1" in result.statistics

    def test_get_statistics_invalid_session(self):
        """Test statistics with invalid session."""
        with pytest.raises(ToolError, match="Session not found"):
            get_statistics("invalid-id")
```

## Writing Effective Tests

### Unit Tests

**Purpose**: Test individual functions/methods in isolation

**Best Practices:**

- Mock all external dependencies
- Test one thing per test
- Use descriptive test names
- Keep tests under 20 lines
- Test edge cases and error conditions

**Example:**

```python
@pytest.mark.asyncio
async def test_add_column_with_default_value():
    """Test adding a column with a default value."""
    # Arrange
    df = pd.DataFrame({"A": [1, 2, 3]})

    # Act
    result = add_column(df, "B", default_value=0)

    # Assert
    assert "B" in result.columns
    assert all(result["B"] == 0)
```

### Integration Tests

**Purpose**: Test interactions between components

**Best Practices:**

- Use real components where possible
- Mock external services (APIs, databases)
- Test data flow through multiple components
- Verify component contracts

**Example:**

```python
@pytest.mark.asyncio
async def test_load_transform_export_workflow():
    """Test complete data processing workflow."""
    # Load CSV
    load_result = await load_csv_from_content("col1,col2\n1,2\n3,4")
    session_id = load_result.session_id

    # Transform data
    filter_result = await filter_rows(session_id, [{"column": "col1", "operator": ">", "value": 1}])
    assert filter_result.success

    # Export results
    export_result = await export_csv(session_id)
    assert "col1,col2\n3,4" in export_result.content
```

### End-to-End Tests

**Purpose**: Validate complete user scenarios

**Best Practices:**

- Test from the user's perspective
- Cover critical user journeys
- Include error scenarios
- Use realistic data
- Accept slower execution

**Example:**

```python
@pytest.mark.asyncio
async def test_data_analysis_workflow():
    """Test complete data analysis workflow as a user would."""
    # User loads a CSV file
    result = await load_csv("sales_data.csv")
    session_id = result.session_id

    # User cleans data
    await remove_duplicates(session_id)
    await fill_missing_values(session_id, strategy="mean")

    # User performs analysis
    stats = await get_statistics(session_id)
    outliers = await detect_outliers(session_id, method="iqr")

    # User exports results
    report = await export_analysis_report(session_id)
    assert report.total_rows > 0
    assert report.outliers_detected == len(outliers)
```

## Testing Patterns

### Fixtures

Use fixtures for common test setup:

```python
@pytest.fixture
async def sample_session():
    """Create a session with sample data."""
    csv_content = """
    product,price,quantity
    Apple,1.50,100
    Banana,0.75,150
    Orange,1.25,80
    """
    result = await load_csv_from_content(csv_content)
    yield result.session_id
    # Cleanup
    await cleanup_session(result.session_id)
```

### Parametrized Tests

Test multiple scenarios with one test:

```python
@pytest.mark.parametrize("operator,value,expected_count", [
    (">", 1.00, 2),   # Apple and Orange
    ("<=", 1.00, 1),  # Banana
    ("==", 1.50, 1),  # Apple
])
async def test_filter_by_price(sample_session, operator, value, expected_count):
    """Test filtering with different operators."""
    result = await filter_rows(
        sample_session,
        [{"column": "price", "operator": operator, "value": value}]
    )
    assert result.rows_after_filter == expected_count
```

### Mocking

Mock external dependencies in unit tests:

```python
@patch('src.databeak.utils.validators.requests.get')
def test_url_validation(mock_get):
    """Test URL validation with mocked HTTP request."""
    mock_get.return_value.status_code = 200

    result = validate_url("https://example.com/data.csv")
    assert result is True
    mock_get.assert_called_once_with("https://example.com/data.csv")
```

### Async Testing

Test async functions properly:

```python
@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent async operations."""
    tasks = [
        load_csv_from_content(f"col\n{i}")
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(r.success for r in results)
```

## Test Data Management

### Test Data Guidelines

1. **Use minimal data**: Only include data necessary for the test
1. **Be explicit**: Define test data in the test or nearby fixture
1. **Avoid external files**: Embed small datasets in tests
1. **Use factories**: Create data generation functions for complex data

### Data Factories

```python
def create_test_dataframe(rows=10, columns=None):
    """Create a test DataFrame with specified dimensions."""
    if columns is None:
        columns = ["col1", "col2", "col3"]

    data = {
        col: np.random.randn(rows)
        for col in columns
    }
    return pd.DataFrame(data)

def create_csv_content(rows=5):
    """Generate CSV content for testing."""
    lines = ["name,value"]
    for i in range(rows):
        lines.append(f"Item{i},{i*10}")
    return "\n".join(lines)
```

## Performance Testing

### Benchmarking

Use pytest-benchmark for performance tests:

```python
def test_large_dataset_performance(benchmark):
    """Test performance with large dataset."""
    df = create_test_dataframe(rows=100000)

    result = benchmark(process_dataframe, df)
    assert result is not None
    assert benchmark.stats["mean"] < 1.0  # Should complete in < 1 second
```

### Load Testing

Test system limits:

```python
@pytest.mark.slow
async def test_concurrent_session_limit():
    """Test system handles maximum concurrent sessions."""
    sessions = []
    for i in range(MAX_SESSIONS):
        session_id = f"session_{i}"
        session = session_manager.get_or_create_session(session_id)
        sessions.append(session)

    # Verify all sessions are active
    for session in sessions:
        assert await is_session_active(session)

    # Cleanup
    for session in sessions:
        await cleanup_session(session)
```

## Error Testing

### Exception Testing

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid column name"):
        process_column(None)

@pytest.mark.asyncio
async def test_timeout_handling():
    """Test operation timeout handling."""
    with pytest.raises(TimeoutError):
        await long_running_operation(timeout=0.001)
```

### Edge Cases

Always test edge cases:

```python
class TestEdgeCases:
    """Test edge cases for data operations."""

    async def test_empty_dataframe(self):
        """Test operations on empty DataFrame."""
        result = await process_empty_data()
        assert result.rows == 0

    async def test_single_row(self):
        """Test with single row of data."""
        result = await process_single_row()
        assert result.statistics is not None

    async def test_null_values(self):
        """Test handling of null values."""
        result = await process_with_nulls()
        assert result.null_count > 0

    async def test_maximum_size(self):
        """Test with maximum allowed size."""
        result = await process_max_size_data()
        assert result.success is True
```

## Continuous Integration

### CI Test Strategy

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: |
          uv sync
          uv run -m pytest tests/unit/ --fail-fast

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: |
          uv sync
          uv run -m pytest tests/integration/

  e2e-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run E2E tests
        run: |
          uv sync
          uv run -m pytest tests/e2e/
```

## Debugging Tests

### Useful Pytest Options

```bash
# Show print statements
uv run -m pytest -s

# Show local variables on failure
uv run -m pytest -l

# Drop into debugger on failure
uv run -m pytest --pdb

# Run only failed tests from last run
uv run -m pytest --lf

# Run tests that match expression
uv run -m pytest -k "statistics and not slow"

# Show slowest tests
uv run -m pytest --durations=10
```

### Using Debugger in Tests

```python
def test_complex_logic():
    """Test with debugger breakpoint."""
    data = create_complex_data()

    # Set breakpoint for debugging
    import pdb; pdb.set_trace()

    result = process_data(data)
    assert result.success
```

## Test Maintenance

### Keeping Tests Healthy

1. **Regular Review**: Review and update tests during refactoring
1. **Remove Redundancy**: Eliminate duplicate tests
1. **Fix Flaky Tests**: Don't ignore intermittent failures
1. **Update Documentation**: Keep test docs in sync with code
1. **Monitor Coverage**: Track coverage trends over time

### Test Refactoring

When refactoring tests:

```python
# Before: Multiple similar tests
def test_filter_greater_than():
    result = filter_data(">", 5)
    assert result.count == 3

def test_filter_less_than():
    result = filter_data("<", 5)
    assert result.count == 2

# After: Parametrized test
@pytest.mark.parametrize("operator,value,expected", [
    (">", 5, 3),
    ("<", 5, 2),
])
def test_filter_operations(operator, value, expected):
    result = filter_data(operator, value)
    assert result.count == expected
```

## Checklist for New Tests

- [ ] Test follows naming conventions
- [ ] Test has clear, descriptive name
- [ ] Test is in correct directory (unit/integration/e2e)
- [ ] Test is isolated and doesn't depend on other tests
- [ ] Test includes both success and failure cases
- [ ] Test has appropriate assertions
- [ ] Test cleans up resources (sessions, files, etc.)
- [ ] Test runs quickly (< 100ms for unit tests)
- [ ] Test is documented if complex
- [ ] Test passes locally before commit

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Test Driven Development](https://testdriven.io/)
- [Mocking in Python](https://realpython.com/python-mock-library/)
- [Async Testing with Pytest](https://pytest-asyncio.readthedocs.io/)
