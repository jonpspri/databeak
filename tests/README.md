# DataBeak Testing Guide

## Overview

DataBeak follows a three-tier testing strategy to ensure code quality,
reliability, and maintainability. Our tests are organized by scope and purpose,
making it easy to run the appropriate tests during development.

## Test Organization

```
tests/
├── unit/           # Fast, isolated module tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end workflow tests
└── conftest.py    # Shared fixtures and configuration
```

### Unit Tests (`tests/unit/`)

Unit tests mirror the source code structure and focus on testing individual
modules in isolation:

```
tests/unit/
├── models/         # Data models and session management
├── servers/        # MCP server modules
├── tools/          # Individual tool operations
├── utils/          # Utility functions
└── resources/      # Resource handling
```

**Characteristics:**

- Fast execution (< 100ms per test)
- Mocked dependencies
- No external I/O (files, network, database)
- Test single functions or classes
- High code coverage target (80%+)

**Run with:** `uv run -m pytest tests/unit/`

### Integration Tests (`tests/integration/`)

Integration tests verify that multiple components work correctly together:

- `test_ai_accessibility.py` - AI assistant workflow tests
- `test_analytics_coverage.py` - Analytics pipeline integration
- `test_mcp_*.py` - MCP protocol integration
- `test_session_coverage.py` - Session management across components

**Characteristics:**

- Moderate execution time (100ms - 1s per test)
- Real components with minimal mocking
- Test component interactions
- Validate data flow between modules

**Run with:** `uv run -m pytest tests/integration/`

### End-to-End Tests (`tests/e2e/`)

E2E tests validate complete user workflows from start to finish:

- `test_io_server_comprehensive_coverage.py` - Full I/O pipeline validation

**Characteristics:**

- Slower execution (> 1s per test)
- No mocking - real system behavior
- Test complete user scenarios
- Validate edge cases and error handling

**Run with:** `uv run -m pytest tests/e2e/`

## Running Tests

### Quick Commands

```bash
# Run all tests
uv run -m pytest

# Run with coverage
uv run -m pytest --cov=src/databeak --cov-report=term-missing

# Run specific test category
uv run -m pytest tests/unit/
uv run -m pytest tests/integration/
uv run -m pytest tests/e2e/

# Run specific module tests
uv run -m pytest tests/unit/servers/
uv run -m pytest tests/unit/models/

# Run with verbose output
uv run -m pytest -v

# Run and stop on first failure
uv run -m pytest -x

# Run specific test
uv run -m pytest tests/unit/servers/test_statistics_server.py::TestGetStatistics::test_get_statistics_all_columns
```

### Test Discovery

Pytest automatically discovers tests following these patterns:

- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*` (no `__init__` method)
- Test functions: `test_*`
- Test methods: `test_*`

### Parallel Execution

For faster test runs on multi-core systems:

```bash
# Install pytest-xdist
uv add --dev pytest-xdist

# Run tests in parallel
uv run -m pytest -n auto
```

## Writing Tests

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
@pytest.mark.asyncio
async def test_function_behavior():
    # Arrange - Set up test data and mocks
    session_id = "test-session"
    test_data = create_test_data()

    # Act - Execute the function being tested
    result = await function_under_test(session_id, test_data)

    # Assert - Verify the results
    assert result.success is True
    assert result.data == expected_data
```

### Fixtures

Common fixtures are defined in `conftest.py` and test-specific fixtures in each
test file:

```python
@pytest.fixture
async def csv_session():
    """Create a test session with sample CSV data."""
    csv_content = "name,age\nAlice,30\nBob,25"
    result = await load_csv_from_content(csv_content)
    yield result.session_id
    # Cleanup
    manager = get_session_manager()
    await manager.remove_session(result.session_id)
```

### Mocking

Use `unittest.mock` for isolation in unit tests:

```python
from unittest.mock import patch, MagicMock

@patch('src.databeak.models.csv_session.get_session_manager')
def test_with_mock(mock_manager):
    mock_session = MagicMock()
    mock_manager.return_value.get_session.return_value = mock_session
    # Test code here
```

### Async Tests

Use `pytest-asyncio` for async function testing:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## Test Coverage

### Coverage Goals

- **Overall**: 80% minimum coverage requirement
- **Success Rate**: 100% test pass rate in CI/CD pipeline
- **Unit Tests**: Should cover all public APIs
- **Integration Tests**: Should cover critical paths
- **E2E Tests**: Should cover user workflows
- **Skipped Tests**: Must be annotated with GitHub Issue # for implementation

### Viewing Coverage

```bash
# Generate coverage report
uv run -m pytest --cov=src/databeak --cov-report=html

# View HTML report
open htmlcov/index.html

# Terminal report with missing lines
uv run -m pytest --cov=src/databeak --cov-report=term-missing
```

### Coverage Configuration

Coverage settings in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/databeak"]
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

## Continuous Integration

Tests run automatically on:

- Pull requests
- Pushes to main branch
- Nightly scheduled runs

CI pipeline runs:

1. Unit tests (fail fast)
1. Integration tests
1. E2E tests
1. Coverage report
1. Quality checks (ruff, mypy)

## Best Practices

### DO

- Write tests before fixing bugs (regression tests)
- Keep tests focused and independent
- Use descriptive test names that explain what's being tested
- Mock external dependencies in unit tests
- Test both success and failure cases
- Use fixtures for common setup
- Run tests locally before pushing

### DON'T

- Write tests that depend on test execution order
- Use production data in tests
- Leave commented-out test code
- Ignore flaky tests - fix them
- Test implementation details - test behavior
- Use `time.sleep()` - use proper async waiting

## Troubleshooting

### Common Issues

**Import Errors:**

```bash
# Ensure databeak is installed in development mode
uv sync
```

**Async Test Failures:**

```python
# Mark async tests properly
@pytest.mark.asyncio
async def test_async():
    pass
```

**Session Cleanup:**

```python
# Always clean up sessions in fixtures
@pytest.fixture
async def session():
    import uuid
    session_id = str(uuid.uuid4())
    session = session_manager.get_session(session_id)
    yield session_id
    await cleanup_session(session_id)  # This runs after test
```

**Flaky Tests:**

- Check for race conditions
- Ensure proper mocking
- Use deterministic test data
- Avoid time-dependent assertions

## Adding New Tests

When adding new features:

1. **Write unit tests first** in `tests/unit/`

   - Test the new functions/classes in isolation
   - Mock all dependencies

1. **Add integration tests** in `tests/integration/`

   - Test how the feature interacts with existing components
   - Use real components where possible

1. **Consider E2E tests** in `tests/e2e/`

   - Only for major features or workflows
   - Test the complete user experience

Example for a new feature:

```bash
# 1. Create unit test file
touch tests/unit/tools/test_new_feature.py

# 2. Write tests
# 3. Run tests
uv run -m pytest tests/unit/tools/test_new_feature.py

# 4. Check coverage
uv run -m pytest tests/unit/tools/test_new_feature.py --cov=src/databeak/tools/new_feature
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Python Mock](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
