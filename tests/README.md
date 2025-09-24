# DataBeak Testing Guide

## Overview

DataBeak currently focuses on comprehensive unit testing with plans for future
integration and E2E testing expansion. Our tests are organized by scope and
purpose, making it easy to run the appropriate tests during development.

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
├── servers/        # MCP server modules (FastMCP composition)
├── services/       # Business logic services
├── utils/          # Utility functions
└── security/       # Security and validation tests
```

**Characteristics:**

- Fast execution (< 100ms per test)
- Mocked dependencies
- No external I/O (files, network, database)
- Test single functions or classes
- High code coverage target (80%+)

**Run with:** `uv run pytest -n auto tests/unit/`

### Integration Tests (`tests/integration/`) - Future Implementation

Integration tests will verify that multiple components work correctly together.
Planned implementation using FastMCP Client for realistic MCP protocol testing:

- MCP protocol integration testing
- Component interaction validation
- Session management across components
- Data flow verification between modules

**Future Characteristics:**

- Moderate execution time (100ms - 1s per test)
- Real components with minimal mocking
- FastMCP Client-based MCP protocol testing
- Validate data flow between modules

**Planned:** Implementation tracked in GitHub issues

### End-to-End Tests (`tests/e2e/`) - Future Implementation

E2E tests will validate complete user workflows from start to finish:

- Full MCP server workflow validation
- Complete CSV data processing pipelines
- End-user scenario testing

**Future Characteristics:**

- Slower execution (> 1s per test)
- No mocking - real system behavior
- Test complete user scenarios
- Validate edge cases and error handling

**Planned:** Implementation tracked in GitHub issues

## Running Tests

### Quick Commands

```bash
# Run all tests
uv run pytest -n auto

# Run with coverage
uv run pytest -n auto --cov=src/databeak --cov-report=term-missing

# Run specific test category
uv run pytest -n auto tests/unit/          # Current primary testing
# Future: uv run pytest -n auto tests/integration/
# Future: uv run pytest -n auto tests/e2e/

# Run specific module tests
uv run pytest -n auto tests/unit/servers/
uv run pytest -n auto tests/unit/models/

# Run with verbose output
uv run pytest -n auto -v

# Run and stop on first failure
uv run pytest -n auto -x

# Run specific test (single test, no parallel)
uv run pytest tests/unit/servers/test_statistics_server.py::TestGetStatistics::test_get_statistics_all_columns
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
uv run pytest -n auto
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


@patch("src.databeak.models.csv_session.get_session_manager")
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
uv run pytest -n auto --cov=src/databeak --cov-report=html

# View HTML report
open htmlcov/index.html

# Terminal report with missing lines
uv run pytest -n auto --cov=src/databeak --cov-report=term-missing
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

CI pipeline currently runs:

1. Unit tests (fail fast)
1. Coverage report
1. Quality checks (ruff, mypy)

Future CI additions:

- Integration tests (FastMCP Client-based)
- E2E tests (complete workflow validation)

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
import uuid


@pytest.fixture
async def session():
    session_id = str(uuid.uuid4())
    session = session_manager.get_or_create_session(session_id)
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

1. **Plan integration tests** for `tests/integration/` (future)

   - Consider how the feature will interact with existing components
   - Document integration test requirements for future implementation

1. **Plan E2E tests** for `tests/e2e/` (future)

   - Only for major features or workflows
   - Document complete user experience test scenarios

Example for a new feature:

```bash
# 1. Create unit test file
touch tests/unit/servers/test_new_feature.py

# 2. Write tests
# 3. Run tests
uv run -m pytest tests/unit/servers/test_new_feature.py

# 4. Check coverage
uv run pytest tests/unit/servers/test_new_feature.py --cov=src/databeak/servers/new_feature
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Python Mock](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
