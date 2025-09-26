# Integration Tests

This directory contains integration tests for DataBeak that test the full MCP
server functionality by running a real server process and connecting to it via
the FastMCP client.

## Key Components

### `conftest.py`

Provides the main fixtures for integration testing:

- `DatabeakServerFixture`: A context manager class that handles server lifecycle
- `databeak_server()`: Async context manager function for direct use
- `server_fixture`: Pytest fixture for use with test functions

### Server Fixture Features

- **Process Management**: Starts `uv run databeak` as a subprocess with HTTP
  transport
- **Health Checking**: Waits for server to be ready before proceeding
- **Client Integration**: Provides a connected FastMCP Client instance
- **Resource Cleanup**: Automatically terminates server and cleans up resources
- **Multiple Tool Calls**: Supports multiple tool invocations within the same
  test

## Usage Patterns

### Using the Pytest Fixture

```python
@pytest.mark.asyncio
async def test_something(server_fixture):
    tools = await server_fixture.list_tools()
    result = await server_fixture.call_tool("get_session_info")
    assert len(tools) > 0
```

### Using the Context Manager Directly

```python
@pytest.mark.asyncio
async def test_something():
    async with databeak_server() as server:
        tools = await server.list_tools()
        result = await server.call_tool("load_csv", file_path="/path/to/file.csv")
        # Multiple tool calls supported
        info = await server.call_tool("get_session_info")
```

### Multiple Tool Calls

The fixture supports multiple tool calls within the same test function:

```python
async def test_workflow(server_fixture):
    # Call 1: Check initial state (no data loaded)
    initial_info = await server_fixture.call_tool("get_session_info")

    # Call 2: Load data
    result = await server_fixture.call_tool("load_csv", file_path="test.csv")

    # Call 3: Verify data was loaded
    loaded_info = await server_fixture.call_tool("get_session_info")

    # Session should now have data
    assert result.isError is False
    assert loaded_info != initial_info  # Session state changed
```

## Configuration

### Port Management

Tests use different ports to avoid conflicts:

- Default: `8000`
- Custom ports can be specified: `databeak_server(port=8001)`
- Server automatically finds available ports if specified port is busy

### Server Options

The fixture starts the server with:

- Transport: HTTP (for easier testing than stdio)
- Host: `127.0.0.1` (localhost only)
- Log Level: `INFO`

## Running Integration Tests

```bash
# Run all integration tests
uv run pytest tests/integration/

# Run specific integration test
uv run pytest tests/integration/test_server_integration.py

# Run with verbose output
uv run pytest -v tests/integration/

# Run integration tests with coverage
uv run pytest --cov=src/databeak tests/integration/
```

## Available Tools for Testing

The integration tests can work with these MCP tools:

- **`load_csv`**: Load CSV files from filesystem paths
- **`load_csv_from_url`**: Load CSV files from HTTP URLs
- **`load_csv_from_content`**: Load CSV data from string content
- **`export_csv`**: Export session data to various formats (CSV, JSON, Excel, etc.)
- **`get_session_info`**: Get information about the current session (data loaded, row/column counts, etc.)

Note: Session lifecycle management (creation, listing, closing) is handled by the MCP server infrastructure, not by individual tools.

## Test Organization

Integration tests should focus on:

- **End-to-end workflows**: Complete user scenarios (load → process → export)
- **Tool integration**: How CSV loading, processing, and export tools work together
- **Data persistence**: Session state management across tool calls
- **Error handling**: Server behavior with invalid files, network issues, etc.
- **Performance**: Response times and resource usage for large datasets

## Comparison with Unit Tests

| Aspect        | Unit Tests             | Integration Tests     |
| ------------- | ---------------------- | --------------------- |
| **Scope**     | Individual functions   | Full server workflows |
| **Speed**     | Fast (ms)              | Slower (seconds)      |
| **Isolation** | Mocked dependencies    | Real server process   |
| **Purpose**   | Code correctness       | Feature functionality |
| **Debugging** | Easy to isolate issues | Complex interactions  |

Use integration tests to verify that the complete system works as expected from
a user's perspective, while unit tests verify individual component correctness.
