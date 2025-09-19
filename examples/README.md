# DataBeak Examples

This directory contains example scripts demonstrating DataBeak's capabilities.

## Available Examples

### auto_save_demo.py

Demonstrates auto-save functionality with different strategies:

- Configuring auto-save settings
- Multiple save strategies (backup, versioned, overwrite)
- Testing auto-save behavior

### auto_save_overwrite.py

Shows auto-save with overwrite strategy for seamless file updates.

### history_demo.py

Comprehensive history and undo/redo demonstration:

- Operation tracking
- Undo/redo capabilities
- History export
- Session management

### dependency_injection_demo.py

Advanced example showing dependency injection patterns for testing and
modularity.

### claude_code_null_example.py

Demonstrates null value handling and data validation.

### update_consignee_example.py

Real-world example updating business data with validation.

### test_default_autosave.py

Testing example for default auto-save behavior.

## Running Examples

```bash
uv run python examples/auto_save_demo.py
uv run python examples/history_demo.py
# etc.
```

## Architecture

Examples demonstrate the current server composition architecture with
specialized FastMCP servers for different operations (I/O, transformation,
statistics, etc.).
