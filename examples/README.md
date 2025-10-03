# DataBeak Examples

This directory contains example scripts demonstrating DataBeak's capabilities.

## Available Examples

### dependency_injection_demo.py

Advanced example showing dependency injection patterns for testing and
modularity. Demonstrates how to structure DataBeak integrations with proper
separation of concerns and testable architecture.

### claude_code_null_example.py

Demonstrates null value handling and data validation. Shows best practices for
handling missing data, validation errors, and defensive programming patterns
when working with CSV data.

### update_consignee_example.py

Real-world example updating business data with validation. Illustrates a
complete workflow for loading, transforming, validating, and exporting modified
CSV data in a business context.

## Running Examples

```bash
uv run python examples/dependency_injection_demo.py
uv run python examples/claude_code_null_example.py
uv run python examples/update_consignee_example.py
```

## Architecture

Examples demonstrate the current server composition architecture with
specialized FastMCP servers for different operations (I/O, transformation,
statistics, etc.).
