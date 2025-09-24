---
sidebar_position: 6
title: Quality Standards
---

## Overview

DataBeak maintains strict code quality standards with comprehensive automated
enforcement. This document outlines our quality metrics, tools, and processes.

## Quality Metrics

### Current Status (All Automated)

- ✅ **Zero ruff violations** - Perfect linting compliance across 46 rules
- ✅ **100% MyPy compliance** - Complete type safety with minimal Any usage
- ✅ **Perfect MCP documentation** - All tools have Field descriptions, zero Args
  sections
- ✅ **High test coverage** - 960+ unit tests validating all functionality
- ✅ **Clean architecture** - Stateless MCP design, eliminated 3,340+ lines of
  complexity

### Quality Enforcement Tools

#### Core Quality Tools

| Tool           | Purpose              | Configuration                          |
| -------------- | -------------------- | -------------------------------------- |
| **Ruff**       | Linting & formatting | 46 rules enabled, 100-char line length |
| **MyPy**       | Static type checking | Strict mode, pandas-stubs integration  |
| **Pytest**     | Unit testing         | 960+ tests, 80% coverage minimum       |
| **Pre-commit** | Quality gates        | Automated enforcement on commits       |

#### MCP Documentation Tools

| Tool                                | Purpose                       | Usage                                     |
| ----------------------------------- | ----------------------------- | ----------------------------------------- |
| **check_docstring_args.py**         | No Args sections in MCP tools | `scripts/check_docstring_args.py`         |
| **check_mcp_field_descriptions.py** | Complete Field descriptions   | `scripts/check_mcp_field_descriptions.py` |

## Quality Commands

### Comprehensive Quality Check

```bash
# Run all quality checks (recommended)
uv run pre-commit run --all-files

# Individual quality checks
uv run ruff check src/ tests/           # Linting
uv run ruff format --check src/ tests/  # Format verification
uv run --directory src mypy .                        # Type checking
uv run pytest tests/unit/ --cov=src     # Testing with coverage
scripts/check_docstring_args.py         # MCP Args compliance
scripts/check_mcp_field_descriptions.py # MCP Field compliance
```

### Auto-fix Commands

```bash
# Auto-fix formatting and linting issues
uv run ruff check --fix src/ tests/     # Fix linting violations
uv run ruff format src/ tests/          # Apply consistent formatting

# Fix pre-commit issues
uv run pre-commit run ruff --all-files
uv run pre-commit run ruff-format --all-files
```

### MCP Documentation Compliance

```bash
# Check MCP documentation standards
scripts/check_docstring_args.py --quiet        # Summary: Args sections check
scripts/check_mcp_field_descriptions.py --quiet # Summary: Field descriptions check

# Check specific files/directories
scripts/check_docstring_args.py src/databeak/servers/
scripts/check_mcp_field_descriptions.py src/databeak/servers/validation_server.py
```

## Quality Standards Details

### Code Quality Requirements

#### Linting (Ruff)

- **Zero violations** across all 46 enabled rules
- **Comprehensive coverage**: Style, complexity, security, imports, docstrings
- **Auto-fix enabled** for most violations
- **100-character line length** for readability

#### Type Safety (MyPy)

- **100% compliance** required for all source code
- **Minimal Any usage** - justified only where necessary
- **Strict configuration** with comprehensive type checking
- **Pandas integration** via pandas-stubs

#### Testing Standards

- **Unit test focus** - 960+ fast, isolated tests
- **80% coverage minimum** - Configured in pyproject.toml
- **Async test support** - Full FastMCP Context integration
- **Session-based patterns** - Tests mirror real MCP usage

#### MCP Documentation Standards

- **No Args sections** - Parameter docs come from Field descriptions
- **Comprehensive Field descriptions** - All MCP tool parameters documented
- **Tool-focused docstrings** - Function purpose and examples only
- **Automated enforcement** - Pre-commit hooks prevent violations

### Architectural Quality

#### Design Principles

- **Stateless MCP design** - External context handles persistence
- **Clear API boundaries** - No boolean traps, keyword-only parameters
- **Defensive programming** - Proper validation, no silent failures
- **Configurable behavior** - Environment variables, no magic numbers

#### Performance Standards

- **Resource management** - Memory thresholds, violation limits
- **Efficient operations** - Streaming, chunking for large datasets
- **Session lifecycle** - Automatic cleanup, TTL management
- **Error resilience** - Graceful degradation, comprehensive logging

## Quality Enforcement

### Pre-commit Hooks

Automated quality gates prevent regression:

- **Python quality**: AST validation, builtin checks, encoding fixes
- **Security**: Private key detection, debug statement removal
- **Linting**: Ruff with auto-fix enabled
- **Type checking**: MyPy with pandas integration
- **Documentation**: Markdown formatting, PyMarkdown linting
- **MCP compliance**: Args sections, Field descriptions validation

### CI/CD Integration

Quality checks designed for automated pipelines:

- **Proper exit codes** - Scripts return 0/1/2 for automation
- **Quiet modes** - Summary output for CI logs
- **Comprehensive reporting** - Detailed feedback for developers
- **Fast execution** - Parallel test execution, efficient tooling

### Quality Metrics Tracking

#### Current Achievements

- **0 ruff violations** (down from 200+ during development)
- **0 MyPy errors** (maintained throughout architectural changes)
- **0 Args violations** (49 removed during MCP compliance)
- **960+ tests passing** (95%+ pass rate maintained)
- **3,340+ lines eliminated** (architectural simplification)

#### Continuous Improvement

- **Regular dependency updates** - Automated via pre-commit.ci
- **Quality standard evolution** - Enhanced MCP compliance tools
- **Performance optimization** - Ongoing efficiency improvements
- **Coverage expansion** - Strategic test development for gaps

## Troubleshooting

### Common Quality Issues

#### Ruff Violations

Most common: D413 (missing blank lines), D401 (imperative mood) **Fix**: Run
`uv run ruff check --fix src/` for auto-correction

#### MyPy Errors

Most common: Missing type annotations, incorrect Any usage **Fix**: Add proper
type hints, justify Any usage with comments

#### Test Failures

Most common: Session management, async context handling **Fix**: Use proper
FastMCP Context patterns, session lifecycle management

#### MCP Documentation

Most common: Missing Field descriptions, Args sections in docstrings **Fix**:
Use our custom scripts for detection and guidance

### Emergency Quality Recovery

If quality standards are compromised:

```bash
# 1. Reset to clean state
git checkout main && git pull origin main

# 2. Apply all auto-fixes
uv run pre-commit run --all-files

# 3. Verify complete compliance
scripts/check_docstring_args.py --quiet
scripts/check_mcp_field_descriptions.py --quiet
uv run ruff check src/ --quiet
uv run --directory src mypy . --quiet

# 4. Run full test suite
uv run pytest tests/unit/ -q
```

This comprehensive quality framework ensures DataBeak maintains its position as
a reference implementation for clean, well-documented MCP server architecture.
