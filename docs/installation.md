---
sidebar_position: 2
title: Installation
---

## Installation Guide

Get DataBeak up and running in just 2 minutes! This guide covers installation
and client configuration.

## Prerequisites

- **Python 3.10+** (3.11+ recommended for best performance)
- **Operating System**: Windows, macOS, or Linux
- **Package Manager**: uv (recommended) or pip

## Quick Install

### Using uvx (Recommended)

The fastest way to install and run DataBeak:

```bash
# Install and run directly from GitHub
uvx --from git+https://github.com/jonpspri/databeak.git databeak
```

### Using uv

For development or local installation:

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and install
git clone https://github.com/jonpspri/databeak.git
cd databeak
uv sync

# Run the server
uv run databeak
```

### Using pip

```bash
# Install directly from GitHub
pip install git+https://github.com/jonpspri/databeak.git

# Run the server
databeak
```

## Client Configuration

### Claude Desktop

Configure Claude Desktop to use DataBeak as an MCP server.

Add this to your MCP Settings file (Claude → Settings → Developer → Show MCP
Settings):

```json
{
  "mcpServers": {
    "databeak": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/jonpspri/databeak.git",
        "databeak"
      ],
      "env": {
        "DATABEAK_MAX_FILE_SIZE_MB": "1024",
        "DATABEAK_CSV_HISTORY_DIR": "/tmp/csv_history"
      }
    }
  }
}
```

### Continue (VS Code)

Edit `~/.continue/config.json`:

```json
{
  "mcpServers": {
    "databeak": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/jonpspri/databeak.git",
        "databeak"
      ]
    }
  }
}
```

### Cline

Add to VS Code settings (`settings.json`):

```json
{
  "cline.mcpServers": {
    "databeak": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/jonpspri/databeak.git",
        "databeak"
      ]
    }
  }
}
```

### Windsurf

Edit `~/.windsurf/mcp_servers.json`:

```json
{
  "mcpServers": {
    "databeak": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/jonpspri/databeak.git",
        "databeak"
      ]
    }
  }
}
```

### Zed Editor

Edit `~/.config/zed/settings.json`:

```json
{
  "experimental.mcp_servers": {
    "databeak": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/jonpspri/databeak.git",
        "databeak"
      ]
    }
  }
}
```

## Environment Variables

Configure DataBeak behavior with these environment variables:

| Variable                    | Default | Description                |
| --------------------------- | ------- | -------------------------- |
| `DATABEAK_MAX_FILE_SIZE_MB` | 1024    | Maximum file size in MB    |
| `DATABEAK_CSV_HISTORY_DIR`  | "."     | History directory path     |
| `DATABEAK_SESSION_TIMEOUT`  | 3600    | Session timeout in seconds |
| `DATABEAK_CHUNK_SIZE`       | 10000   | Processing chunk size      |
| `DATABEAK_AUTO_SAVE`        | true    | Enable auto-save           |

## Verification

### Test the Installation

```bash
# Check if server starts (if installed locally)
uv run databeak --help

# Run with verbose output
DATABEAK_LOG_LEVEL=DEBUG uv run databeak
```

### Test with MCP Inspector

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test the server
mcp-inspector uvx --from \
  git+https://github.com/jonpspri/databeak.git databeak
```

### Verify in Your AI Client

1. **Claude Desktop**: Look for "databeak" in the MCP servers list
1. **VS Code**: Check the extension's MCP panel
1. **Test Command**: Try asking your AI to "list available CSV tools"

## Troubleshooting

### Common Issues

#### Server not starting

- Check Python version: `python --version` (must be 3.10+)
- Verify installation:
  `uvx --from \ git+https://github.com/jonpspri/databeak.git databeak --version`
- Check logs with debug level

#### Client can't connect

- Verify the command path in your configuration
- Ensure uvx is installed and accessible
- Check firewall settings for local connections

#### Permission errors

- On macOS/Linux: Check file permissions
- On Windows: Run as administrator if needed
- Verify the history directory is writable

### Performance Tips

- Use uv instead of pip for faster package management
- Set appropriate `DATABEAK_MAX_FILE_SIZE_MB` for your use case
- Configure `DATABEAK_CHUNK_SIZE` for large datasets
- Use SSD storage for `DATABEAK_CSV_HISTORY_DIR`

### Getting Help

- **[GitHub Issues](https://github.com/jonpspri/databeak/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/jonpspri/databeak/discussions)** Ask
  questions
- **[Documentation](index.md)** - Browse complete docs

## Next Steps

Now that DataBeak is installed:

1. **[Quick Start Tutorial](tutorials/quickstart.md)** - Learn the basics
1. **[API Reference](api/index.md)** - Explore all available tools
1. **[Examples](https://github.com/jonpspri/databeak/tree/main/examples)**
   - See real-world use cases

______________________________________________________________________

**Installation complete!** Your AI assistant now has powerful data manipulation
capabilities.
