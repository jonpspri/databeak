---
sidebar_position: 2
title: Installation
---

# Installation Guide

Get CSV Editor up and running in just 2 minutes! This guide covers installation and client configuration.

## Prerequisites

- **Python 3.10+** (3.11+ recommended for best performance)
- **Operating System**: Windows, macOS, or Linux
- **Package Manager**: uv (recommended) or pip

## Quick Install

### Using uvx (Recommended)

The fastest way to install and run CSV Editor:

```bash
# Install and run directly from GitHub
uvx --from git+https://github.com/jonpspri/csv-editor.git csv-editor
```

### Using uv

For development or local installation:

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and install
git clone https://github.com/jonpspri/csv-editor.git
cd csv-editor
uv sync

# Run the server
uv run csv-editor
```

### Using pip

```bash
# Install directly from GitHub
pip install git+https://github.com/jonpspri/csv-editor.git

# Run the server
csv-editor
```

## Client Configuration

### Claude Desktop

Configure Claude Desktop to use CSV Editor as an MCP server.

Add this to your MCP Settings file (Claude → Settings → Developer → Show MCP Settings):

```json
{
  "mcpServers": {
    "csv-editor": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jonpspri/csv-editor.git", "csv-editor"],
      "env": {
        "CSV_EDITOR_MAX_FILE_SIZE_MB": "1024",
        "CSV_EDITOR_CSV_HISTORY_DIR": "/tmp/csv_history"
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
    "csv-editor": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jonpspri/csv-editor.git", "csv-editor"]
    }
  }
}
```

### Cline

Add to VS Code settings (`settings.json`):

```json
{
  "cline.mcpServers": {
    "csv-editor": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jonpspri/csv-editor.git", "csv-editor"]
    }
  }
}
```

### Windsurf

Edit `~/.windsurf/mcp_servers.json`:

```json
{
  "mcpServers": {
    "csv-editor": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jonpspri/csv-editor.git", "csv-editor"]
    }
  }
}
```

### Zed Editor

Edit `~/.config/zed/settings.json`:

```json
{
  "experimental.mcp_servers": {
    "csv-editor": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jonpspri/csv-editor.git", "csv-editor"]
    }
  }
}
```

## Environment Variables

Configure CSV Editor behavior with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_EDITOR_MAX_FILE_SIZE_MB` | 1024 | Maximum file size in MB |
| `CSV_EDITOR_CSV_HISTORY_DIR` | "." | History directory path |
| `CSV_EDITOR_SESSION_TIMEOUT` | 3600 | Session timeout in seconds |
| `CSV_EDITOR_CHUNK_SIZE` | 10000 | Processing chunk size |
| `CSV_EDITOR_AUTO_SAVE` | true | Enable auto-save |

## Verification

### Test the Installation

```bash
# Check if server starts (if installed locally)
uv run csv-editor --help

# Run with verbose output
CSV_EDITOR_LOG_LEVEL=DEBUG uv run csv-editor
```

### Test with MCP Inspector

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test the server
mcp-inspector uvx --from git+https://github.com/jonpspri/csv-editor.git csv-editor
```

### Verify in Your AI Client

1. **Claude Desktop**: Look for "csv-editor" in the MCP servers list
2. **VS Code**: Check the extension's MCP panel
3. **Test Command**: Try asking your AI to "list available CSV tools"

## Troubleshooting

### Common Issues

#### Server not starting

- Check Python version: `python --version` (must be 3.10+)
- Verify installation: `uvx --from git+https://github.com/jonpspri/csv-editor.git csv-editor --version`
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
- Set appropriate `CSV_EDITOR_MAX_FILE_SIZE_MB` for your use case
- Configure `CSV_EDITOR_CHUNK_SIZE` for large datasets
- Use SSD storage for `CSV_EDITOR_CSV_HISTORY_DIR`

### Getting Help

- **[GitHub Issues](https://github.com/jonpspri/csv-editor/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/jonpspri/csv-editor/discussions)** - Ask questions
- **[Documentation](/)** - Browse complete docs

## Next Steps

Now that CSV Editor is installed:

1. **[Quick Start Tutorial](./tutorials/quickstart)** - Learn the basics
2. **[API Reference](./api/overview)** - Explore all available tools
3. **[Examples](https://github.com/jonpspri/csv-editor/tree/main/examples)** - See real-world use cases

---

**Installation complete!** Your AI assistant now has powerful data manipulation capabilities. 🎉
