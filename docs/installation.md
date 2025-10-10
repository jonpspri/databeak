---
sidebar_position: 2
title: Installation
---

## Installation Guide

Get DataBeak up and running in just 2 minutes! This guide covers installation
and client configuration.

## Prerequisites

- **Python 3.12+** (required)
- **Operating System**: Windows, macOS, or Linux
- **Package Manager**: uv (recommended) or pip

## Quick Install

### Using uvx (Recommended)

The fastest way to install and run DataBeak:

```bash
# Install and run from PyPI
uvx databeak
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
# Install from PyPI
pip install databeak

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
      "args": ["databeak"],
      "env": {
        "DATABEAK_MAX_DOWNLOAD_SIZE_MB": "200",
        "DATABEAK_SESSION_TIMEOUT": "7200"
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
      "args": ["databeak"]
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
      "args": ["databeak"]
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
      "args": ["databeak"]
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
      "args": ["databeak"]
    }
  }
}
```

## Environment Variables

Configure DataBeak behavior with these environment variables:

### Core Configuration

| Variable                                      | Default | Description                              |
| --------------------------------------------- | ------- | ---------------------------------------- |
| `DATABEAK_MAX_FILE_SIZE_MB`                   | 1024    | Maximum file size in MB                  |
| `DATABEAK_SESSION_TIMEOUT`                    | 3600    | Session timeout in seconds               |
| `DATABEAK_CHUNK_SIZE`                         | 10000   | Processing chunk size for large datasets |
| `DATABEAK_MEMORY_THRESHOLD_MB`                | 2048    | Memory threshold for health monitoring   |
| `DATABEAK_MEMORY_WARNING_THRESHOLD`           | 0.75    | Memory ratio triggering warning (0-1)    |
| `DATABEAK_MEMORY_CRITICAL_THRESHOLD`          | 0.90    | Memory ratio triggering critical (0-1)   |
| `DATABEAK_SESSION_CAPACITY_WARNING_THRESHOLD` | 0.90    | Session capacity warning ratio (0-1)     |
| `DATABEAK_MAX_VALIDATION_VIOLATIONS`          | 1000    | Max validation violations to report      |
| `DATABEAK_MAX_ANOMALY_SAMPLE_SIZE`            | 10000   | Max sample size for anomaly detection    |

### HTTP Mode Configuration

For HTTP transport mode (`--transport http`), additional configuration options
are available:

#### OIDC Authentication (HTTP Mode Only)

OpenID Connect authentication for secure HTTP deployments. All four variables
must be set to enable OIDC:

| Variable                      | Required | Description                      |
| ----------------------------- | -------- | -------------------------------- |
| `DATABEAK_OIDC_CONFIG_URL`    | Yes      | OIDC discovery configuration URL |
| `DATABEAK_OIDC_CLIENT_ID`     | Yes      | OAuth2 client ID                 |
| `DATABEAK_OIDC_CLIENT_SECRET` | Yes      | OAuth2 client secret             |
| `DATABEAK_OIDC_BASE_URL`      | Yes      | Application base URL for OAuth2  |

**Example HTTP deployment with OIDC:**

```bash
export DATABEAK_OIDC_CONFIG_URL="https://auth.example.com/.well-known/openid-configuration"
export DATABEAK_OIDC_CLIENT_ID="databeak-client"
export DATABEAK_OIDC_CLIENT_SECRET="your-secret"
export DATABEAK_OIDC_BASE_URL="https://databeak.example.com"
uvx databeak --transport http --host 0.0.0.0 --port 8000
```

**Note**: OIDC authentication is only applicable for HTTP transport mode. Stdio
mode (default for MCP clients) does not use OIDC authentication.

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
mcp-inspector uvx databeak
```

### Verify in Your AI Client

1. **Claude Desktop**: Look for "databeak" in the MCP servers list
1. **VS Code**: Check the extension's MCP panel
1. **Test Command**: Try asking your AI to "list available CSV tools"

## Troubleshooting

### Common Issues

#### Server not starting

- Check Python version: `python --version` (must be 3.12+)
- Verify installation: `uvx databeak --version`
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
- Adjust `DATABEAK_MEMORY_THRESHOLD_MB` for available system memory

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
