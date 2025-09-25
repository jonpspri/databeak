# Docker Deployment Guide

This document describes how to deploy DataBeak as an HTTP streaming server using
Docker.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the HTTP streaming server
docker-compose up -d

# Check server status
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

The server will be available at `http://localhost:8000` with:

- MCP endpoint: `http://localhost:8000/mcp`
- Health check: `http://localhost:8000/health`
- Server info: `http://localhost:8000/`

### Using Docker CLI

```bash
# Build the image
docker build -t databeak:latest .

# Run the container
docker run -d \
  --name databeak \
  -p 8000:8000 \
  --restart unless-stopped \
  databeak:latest

# Check health
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

Set these environment variables to configure the server:

| Variable             | Default | Description                                 |
| -------------------- | ------- | ------------------------------------------- |
| `DATABEAK_LOG_LEVEL` | `INFO`  | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PYTHONUNBUFFERED`   | `1`     | Ensure Python output is not buffered        |

### Docker Compose Configuration

The `docker-compose.yml` provides two service profiles:

1. **Production** (default): `databeak`

   - Port: 8000
   - Log level: INFO
   - Restart policy: unless-stopped
   - Health checks enabled

1. **Development**: `databeak-dev`

   - Port: 8001
   - Log level: DEBUG
   - Source code mounted for development
   - Use: `docker-compose --profile dev up`

## Health Monitoring

The container includes built-in health checks:

```bash
# Check container health
docker ps

# View health check logs
docker inspect --format='{{json .State.Health}}' databeak
```

Health check endpoint responses:

```json
{
  "status": "healthy",
  "service": "DataBeak MCP Server",
  "transport": "http",
  "version": "DataBeak"
}
```

## Production Deployment

### Resource Requirements

Minimum requirements:

- CPU: 0.5 cores
- Memory: 512MB
- Storage: 1GB

Recommended for production:

- CPU: 1-2 cores
- Memory: 1-2GB
- Storage: 5GB

### Security Considerations

The Docker image implements security best practices:

1. **Non-root user**: Runs as `databeak` user (UID 1001)
1. **Minimal base image**: Python 3.11 slim
1. **Multi-stage build**: Reduces image size and attack surface
1. **Health checks**: Automatic container health monitoring
1. **Read-only source**: Application code mounted read-only in dev mode

### Scaling

For high-availability deployments:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  databeak:
    image: databeak:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '1'
        reservations:
          memory: 512M
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "8000:8000"
    environment:
      - DATABEAK_LOG_LEVEL=INFO
```

### Reverse Proxy

Example Nginx configuration for production:

```nginx
upstream databeak {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://databeak;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # For streaming support
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Development

### Development Mode

```bash
# Start in development mode with source mounting
docker-compose --profile dev up

# Server available at http://localhost:8001
# Source code changes are reflected without rebuild
```

### Building Custom Images

```bash
# Build with custom tag
docker build -t your-registry/databeak:v1.0.0 .

# Build with build arguments
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  -t databeak:python3.12 \
  .
```

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Check what's using port 8000
   lsof -i :8000

   # Use different port
   docker run -p 8001:8000 databeak:latest
   ```

1. **Container fails health check**

   ```bash
   # Check container logs
   docker logs databeak

   # Test health endpoint manually
   docker exec databeak curl http://localhost:8000/health
   ```

1. **Memory issues**

   ```bash
   # Monitor container resources
   docker stats databeak

   # Increase memory limit
   docker run -m 1g databeak:latest
   ```

### Debugging

```bash
# Enter running container
docker exec -it databeak /bin/bash

# View application logs
docker logs -f databeak

# Check FastMCP server status
curl -v http://localhost:8000/mcp
```

## Integration Examples

### MCP Client Connection

```python
import httpx
from mcp import ClientSession, HttpTransport


async def connect_to_databeak():
    transport = HttpTransport("http://localhost:8000/mcp")
    session = ClientSession(transport)

    # Use DataBeak tools
    result = await session.call_tool("system_info")
    print(result)
```

### HTTP API Usage

```bash
# Check server status
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# MCP endpoint (requires MCP client)
curl -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
     http://localhost:8000/mcp
```
