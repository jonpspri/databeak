---
sidebar_position: 5
title: Version Management
---

## Version Management

This document describes how version numbers are managed in the DataBeak project.

## Current Approach - Script-Based Version Sync

### 1. Single Source of Truth

- The **primary version** is defined in `pyproject.toml` under `project.version`
- All other files are synchronized from this single source

### 2. Dynamic Version Loading

The application uses `importlib.metadata` to read the version at runtime:

```python
# In src/databeak/_version.py
import importlib.metadata

try:
    __version__ = importlib.metadata.version("databeak")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development mode
    __version__ = "1.0.2-dev"
```

### 3. Version Synchronization Script

Run this command to sync versions across all files:

```bash
# Using uv (recommended)
uv run sync-versions

# Or directly with Python
python scripts/sync_versions.py
```

The script automatically updates:

- `package.json` - For npm/MCP registry compatibility
- `src/databeak/_version.py` - Fallback version for development

## Release Workflow

### Simple 3-Step Process

1. **Update the version in pyproject.toml:**

   ```toml
   [project]
   version = "1.0.3"  # Change this line
   ```

1. **Sync all other files:**

   ```bash
   uv run sync-versions
   ```

1. **Commit and tag:**

   ```bash
   git add .
   git commit -m "🔖 Bump version to 1.0.3"
   git tag "v1.0.3"
   git push origin main --tags
   ```

## Files That Are Synchronized

1. **pyproject.toml** - Primary source of truth ✅
1. **package.json** - Synced automatically 🔄
1. **src/databeak/\_version.py** - Synced automatically 🔄

## Benefits

1. **Simplicity** - Easy to understand and maintain
1. **Reliability** - No complex external tool dependencies
1. **Consistency** - One source of truth prevents version mismatches
1. **Runtime accuracy** - Code always uses the actual installed package version
1. **Development friendly** - Fallback version for development mode

## Advanced Options (Future)

While the current script-based approach works well, you could consider these
tools for more automation:

- **semantic-release** - Fully automated releases based on commit messages
- **hatch version** - Built into the build system
- **GitHub Actions** - Automate the entire release process

### Example GitHub Action Workflow

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run sync-versions
      - run: uv build
      - run: uv publish
```

## Best Practices

- Always run `uv run sync-versions` after changing the version in pyproject.toml
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Test the sync in your development environment before releases
- Keep the sync script simple and maintainable
- Document version changes in CHANGELOG.md
