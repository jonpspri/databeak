---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Install DataBeak with '...'
2. Run command '....'
3. Process CSV file '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

## Error message

```text
Paste the full error message here
```

**CSV file sample**
If possible, provide a minimal CSV file that reproduces the issue:

```csv
header1,header2,header3
value1,value2,value3
```

**Environment:**

- OS: [e.g. Ubuntu 22.04, macOS 13, Windows 11]
- Python version: [e.g. 3.11.0]
- DataBeak version: [e.g. 1.0.1]
- Installation method: [e.g. uvx --from
  git+https://github.com/jonpspri/databeak.git databeak]

**MCP Configuration**
If using with an AI assistant, please share your MCP configuration:

```json
{
  "databeak": {
    "command": "uvx",
    "args": [
      "--from",
      "git+https://github.com/jonpspri/databeak.git",
      "databeak"
    ]
  }
}
```

**Additional context**
Add any other context about the problem here.
