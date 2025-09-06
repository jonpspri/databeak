# Python Markdown Linter Research & Recommendations

## Executive Summary

Research into Python-based alternatives to `markdownlint-cli` for DataBeak's
documentation linting needs. **Recommendation**: Adopt `pymarkdownlnt` +
`mdformat` combination for pure Python markdown tooling aligned with existing
uv/Python development workflow.

## Current State: markdownlint-cli

### **What We're Using**

- **Tool**: `markdownlint-cli` v0.45.0 (Node.js-based)
- **Dependencies**: Requires Node.js/npm ecosystem
- **Configuration**: `.markdownlintignore` for excluded files
- **Integration**: Pre-commit hook in `.pre-commit-config.yaml`

### **Issues with Current Setup**

- **Node.js dependency**: Conflicts with pure Python development workflow
- **Package management**: npm dependency separate from uv Python management
- **Build complexity**: Requires Node.js in CI/CD environments
- **Maintenance overhead**: Managing both Python and Node.js dependencies

## Python-Based Alternatives Analysis

### **1. PyMarkdownLnt (Recommended Linter)**

#### **PyMarkdownLnt Overview**

- **Package**: `pymarkdownlnt` (actively maintained, latest: v0.9.26, Feb 2025)
- **Language**: Pure Python (3.9+)
- **Compliance**: GitHub Flavored Markdown + CommonMark specifications
- **Philosophy**: Comprehensive linting with token-based analysis

#### **PyMarkdownLnt Key Features**

```python
# Installation
uv add pymarkdownlnt --group dev

# Basic usage
uv run pymarkdown scan docs/
uv run pymarkdown --fix scan docs/  # Auto-fix mode
```

#### **PyMarkdownLnt Strengths**

- ✅ **Pure Python**: Integrates seamlessly with uv workflow
- ✅ **Comprehensive rules**: 46+ built-in rules (equivalent to MD\* rules)
- ✅ **Token-based analysis**: More accurate than regex-based linters
- ✅ **Auto-fix capability**: Can automatically correct many issues
- ✅ **Extensible**: Plugin system for custom rules
- ✅ **CI/CD friendly**: Single command for multiple files/directories
- ✅ **Pre-commit integration**: Built-in support

#### **Configuration Example**

```yaml
# pymarkdown.yml
default-rules: true
plugins:
  md013:
    line_length: 80
  md033:
    allowed_elements: ["details", "summary"]  # For collapsible sections
```

#### **Pre-commit Integration**

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/jackdewinter/pymarkdownlnt
  rev: v0.9.26
  hooks:
    - id: pymarkdownlnt
      args: [scan]
```

### **2. mdformat (Recommended Formatter)**

#### **mdformat Overview**

- **Package**: `mdformat` with plugins
- **Language**: Pure Python
- **Philosophy**: Opinionated formatter (like Black for Python)
- **Compliance**: Strict CommonMark with plugin extensions

#### **mdformat Key Features**

```python
# Installation with GitHub support
uv add mdformat mdformat-gfm mdformat-frontmatter --group dev

# Usage
uv run mdformat docs/ --check  # Check mode
uv run mdformat docs/          # Format in-place
```

#### **mdformat Strengths**

- ✅ **Pure Python**: No Node.js dependencies
- ✅ **Consistent formatting**: Opinionated like Black
- ✅ **Plugin ecosystem**: Extensible for specific needs
- ✅ **Fast**: Lightweight with minimal dependencies
- ✅ **Pre-commit ready**: Built-in hooks available

#### **Plugin Options for DataBeak**

```python
# Recommended plugin set
mdformat-gfm          # GitHub Flavored Markdown (tables, strikethrough)
mdformat-frontmatter  # YAML frontmatter support
mdformat-footnote     # Footnote support
mdformat-tables       # Enhanced table formatting
```

### **3. Alternative: ruff-md (Future Option)**

#### **Emerging Tool**

- **Status**: Under development by Astral (Ruff creators)
- **Concept**: Extend Ruff to markdown linting
- **Timeline**: Not yet available, but aligns with Ruff ecosystem

## Recommended Migration Strategy

### **Option A: PyMarkdownLnt Only (Simplest)**

#### Option A Risk Assessment

Risk: Low | Effort: 1 day | Compatibility: High

```python
# Replace markdownlint-cli with pymarkdownlnt
uv add pymarkdownlnt --group dev
uv remove markdownlint-cli  # Remove Node.js dependency

# Update pre-commit config
# .pre-commit-config.yaml
- repo: https://github.com/jackdewinter/pymarkdownlnt
  rev: v0.9.26
  hooks:
    - id: pymarkdownlnt
      args: [scan]
```

**Pros**: Drop-in replacement, similar rule set
**Cons**: Still just linting, no formatting

### **Option B: mdformat + pymarkdownlnt (Recommended)**

#### Option B Risk Assessment

Risk: Low | Effort: 2 days | Value: High

```python
# Install both tools
uv add pymarkdownlnt mdformat mdformat-gfm mdformat-frontmatter --group dev

# Pre-commit integration
- repo: https://github.com/executablebooks/mdformat
  rev: 0.7.22
  hooks:
    - id: mdformat
      additional_dependencies:
        - mdformat-gfm
        - mdformat-frontmatter
      args: [--wrap=80]

- repo: https://github.com/jackdewinter/pymarkdownlnt
  rev: v0.9.26
  hooks:
    - id: pymarkdownlnt
      args: [scan]
```

**Pros**: Formatting + linting, pure Python, comprehensive
**Cons**: Two tools to configure and maintain

### **Option C: mdformat Only (Minimalist)**

#### Option C Risk Assessment

Risk: Medium | Effort: 1 day | Trade-offs: Less validation

```python
# Use only mdformat with strict settings
uv add mdformat mdformat-gfm mdformat-frontmatter --group dev

# Rely on consistent formatting to prevent most issues
uv run mdformat docs/ --check --wrap=80
```

**Pros**: Single tool, automatic formatting
**Cons**: Less comprehensive validation than dedicated linter

## Feature Comparison Matrix

| Feature                | markdownlint-cli | pymarkdownlnt  | mdformat         |
| ---------------------- | ---------------- | -------------- | ---------------- |
| **Language**           | Node.js          | Pure Python    | Pure Python      |
| **Primary Purpose**    | Linting          | Linting        | Formatting       |
| **Rule Count**         | 50+ MD\* rules   | 46+ MD\* rules | Style rules only |
| **Auto-fix**           | Limited          | Yes            | Complete         |
| **Performance**        | Fast             | Fast           | Very Fast        |
| **GitHub Integration** | Excellent        | Excellent      | Good             |
| **Plugin System**      | Limited          | Yes            | Extensive        |
| **Pre-commit Support** | Native           | Native         | Native           |
| **Configuration**      | JSON/YAML        | YAML/TOML      | pyproject.toml   |
| **Dependencies**       | Node.js + npm    | Python only    | Python only      |

## DataBeak-Specific Considerations

### **Alignment with Project Goals**

1. **Pure Python stack**: pymarkdownlnt + mdformat eliminate Node.js
1. **uv integration**: Both tools install via `uv add --group dev`
1. **CI/CD simplification**: No Node.js setup required in GitHub Actions
1. **Maintenance**: Single language ecosystem reduces complexity

### **Current Pain Points Addressed**

```python
# Current issues with markdownlint-cli:
docs_docusaurus_backup/blog/2019-05-29-long-blog-post.md:14:81 MD013/line-length
# Too many false positives on backup files

# Solution with Python tools:
# 1. Better ignore patterns in pymarkdownlnt
# 2. Auto-formatting with mdformat reduces manual fixes
# 3. Pure Python configuration in pyproject.toml
```

### **Integration with Existing Tools**

```toml
# pyproject.toml integration
[tool.pymarkdownlnt]
plugins.md013.line_length = 80
plugins.md033.allowed_elements = ["details", "summary"]
plugins.md024.siblings_only = true  # Allow duplicate headers in different sections

[tool.mdformat]
wrap = 80
number = true  # Number ordered lists consistently
end_of_line = "lf"
```

## Migration Implementation Plan

### **Phase 1: Add Python Tools (Day 1)**

```bash
# Add new Python-based tools
uv add pymarkdownlnt mdformat mdformat-gfm mdformat-frontmatter --group dev

# Test with existing documentation
uv run pymarkdownlnt scan docs/
uv run mdformat docs/ --check

# Compare results with current markdownlint-cli output
```

### **Phase 2: Update Configuration (Day 1-2)**

```bash
# Create pymarkdownlnt configuration
touch .pymarkdownlnt.yml

# Update pre-commit configuration
# Add Python-based hooks alongside existing markdownlint-cli

# Test pre-commit integration
pre-commit run --all-files
```

### **Phase 3: Parallel Testing (Day 2-3)**

```bash
# Run both systems in parallel
uv run pymarkdownlnt scan docs/ > pymarkdown_results.txt
markdownlint docs/ > markdownlint_results.txt

# Compare results and tune configuration
# Ensure no regressions in validation coverage
```

### **Phase 4: Migration (Day 3)**

```bash
# Remove Node.js dependency
# Update .pre-commit-config.yaml to use Python tools only
# Update CI/CD workflows to remove Node.js setup
# Update CLAUDE.md with new commands
```

## Performance & Dependency Analysis

### **Build Time Impact**

```bash
# Current (with Node.js):
setup-node -> install-deps -> run-markdownlint
# ~30-60 seconds in CI

# Proposed (Python-only):
setup-python -> uv-sync -> run-pymarkdownlnt
# ~10-20 seconds in CI (no Node.js setup)
```

### **Local Development Impact**

```bash
# Current workflow:
npm install (if node_modules missing)
markdownlint docs/

# Proposed workflow:
uv run pymarkdownlnt scan docs/
uv run mdformat docs/ --check
```

### **Dependency Reduction**

- **Remove**: Node.js, npm, markdownlint-cli package
- **Add**: pymarkdownlnt, mdformat (pure Python, ~2MB total)
- **Net result**: Simpler dependency management, faster CI builds

## Recommended Configuration

### **pyproject.toml Integration**

```toml
# Add to pyproject.toml
[dependency-groups]
dev = [
    # ... existing dev dependencies ...
    "pymarkdownlnt>=0.9.26",
    "mdformat>=0.7.22",
    "mdformat-gfm>=0.3.5",
    "mdformat-frontmatter>=2.0.8",
]

[tool.pymarkdownlnt]
plugins.md013.line_length = 80
plugins.md033.allowed_elements = ["details", "summary"]
plugins.md024.siblings_only = true

[tool.mdformat]
wrap = 80
number = true
end_of_line = "lf"
```

### **Pre-commit Configuration**

```yaml
# .pre-commit-config.yaml replacement
- repo: https://github.com/executablebooks/mdformat
  rev: 0.7.22
  hooks:
    - id: mdformat
      additional_dependencies:
        - mdformat-gfm
        - mdformat-frontmatter
      args: [--wrap=80]

- repo: https://github.com/jackdewinter/pymarkdownlnt
  rev: v0.9.26
  hooks:
    - id: pymarkdownlnt
      args: [scan]
```

### **CLAUDE.md Commands Update**

```bash
# Replace:
markdownlint docs/

# With:
uv run mdformat docs/ --check    # Check formatting
uv run mdformat docs/            # Auto-format
uv run pymarkdownlnt scan docs/  # Lint for issues
```

## Risk Assessment & Migration Path

### **Low Risk Migration**

- **Backward compatibility**: Both tools validate same MD\* rules
- **Gradual adoption**: Can run both systems in parallel during transition
- **Rollback available**: Keep current system until Python tools proven
- **Documentation impact**: Minimal - same rules, better tooling

### **Immediate Benefits**

- **Simplified CI/CD**: No Node.js setup required
- **Consistent tooling**: All development tools in Python/uv ecosystem
- **Better integration**: Configuration in pyproject.toml
- **Auto-formatting**: Reduce manual markdown formatting work

### **Success Metrics**

- [ ] All current markdown files pass new linting
- [ ] CI/CD build time improvement (target: 50% faster)
- [ ] Pre-commit hooks run successfully
- [ ] No regression in documentation quality standards

## Next Steps

1. **Get approval** for Python markdown tooling migration
1. **Implement Option B** (mdformat + pymarkdownlnt)
1. **Test in parallel** with existing markdownlint-cli
1. **Migrate configuration** and remove Node.js dependency
1. **Update documentation** and development workflow

______________________________________________________________________

**Conclusion**: PyMarkdownLnt + mdformat provides a superior, pure Python
markdown tooling solution that aligns perfectly with DataBeak's development
philosophy while maintaining all current validation capabilities and adding
automatic formatting benefits.
