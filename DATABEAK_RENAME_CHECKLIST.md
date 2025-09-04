# DataBeak Rename Checklist

Based on analysis of the codebase, here's a comprehensive checklist of places that need updates for the rename from CSV Editor to DataBeak:

## Package Configuration Files
- [ ] `pyproject.toml:6` - Update package name from "csv-editor" to "databeak"
- [ ] `pyproject.toml:8` - Update description to mention DataBeak
- [ ] `pyproject.toml:108` - Update self-reference in optional dependencies
- [ ] `pyproject.toml:112-117` - Update all GitHub URLs from csv-editor to databeak
- [ ] `pyproject.toml:120-121` - Update script names (csv-editor, csv) to databeak equivalents
- [ ] `pyproject.toml:302` - Update coverage report title
- [ ] `pyproject.toml:310-314` - Update MkDocs site configuration
- [ ] `package.json:2` - Update npm package name
- [ ] `package.json:4` - Update description
- [ ] `package.json:17-24` - Update all GitHub URLs
- [ ] `package.json:46` - Update MCP module path if needed

## Source Code Directory Structure
- [ ] Rename `src/csv_editor/` directory to `src/databeak/`
- [ ] Update all import statements throughout codebase (40+ files affected)
- [ ] `src/csv_editor/_version.py` - Update internal references

## Documentation Files
- [ ] `README.md` - Replace all "CSV Editor" references with "DataBeak"
- [ ] `CLAUDE.md` - Update project context and instructions
- [ ] `CHANGELOG.md` - Add entry for rename, update historical references
- [ ] `CONTRIBUTING.md` - Update project name references
- [ ] `SECURITY.md` - Update project name references
- [ ] `NOTICE` - Update project name references

## Documentation Site (docs/)
- [ ] `docs/docs/intro.md` - Update introduction
- [ ] `docs/docs/installation.md` - Update installation instructions
- [ ] `docs/docs/architecture.md` - Update architecture documentation
- [ ] `docs/docs/tutorials/quickstart.md` - Update tutorial content
- [ ] `docs/docs/api/overview.md` - Update API documentation
- [ ] `docs/docs/VERSION_MANAGEMENT.md` - Update version management docs
- [ ] `docs/docusaurus.config.ts` - Update site configuration
- [ ] `docs/package.json` - Update npm package metadata

## Test Files (tests/)
- [ ] Update all test file imports and references (21+ test files)
- [ ] `tests/README.md` - Update test documentation
- [ ] `tests/conftest.py` - Update test configuration

## Example Files (examples/)
- [ ] Update all example file imports and references (8+ example files)
- [ ] `examples/README.md` - Update examples documentation

## Infrastructure Files
- [ ] `smithery.yaml` - Update MCP server configuration
- [ ] `Dockerfile` - Update container configuration
- [ ] `.github/ISSUE_TEMPLATE/` - Update issue templates

## Scripts
- [ ] `scripts/sync_versions.py` - Update version sync script references
- [ ] `scripts/publish.py` - Update publishing script

## Tool Definitions (40+ MCP tools)
All files in `src/csv_editor/tools/` need their descriptions and docstrings updated:
- [ ] `mcp_analytics_tools.py` - Update tool descriptions
- [ ] `mcp_data_tools.py` - Update tool descriptions  
- [ ] `mcp_history_tools.py` - Update tool descriptions
- [ ] `mcp_io_tools.py` - Update tool descriptions
- [ ] `mcp_row_tools.py` - Update tool descriptions
- [ ] `mcp_system_tools.py` - Update tool descriptions
- [ ] `mcp_validation_tools.py` - Update tool descriptions

## Environment Variables
- [ ] Update `CSV_EDITOR_` prefixed environment variables to `DATABEAK_`
- [ ] Update `CSVSettings` class name in `csv_session.py`

## GitHub Repository Settings
- [ ] Repository name: csv-editor → databeak
- [ ] Repository description
- [ ] Topics/tags
- [ ] Branch protection rules (if any reference the old name)

## Priority Order
1. **Package Configuration** - Start with `pyproject.toml` and `package.json`
2. **Directory Structure** - Rename `src/csv_editor/` to `src/databeak/`
3. **Import Statements** - Update all module imports across 80+ files
4. **Documentation** - Update user-facing documentation
5. **Tests & Examples** - Update test and example code
6. **Infrastructure** - Update deployment and CI configuration

## Notes
- The rename affects 80+ files across the codebase
- Pay special attention to module import paths
- All GitHub URLs need to be updated consistently
- Environment variable prefixes need to change from `CSV_EDITOR_` to `DATABEAK_`
- Consider creating a migration script for bulk find/replace operations