# Prepare databeak for release $ARGUMENTS

Take the following steps. If any step fails, fix the issue (if fixable) and
start again. If you cannot fix the issue, ask for help.

1. Run the quality checks and all tests to ensure everything is passing.
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```
1. Update the version number in `pyproject.toml` to $ARGUMENTS.
1. Update `CHANGELOG.md` with the new version ($ARGUMENTS) and changes made
   since the last release. Use GitHub issues and pull requests to help identify
   changes.
1. Run `uv build` and `tests/packaging/test_packaged_version.sh` to ensure the
   package builds correctly.
1. Commit the changes with a message like "Prepare for release x.y.z" and push
   to a new branch named `release-x.y.z`.
1. Create a pull request from the `release-x.y.z` branch to `main`
