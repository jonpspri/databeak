#!/bin/bash -eu

set -o pipefail

SCRIPT_DIR=$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

version=$(sed -n -E '/^version/s/^.*"([0-9]+\.[0-9]+\.[0-9]+)"$/\1/p' \
    "${SCRIPT_DIR}/../../pyproject.toml" )
echo "Testing packaged version: $version"

# Validate semver format (MAJOR.MINOR.PATCH)
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Version '$version' is not a valid three-part semver (expected MAJOR.MINOR.PATCH)"
    exit 1
fi

databeak_version="$(mcpt call --format json get_server_info \
    uvx --from "$SCRIPT_DIR/../../dist/databeak-$version-py3-none-any.whl" databeak | \
    jq -r '.content[0].text | fromjson | .version')"

echo "Databeak version from package: $databeak_version"

if [[ "$version" != "$databeak_version" ]]; then
    echo "Version mismatch: $version (pyproject.toml) != $databeak_version (package)"
    exit 1
fi
