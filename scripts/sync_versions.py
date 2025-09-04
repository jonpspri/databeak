#!/usr/bin/env python3
"""Synchronize version numbers across all project files."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


def get_package_version() -> str:
    """Get the version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def update_package_json(version: str) -> None:
    """Update version in package.json."""
    package_json_path = Path(__file__).parent.parent / "package.json"
    with open(package_json_path) as f:
        data = json.load(f)

    data["version"] = version

    with open(package_json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_version_file(version: str) -> None:
    """Update the _version.py file fallback version."""
    version_file = Path(__file__).parent.parent / "src" / "csv_editor" / "_version.py"
    content = version_file.read_text()

    # Update the fallback version
    updated_content = re.sub(r'__version__ = "[^"]+dev"', f'__version__ = "{version}-dev"', content)

    version_file.write_text(updated_content)


def main() -> None:
    """Synchronize all version numbers."""
    version = get_package_version()
    print(f"Synchronizing version to {version}")

    update_package_json(version)
    print("✓ Updated package.json")

    update_version_file(version)
    print("✓ Updated _version.py")

    print("All version numbers synchronized!")


if __name__ == "__main__":
    main()
