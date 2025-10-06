#!/usr/bin/env python3
"""Fix test files to expect domain exceptions instead of ToolError."""

import re
from pathlib import Path

# Map of exception patterns to their correct exception types
EXCEPTION_MAPPINGS = [
    (r'pytest\.raises\(ToolError, match="Column.*not found"\)', 'pytest.raises(ColumnNotFoundError, match="Column.*not found")'),
    (r'pytest\.raises\(ToolError, match=".*not found"\)', 'pytest.raises(ColumnNotFoundError, match=".*not found")'),
    (r'pytest\.raises\(ToolError, match="No data loaded in session"\)', 'pytest.raises(NoDataLoadedError, match="No data loaded in session")'),
    (r'pytest\.raises\(ToolError, match="Invalid value"\)', 'pytest.raises(InvalidParameterError, match="Invalid value")'),
    (r'pytest\.raises\(ToolError, match="Session.*not found"\)', 'pytest.raises(SessionNotFoundError, match="Session.*not found")'),
]

# Import lines to add if ToolError is imported
IMPORTS_TO_ADD = {
    "from databeak.exceptions import ColumnNotFoundError, InvalidParameterError, NoDataLoadedError, SessionNotFoundError",
}

def fix_test_file(file_path: Path) -> bool:
    """Fix a single test file."""
    content = file_path.read_text()
    original_content = content

    # Check if file uses ToolError
    if "from fastmcp.exceptions import ToolError" not in content:
        return False

    # Remove ToolError import
    content = re.sub(r"from fastmcp\.exceptions import ToolError\n", "", content)

    # Add domain exception imports if not present
    if "from databeak.exceptions import" not in content:
        # Find where to insert (after fastmcp imports)
        insert_pos = content.find("\n# Ensure full module coverage")
        if insert_pos == -1:
            insert_pos = content.find("\nimport databeak")
        if insert_pos == -1:
            insert_pos = content.find("\nfrom databeak")

        if insert_pos != -1:
            import_line = "\nfrom databeak.exceptions import ColumnNotFoundError, InvalidParameterError, NoDataLoadedError, SessionNotFoundError\n"
            content = content[:insert_pos] + import_line + content[insert_pos:]

    # Apply exception replacements
    for pattern, replacement in EXCEPTION_MAPPINGS:
        content = re.sub(pattern, replacement, content)

    # Write back if changed
    if content != original_content:
        file_path.write_text(content)
        print(f"✓ Fixed {file_path.name}")
        return True

    return False

def main():
    """Fix all test files in servers directory."""
    test_dir = Path(__file__).parent.parent / "tests" / "unit" / "servers"

    files_to_fix = [
        "test_discovery_server.py",
        "test_io_server.py",
        "test_io_server_coverage_fixes.py",
        "test_row_operations_server.py",
        "test_statistics_server.py",
        "test_statistics_server_coverage.py",
        "test_transformation_server.py",
        "test_validation_server.py",
        "test_system_server.py",
    ]

    fixed_count = 0
    for filename in files_to_fix:
        file_path = test_dir / filename
        if file_path.exists():
            if fix_test_file(file_path):
                fixed_count += 1
        else:
            print(f"✗ File not found: {filename}")

    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
