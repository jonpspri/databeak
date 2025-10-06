#!/usr/bin/env python3
"""Fix remaining test exception expectations."""

import re
from pathlib import Path


def fix_discovery_server_coverage():
    """Fix test_discovery_server_coverage.py."""
    file_path = Path("tests/unit/servers/test_discovery_server_coverage.py")
    content = file_path.read_text()

    # Add domain exception imports if not present
    if "from databeak.exceptions import" not in content:
        # Find where to insert
        insert_pos = content.find("\nfrom databeak.core")
        if insert_pos != -1:
            import_line = "\nfrom databeak.exceptions import ColumnNotFoundError, InvalidParameterError\n"
            content = content[:insert_pos] + import_line + content[insert_pos:]

    # Replace ToolError with appropriate domain exceptions
    replacements = [
        # test_outliers_isolation_forest, test_outliers_invalid_method, test_outliers_non_numeric_columns
        (r"with pytest\.raises\(ToolError\):\s+await detect_outliers\([^)]+method=",
         lambda m: m.group(0).replace("ToolError", "InvalidParameterError")),
        (r'with pytest\.raises\(ToolError\):\s+await detect_outliers\([^)]+columns=\["text"\]',
         lambda m: m.group(0).replace("ToolError", "InvalidParameterError")),
        # test_group_by_invalid_column, test_inspect_around_invalid_column
        (r"with pytest\.raises\(ToolError\):\s+await group_by_aggregate",
         lambda m: m.group(0).replace("ToolError", "ColumnNotFoundError")),
        (r"with pytest\.raises\(ToolError\):\s+await inspect_data_around",
         lambda m: m.group(0).replace("ToolError", "ColumnNotFoundError")),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    file_path.write_text(content)
    print("✓ Fixed test_discovery_server_coverage.py")

def fix_io_server_additional():
    """Fix test_io_server_additional.py."""
    file_path = Path("tests/unit/servers/test_io_server_additional.py")
    content = file_path.read_text()

    # Add domain exception imports if not present
    if "NoDataLoadedError" not in content:
        # Find ToolError import line
        import_line = "from fastmcp.exceptions import ToolError\n"
        if import_line in content:
            content = content.replace(import_line, "from fastmcp.exceptions import ToolError\n\nfrom databeak.exceptions import NoDataLoadedError\n")

    # Replace specific ToolError instances with NoDataLoadedError
    replacements = [
        ("with pytest.raises(ToolError):\n            await get_session_info",
         "with pytest.raises(NoDataLoadedError):\n            await get_session_info"),
        ("with pytest.raises(ToolError):\n                await export_csv",
         "with pytest.raises(NoDataLoadedError):\n                await export_csv"),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    file_path.write_text(content)
    print("✓ Fixed test_io_server_additional.py")

def main():
    """Fix all remaining test files."""
    fix_discovery_server_coverage()
    fix_io_server_additional()
    print("\nDone!")

if __name__ == "__main__":
    main()
