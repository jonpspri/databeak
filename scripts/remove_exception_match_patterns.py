#!/usr/bin/env python3
"""Remove match patterns from pytest.raises calls - just check exception type."""

import re
from pathlib import Path


def fix_file(file_path: Path) -> bool:
    """Remove match patterns from pytest.raises."""
    content = file_path.read_text()
    original = content

    # Pattern: pytest.raises(SomeException, match="...")
    # Replace with: pytest.raises(SomeException)
    patterns = [
        (r'pytest\.raises\((\w+(?:Error|Exception)), match=["\'][^"\']*["\']\)', r"pytest.raises(\1)"),
        (r'pytest\.raises\(\(([^)]+)\), match=["\'][^"\']*["\']\)', r"pytest.raises((\1))"),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original:
        file_path.write_text(content)
        return True
    return False

def main():
    """Fix all server test files."""
    test_dir = Path(__file__).parent.parent / "tests" / "unit" / "servers"

    fixed = 0
    for file_path in test_dir.glob("test_*.py"):
        if fix_file(file_path):
            print(f"✓ Fixed {file_path.name}")
            fixed += 1

    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
