#!/usr/bin/env -S uv run --script
"""Check for Args sections in docstrings that violate MCP documentation standards.

MCP tools should use Field descriptions for parameter documentation rather than
Args sections in docstrings. This script scans the specified directories/files
or defaults to src/ directory to identify violations and reports them with file
locations and function names.

Usage:
    python scripts/check_docstring_args.py                    # Check src/ directory
    python scripts/check_docstring_args.py file1.py file2.py  # Check specific files
    python scripts/check_docstring_args.py src/databeak/servers/  # Check specific directory

Exit codes:
    0: No violations found
    1: Args sections detected (violations found)
    2: Error during scanning
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple


class ArgsViolation(NamedTuple):
    """Represents a docstring Args violation."""

    file_path: str
    function_name: str
    line_number: int
    args_line_number: int


class DocstringArgsChecker(ast.NodeVisitor):
    """AST visitor to find Args sections in function docstrings."""

    def __init__(self, file_path: str):
        """Initialize checker with file path."""
        self.file_path = file_path
        self.violations: list[ArgsViolation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and check their docstrings."""
        self._check_function_docstring(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions and check their docstrings."""
        self._check_function_docstring(node)
        self.generic_visit(node)

    def _check_function_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check if function docstring contains Args section."""
        # Only check MCP tools for Args violations
        if not self._is_mcp_tool_function(node):
            return

        # Get docstring from function
        docstring = ast.get_docstring(node)
        if not docstring:
            return

        # Split docstring into lines and check for Args sections
        lines = docstring.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for Args section indicators
            if stripped.startswith(("Args:", "Arguments:", "Parameters:")):
                # Calculate line number in original file
                # AST line numbers are 1-based, docstring starts after function definition
                args_line_number = node.lineno + 1 + i

                violation = ArgsViolation(
                    file_path=self.file_path,
                    function_name=node.name,
                    line_number=node.lineno,
                    args_line_number=args_line_number,
                )
                self.violations.append(violation)
                break  # Only report first Args section per function

    def _is_mcp_tool_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Determine if function appears to be an MCP tool."""
        # MCP tools must have a Context parameter as first argument (after self)
        if not node.args.args:
            return False

        # Look for Context parameter (usually first or second parameter)
        has_context = False
        for arg in node.args.args[:2]:  # Check first two args
            if arg.annotation:
                annotation_str = ast.dump(arg.annotation)
                if "Context" in annotation_str and "Field" in annotation_str:
                    has_context = True
                    break

        if not has_context:
            return False

        # Additional validation: should have Result return type
        if node.returns:
            returns_str = ast.dump(node.returns)
            if "Result" in returns_str:
                return True

        # Or should be in a servers/ file (MCP tools are typically in servers)
        return "/servers/" in self.file_path


def scan_file(file_path: Path) -> list[ArgsViolation]:
    """Scan a single Python file for Args violations.

    Args:
        file_path: Path to Python file to scan

    Returns:
        List of violations found in the file
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file using AST
        tree = ast.parse(content)

        # Check for Args violations
        checker = DocstringArgsChecker(str(file_path))
        checker.visit(tree)

        return checker.violations

    except (SyntaxError, UnicodeDecodeError, OSError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def scan_directory(src_dir: Path) -> list[ArgsViolation]:
    """Scan all Python files in directory for Args violations.

    Args:
        src_dir: Path to directory to scan

    Returns:
        List of all violations found
    """
    all_violations = []

    # Find all Python files in directory
    python_files = list(src_dir.rglob("*.py"))

    if not python_files:
        print(f"Warning: No Python files found in {src_dir}", file=sys.stderr)
        return []

    print(f"Scanning {len(python_files)} Python files in {src_dir}...")

    for file_path in python_files:
        violations = scan_file(file_path)
        all_violations.extend(violations)

    return all_violations


def scan_paths(paths: list[str]) -> list[ArgsViolation]:
    """Scan specified paths (files or directories) for Args violations.

    Args:
        paths: List of file or directory paths to scan

    Returns:
        List of all violations found
    """
    all_violations = []
    all_files = []

    for path_str in paths:
        path = Path(path_str)

        if not path.exists():
            print(f"Warning: Path does not exist: {path}", file=sys.stderr)
            continue

        if path.is_file():
            if path.suffix == ".py":
                all_files.append(path)
            else:
                print(f"Warning: Skipping non-Python file: {path}", file=sys.stderr)
        elif path.is_dir():
            python_files = list(path.rglob("*.py"))
            all_files.extend(python_files)
            print(f"Found {len(python_files)} Python files in {path}")
        else:
            print(f"Warning: Unknown path type: {path}", file=sys.stderr)

    if not all_files:
        print("Warning: No Python files found to scan", file=sys.stderr)
        return []

    print(f"Scanning {len(all_files)} Python files total...")

    for file_path in all_files:
        violations = scan_file(file_path)
        all_violations.extend(violations)

    return all_violations


def format_violations_report(violations: list[ArgsViolation]) -> str:
    """Format violations into a readable report.

    Args:
        violations: List of violations to format

    Returns:
        Formatted report string
    """
    if not violations:
        return "✅ No Args sections found in docstrings - MCP documentation standards met!"

    report_lines = [
        f"❌ Found {len(violations)} Args section(s) in docstrings",
        "",
        "MCP tools should use Field descriptions for parameter documentation",
        "instead of Args sections in docstrings.",
        "",
        "Violations found:",
        "",
    ]

    # Group by file for cleaner output
    by_file: dict[str, list[ArgsViolation]] = {}
    for violation in violations:
        if violation.file_path not in by_file:
            by_file[violation.file_path] = []
        by_file[violation.file_path].append(violation)

    for file_path, file_violations in sorted(by_file.items()):
        report_lines.append(f"📁 {file_path}")
        for violation in file_violations:
            report_lines.append(
                f"   └─ {violation.function_name}() at line {violation.line_number} "
                f"(Args section at line {violation.args_line_number})"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "To fix these violations:",
            "1. Remove the Args section from the docstring",
            "2. Ensure Field descriptions are comprehensive",
            "3. Keep only the function summary and examples in docstrings",
            "",
            "Example of proper MCP tool documentation:",
            "def my_tool(",
            '    ctx: Annotated[Context, Field(description="FastMCP context")],',
            '    param: Annotated[str, Field(description="Parameter description")]',
            ") -> Result:",
            '    """Brief tool description."""',
            "    # Implementation...",
        ]
    )

    return "\n".join(report_lines)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Check for Args sections in docstrings that violate MCP documentation standards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Check default src/ directory
    %(prog)s file1.py file2.py            # Check specific files
    %(prog)s src/databeak/servers/        # Check specific directory
    %(prog)s src/databeak/servers/ tests/ # Check multiple directories

MCP tools should use Field descriptions for parameter documentation
instead of Args sections in docstrings.
        """,
    )

    parser.add_argument(
        "paths", nargs="*", help="Files or directories to check (default: src/ directory)"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only show summary, not detailed violations"
    )

    return parser.parse_args()


def main() -> int:
    """Execute main script logic.

    Returns:
        Exit code: 0 for success, 1 for violations found, 2 for errors
    """
    try:
        args = parse_arguments()

        # Determine what to scan
        if args.paths:
            # Use specified paths
            violations = scan_paths(args.paths)
        else:
            # Default to src directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            src_dir = project_root / "src"

            if not src_dir.exists():
                print(f"Error: src directory not found at {src_dir}", file=sys.stderr)
                return 2

            violations = scan_directory(src_dir)

        # Generate and print report
        if args.quiet:
            if violations:
                print(f"❌ Found {len(violations)} Args section(s) in docstrings")
                return 1
            print("✅ No Args sections found - MCP documentation standards met!")
            return 0
        report = format_violations_report(violations)
        print(report)
        return 1 if violations else 0

    except Exception as e:
        print(f"Error during scanning: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
