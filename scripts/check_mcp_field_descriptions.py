#!/usr/bin/env -S uv run --script
"""Check that all MCP tool parameters have Field descriptions.

MCP tools should have comprehensive Field descriptions for all parameters
(except ctx: Context which is excluded from this requirement). This script
scans the specified directories/files or defaults to src/ directory to identify
parameters without proper Field descriptions.

Usage:
    python scripts/check_mcp_field_descriptions.py                    # Check src/ directory
    python scripts/check_mcp_field_descriptions.py file1.py file2.py  # Check specific files
    python scripts/check_mcp_field_descriptions.py src/databeak/servers/  # Check specific directory

Exit codes:
    0: No violations found
    1: Missing Field descriptions detected (violations found)
    2: Error during scanning
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple


class FieldDescriptionViolation(NamedTuple):
    """Represents a missing Field description violation."""

    file_path: str
    function_name: str
    parameter_name: str
    line_number: int


class MCPFieldChecker(ast.NodeVisitor):
    """AST visitor to find MCP tool parameters without Field descriptions."""

    def __init__(self, file_path: str):
        """Initialize checker with file path."""
        self.file_path = file_path
        self.violations: list[FieldDescriptionViolation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and check for MCP tool patterns."""
        self._check_mcp_tool_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions and check for MCP tool patterns."""
        self._check_mcp_tool_function(node)
        self.generic_visit(node)

    def _check_mcp_tool_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check if function is an MCP tool and validate Field descriptions."""
        # Skip functions that don't look like MCP tools
        if not self._is_mcp_tool_function(node):
            return

        # Check each parameter for Field description
        for arg in node.args.args:
            param_name = arg.arg

            # Skip 'self', 'cls', and 'ctx' parameters
            if param_name in ("self", "cls", "ctx"):
                continue

            # Check if parameter has proper Field annotation
            if not self._has_field_description(arg):
                violation = FieldDescriptionViolation(
                    file_path=self.file_path,
                    function_name=node.name,
                    parameter_name=param_name,
                    line_number=node.lineno,
                )
                self.violations.append(violation)

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

    def _has_field_description(self, arg: ast.arg) -> bool:
        """Check if argument has Field annotation with description."""
        if not arg.annotation:
            return False

        # Check if this is an Annotated type
        if not self._is_annotated_type(arg.annotation):
            return False

        # Look for Field with description in the annotation
        return self._has_field_with_description(arg.annotation)

    def _is_annotated_type(self, annotation: ast.AST) -> bool:
        """Check if annotation is Annotated[...]."""
        return (isinstance(annotation, ast.Subscript) and
                isinstance(annotation.value, ast.Name) and
                annotation.value.id == "Annotated")

    def _has_field_with_description(self, annotation: ast.AST) -> bool:
        """Check if Annotated type has Field with description."""
        if not isinstance(annotation, ast.Subscript):
            return False

        # Get the slice (the contents inside the brackets)
        if isinstance(annotation.slice, ast.Tuple):
            # Handle Annotated[Type, Field(...), ...]
            for elt in annotation.slice.elts:
                if self._is_field_call_with_description(elt):
                    return True
        else:
            # Handle single annotation Annotated[Type, Field(...)]
            return self._is_field_call_with_description(annotation.slice)

        return False

    def _is_field_call_with_description(self, node: ast.AST) -> bool:
        """Check if node is Field(...) call with description."""
        if not isinstance(node, ast.Call):
            return False

        # Check if it's a Field call
        if isinstance(node.func, ast.Name) and node.func.id == "Field":
            # Look for description in keywords
            for keyword in node.keywords:
                if keyword.arg == "description":
                    return True

        return False


def scan_file(file_path: Path) -> list[FieldDescriptionViolation]:
    """Scan a single Python file for Field description violations.

    Returns:
        List of violations found in the file
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file using AST
        tree = ast.parse(content)

        # Check for Field description violations
        checker = MCPFieldChecker(str(file_path))
        checker.visit(tree)

        return checker.violations

    except (SyntaxError, UnicodeDecodeError, OSError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def scan_directory(src_dir: Path) -> list[FieldDescriptionViolation]:
    """Scan all Python files in directory for Field description violations.

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


def scan_paths(paths: list[str]) -> list[FieldDescriptionViolation]:
    """Scan specified paths (files or directories) for Field description violations.

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


def format_violations_report(violations: list[FieldDescriptionViolation]) -> str:
    """Format violations into a readable report.

    Returns:
        Formatted report string
    """
    if not violations:
        return "✅ All MCP tool parameters have Field descriptions - Documentation standards met!"

    report_lines = [
        f"❌ Found {len(violations)} parameter(s) without Field descriptions",
        "",
        "All MCP tool parameters should have comprehensive Field descriptions",
        "for proper documentation and client integration.",
        "",
        "Violations found:",
        "",
    ]

    # Group by file for cleaner output
    by_file: dict[str, list[FieldDescriptionViolation]] = {}
    for violation in violations:
        if violation.file_path not in by_file:
            by_file[violation.file_path] = []
        by_file[violation.file_path].append(violation)

    for file_path, file_violations in sorted(by_file.items()):
        report_lines.append(f"📁 {file_path}")
        for violation in file_violations:
            report_lines.append(
                f"   └─ {violation.function_name}() parameter '{violation.parameter_name}' "
                f"at line {violation.line_number}"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "To fix these violations:",
            "1. Add Field() annotation with description to each parameter",
            '2. Use Annotated[Type, Field(description="...")]',
            "3. Provide clear, helpful parameter descriptions",
            "",
            "Example of proper MCP tool parameter documentation:",
            "def my_tool(",
            '    ctx: Annotated[Context, Field(description="FastMCP context for session access")],',
            '    data: Annotated[str, Field(description="Input data to process")],',
            '    option: Annotated[bool, Field(description="Enable special processing mode")] = False',
            ") -> Result:",
            '    """Process data with specified options."""',
            "    # Implementation...",
            "",
            "Note: The 'ctx: Context' parameter is excluded from this requirement.",
        ]
    )

    return "\n".join(report_lines)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Check that all MCP tool parameters have Field descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Check default src/ directory
    %(prog)s file1.py file2.py            # Check specific files
    %(prog)s src/databeak/servers/        # Check specific directory
    %(prog)s src/databeak/servers/ tests/ # Check multiple directories

All MCP tool parameters should have comprehensive Field descriptions
for proper documentation and client integration.
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
                print(f"❌ Found {len(violations)} parameter(s) without Field descriptions")
                return 1
            print(
                "✅ All MCP tool parameters have Field descriptions - Documentation standards met!"
            )
            return 0
        report = format_violations_report(violations)
        print(report)
        return 1 if violations else 0

    except Exception as e:
        print(f"Error during scanning: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
