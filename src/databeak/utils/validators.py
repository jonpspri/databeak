"""Validation utilities for CSV Editor."""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


def validate_file_path(file_path: str, must_exist: bool = True) -> tuple[bool, str]:
    """Validate a file path for security and existence."""
    try:
        # Convert to Path object
        path = Path(file_path).resolve()

        # Security: Check for path traversal attempts
        if ".." in file_path or file_path.startswith("~"):
            return False, "Path traversal not allowed"

        # Check file existence if required
        if must_exist and not path.exists():
            return False, f"File not found: {file_path}"

        # Check if it's a file (not directory)
        if must_exist and not path.is_file():
            return False, f"Not a file: {file_path}"

        # Check file extension
        valid_extensions = [".csv", ".tsv", ".txt", ".dat"]
        if path.suffix.lower() not in valid_extensions:
            return False, f"Invalid file extension. Supported: {valid_extensions}"

        # Check file size (max 1GB)
        if must_exist:
            max_size = 1024 * 1024 * 1024  # 1GB
            if path.stat().st_size > max_size:
                return False, "File too large. Maximum size: 1GB"

        return True, str(path)

    except Exception as e:
        return False, f"Error validating path: {e!s}"


def validate_url(url: str) -> tuple[bool, str]:
    """Validate a URL for CSV download with security checks against private networks."""
    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            return False, "Only HTTP/HTTPS URLs are supported"

        # Check if URL is valid
        if not parsed.netloc:
            return False, "Invalid URL format"

        # Extract hostname (remove port if present)
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid hostname in URL"

        # Check for private/local network addresses
        try:
            # Try to parse as IP address
            ip = ipaddress.ip_address(hostname)

            # Block private networks
            if ip.is_private:
                return False, "Private network addresses not allowed"
            if ip.is_loopback:
                return False, "Loopback addresses not allowed"
            if ip.is_link_local:
                return False, "Link-local addresses not allowed"
            if ip.is_multicast:
                return False, "Multicast addresses not allowed"

        except ValueError:
            # Not an IP address - check for localhost/private hostnames
            if hostname.lower() in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]:  # nosec B104
                return False, "Local addresses not allowed"

            # Try to resolve hostname to check for private IPs
            try:
                # Get IP addresses for hostname
                addr_info = socket.getaddrinfo(
                    hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
                for _, _, _, _, sockaddr in addr_info:
                    ip_addr = sockaddr[0]
                    try:
                        ip = ipaddress.ip_address(ip_addr)
                        if ip.is_private or ip.is_loopback or ip.is_link_local:
                            return False, f"Hostname resolves to private address: {ip_addr}"
                    except ValueError:
                        # IPv6 addresses with scope might not parse cleanly - be conservative
                        if ":" in ip_addr and ("fe80" in ip_addr.lower() or "::1" in ip_addr):
                            return False, f"Hostname resolves to local address: {ip_addr}"
            except (socket.gaierror, OSError):
                # DNS resolution failed - allow but log warning
                pass

        return True, url

    except Exception as e:
        return False, f"Invalid URL: {e!s}"


def validate_column_name(column_name: str) -> tuple[bool, str]:
    """Validate a column name."""
    if not column_name or not isinstance(column_name, str):
        return False, "Column name must be a non-empty string"

    # Check for invalid characters
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", column_name):
        return True, column_name
    else:
        return (
            False,
            "Column name must start with letter/underscore and contain only letters, numbers, underscores",
        )


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a DataFrame for common issues."""
    issues: dict[str, Any] = {"errors": [], "warnings": [], "info": {}}

    # Check if empty
    if df.empty:
        issues["errors"].append("DataFrame is empty")
        return issues

    # Check shape
    issues["info"]["shape"] = df.shape
    issues["info"]["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Check for duplicate columns
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        issues["errors"].append(f"Duplicate column names: {dupes}")

    # Check for completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues["warnings"].append(f"Completely null columns: {null_cols}")

    # Check for mixed types in columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Try to infer if it's mixed types
            unique_types = df[col].dropna().apply(lambda x: type(x).__name__).unique()
            if len(unique_types) > 1:
                issues["warnings"].append(f"Column '{col}' has mixed types: {list(unique_types)}")

    # Check for high cardinality in string columns
    for col in df.select_dtypes(include=["object"]).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            issues["info"][f"{col}_high_cardinality"] = True

    # Check for potential datetime columns
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(100)
        if sample.empty:
            continue
        try:
            pd.to_datetime(sample, errors="raise")
            issues["info"][f"{col}_potential_datetime"] = True
        except (ValueError, TypeError):
            pass

    return issues


def validate_expression(expression: str, allowed_vars: list[str]) -> tuple[bool, str]:
    """Validate a calculation expression for safety."""
    # Remove whitespace
    expr = expression.replace(" ", "")

    # Check for dangerous operations
    dangerous_patterns = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
    ]

    for pattern in dangerous_patterns:
        if pattern in expr.lower():
            return False, f"Dangerous operation '{pattern}' not allowed"

    # Check if only allowed variables and safe operations are used
    # This is a simplified check - in production use ast module for proper parsing
    set("0123456789+-*/().,<>=! ")
    safe_functions = {"abs", "min", "max", "sum", "len", "round", "int", "float", "str"}

    # Extract potential variable/function names
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)

    for token in tokens:
        if token not in allowed_vars and token not in safe_functions:
            return False, f"Unknown variable or function: {token}"

    return True, expression


def validate_sql_query(query: str) -> tuple[bool, str]:
    """Validate SQL query for safety (basic check)."""
    query_lower = query.lower()

    # Only allow SELECT queries
    if not query_lower.strip().startswith("select"):
        return False, "Only SELECT queries are allowed"

    # Check for dangerous keywords
    dangerous = [
        "drop",
        "delete",
        "insert",
        "update",
        "alter",
        "create",
        "exec",
        "execute",
    ]
    for keyword in dangerous:
        if keyword in query_lower:
            return False, f"Dangerous operation '{keyword}' not allowed"

    return True, query


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe file operations."""
    # Remove path components
    filename = Path(filename).name

    # Remove/replace invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Limit length
    path_obj = Path(filename)
    name, ext = path_obj.stem, path_obj.suffix
    if len(name) > 100:
        name = name[:100]

    return name + ext


def convert_pandas_na_to_none(value: Any) -> Any:
    """Convert pandas NA values to Python None for Pydantic serialization.

    This function handles the conversion of pandas nullable data types' NA values
    to Python None, which is compatible with Pydantic models.

    Args:
        value: Any value that might be a pandas NA

    Returns:
        The original value if not pandas NA, otherwise None
    """
    import pandas as pd

    # Handle pandas NA values (from nullable dtypes)
    if pd.isna(value):
        return None
    return value


def convert_pandas_na_list(values: list[Any]) -> list[Any]:
    """Convert a list of values that may contain pandas NA values to Python None.

    Args:
        values: List of values that may contain pandas NA

    Returns:
        List with pandas NA values converted to None
    """
    return [convert_pandas_na_to_none(val) for val in values]
