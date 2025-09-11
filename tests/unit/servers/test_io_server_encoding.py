"""Tests specifically for encoding handling in io_server to reach 80% coverage."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.io_server import (
    detect_file_encoding,
    load_csv,
    load_csv_from_url,
)


class TestFileEncodingDetection:
    """Test file encoding detection."""

    @patch("chardet.detect")
    def test_detect_encoding_high_confidence(self, mock_detect):
        """Test encoding detection with high confidence."""
        mock_detect.return_value = {"encoding": "UTF-8", "confidence": 0.95}

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            encoding = detect_file_encoding(temp_path)
            assert encoding == "utf-8"
            mock_detect.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("chardet.detect")
    def test_detect_encoding_low_confidence(self, mock_detect):
        """Test encoding detection with low confidence fallback."""
        mock_detect.return_value = {"encoding": "ISO-8859-1", "confidence": 0.3}

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            # Write UTF-8 BOM to make pandas detection work
            f.write(b"\xef\xbb\xbftest,data\n1,2")
            temp_path = f.name

        try:
            encoding = detect_file_encoding(temp_path)
            # Should fall back to pandas detection
            assert encoding in ["utf-8", "utf-8-sig"]
        finally:
            Path(temp_path).unlink()

    @patch("chardet.detect")
    def test_detect_encoding_none_result(self, mock_detect):
        """Test encoding detection when chardet returns None."""
        mock_detect.return_value = {"encoding": None, "confidence": 0}

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            f.write("test,data\n1,2")
            temp_path = f.name

        try:
            encoding = detect_file_encoding(temp_path)
            # Should fall back to pandas detection
            assert encoding == "utf-8"
        finally:
            Path(temp_path).unlink()


class TestLoadCsvEncodingFallbacks:
    """Test CSV loading with encoding fallbacks."""

    async def test_load_csv_with_context_reporting(self):
        """Test load_csv with context for progress reporting."""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4")
            temp_path = f.name

        try:
            # Mock context
            mock_ctx = MagicMock()
            mock_ctx.info = AsyncMock(return_value=None)
            mock_ctx.report_progress = AsyncMock(return_value=None)

            result = await load_csv(file_path=temp_path, ctx=mock_ctx)

            assert result.rows_affected == 2
            # Progress should be reported
            mock_ctx.report_progress.assert_called()
            mock_ctx.info.assert_called()
        finally:
            Path(temp_path).unlink()

    @patch("pandas.read_csv")
    async def test_load_csv_all_encodings_fail(self, mock_read_csv):
        """Test when all encoding attempts fail."""
        # Make all read attempts fail
        mock_read_csv.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"test data")
            temp_path = f.name

        try:
            with pytest.raises(ToolError, match="Failed to load CSV with any encoding"):
                await load_csv(file_path=temp_path, encoding="utf-8")
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_memory_check_on_fallback(self):
        """Test memory limit check during encoding fallback."""
        # Create a file that will fail with first encoding
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            # Write Latin-1 specific characters
            f.write(b"name,value\n")
            f.write("José,100\n".encode("latin-1"))
            temp_path = f.name

        try:
            # Mock to make memory check fail
            with patch("src.databeak.servers.io_server.MAX_MEMORY_USAGE_MB", 0.0001), pytest.raises(ToolError, match="memory limit"):
                await load_csv(file_path=temp_path, encoding="utf-8")
        finally:
            Path(temp_path).unlink()

    async def test_load_csv_row_limit_on_fallback(self):
        """Test row limit check during encoding fallback."""
        # Create a file with many rows
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="latin-1", suffix=".csv", delete=False
        ) as f:
            f.write("col1,col2\n")
            for i in range(10):
                f.write(f"{i},value{i}\n")
            temp_path = f.name

        try:
            # Mock MAX_ROWS to trigger limit
            with patch("src.databeak.servers.io_server.MAX_ROWS", 5), pytest.raises(ToolError, match="rows exceeds limit"):
                await load_csv(
                    file_path=temp_path, encoding="ascii"
                )  # Wrong encoding to trigger fallback
        finally:
            Path(temp_path).unlink()


class TestLoadCsvFromUrlFallbacks:
    """Test URL loading with encoding fallbacks."""

    @patch("pandas.read_csv")
    async def test_load_url_encoding_fallback_success(self, mock_read_csv):
        """Test URL loading with successful encoding fallback."""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # First call fails with encoding error, second succeeds
        mock_read_csv.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), mock_df]

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock(return_value=None)
        mock_ctx.report_progress = AsyncMock(return_value=None)

        result = await load_csv_from_url(
            url="http://example.com/data.csv", encoding="utf-8", ctx=mock_ctx
        )

        assert result.rows_affected == 2
        assert mock_read_csv.call_count == 2
        mock_ctx.info.assert_called()

    @patch("pandas.read_csv")
    async def test_load_url_memory_check_fallback(self, mock_read_csv):
        """Test URL loading with memory check during fallback."""
        # Create large dataframe
        large_df = pd.DataFrame(
            {
                "col1": range(100),
                "col2": ["x" * 10000] * 100,  # Large strings
            }
        )

        # First fails, second returns large df
        mock_read_csv.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), large_df]

        with patch("src.databeak.servers.io_server.MAX_MEMORY_USAGE_MB", 0.001), pytest.raises(ToolError, match="memory limit"):
            await load_csv_from_url(url="http://example.com/data.csv", encoding="utf-8")

    @patch("pandas.read_csv")
    async def test_load_url_row_limit_fallback(self, mock_read_csv):
        """Test URL loading with row limit during fallback."""
        # Create dataframe with many rows
        large_df = pd.DataFrame({"col1": range(1000), "col2": range(1000)})

        # First fails, second returns large df
        mock_read_csv.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), large_df]

        with patch("src.databeak.servers.io_server.MAX_ROWS", 100), pytest.raises(ToolError, match="rows exceeds limit"):
            await load_csv_from_url(url="http://example.com/data.csv", encoding="utf-8")

    @patch("pandas.read_csv")
    async def test_load_url_all_encodings_fail(self, mock_read_csv):
        """Test URL loading when all encodings fail."""
        # All attempts fail
        mock_read_csv.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")

        with pytest.raises(ToolError, match="Encoding error with all attempted encodings"):
            await load_csv_from_url(url="http://example.com/data.csv", encoding="utf-8")

    @patch("pandas.read_csv")
    async def test_load_url_other_exception_during_fallback(self, mock_read_csv):
        """Test URL loading with non-encoding exception during fallback."""
        # First encoding error, then different error
        mock_read_csv.side_effect = [
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
            ValueError("Different error"),
            pd.DataFrame({"col": [1]}),  # Eventually succeeds
        ]

        result = await load_csv_from_url(url="http://example.com/data.csv", encoding="utf-8")

        assert result.rows_affected == 1
        assert mock_read_csv.call_count == 3
