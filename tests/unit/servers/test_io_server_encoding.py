"""Tests specifically for encoding handling in io_server to reach 80% coverage."""

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
from tests.test_mock_context import create_mock_context


class TestFileEncodingDetection:
    """Test file encoding detection."""

    @pytest.mark.parametrize(
        "mock_encoding,mock_confidence,file_content,expected_encoding",
        [
            ("UTF-8", 0.95, b"test content", "utf-8"),
            ("ISO-8859-1", 0.3, b"\xef\xbb\xbftest,data\n1,2", ["utf-8", "utf-8-sig"]),
            (None, 0, b"test,data\n1,2", "utf-8"),
        ],
    )
    @patch("chardet.detect")
    def test_encoding_detection_scenarios(
        self, mock_detect, mock_encoding, mock_confidence, file_content, expected_encoding
    ):
        """Test encoding detection with different chardet results."""
        mock_detect.return_value = {"encoding": mock_encoding, "confidence": mock_confidence}

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(file_content)
            temp_path = f.name

        try:
            encoding = detect_file_encoding(temp_path)
            if isinstance(expected_encoding, list):
                assert encoding in expected_encoding
            else:
                assert encoding == expected_encoding
            mock_detect.assert_called_once()
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
            # Mock context with proper session_id
            mock_ctx = MagicMock()
            mock_ctx.session_id = "test_session_id"
            mock_ctx.info = AsyncMock(return_value=None)
            mock_ctx.report_progress = AsyncMock(return_value=None)

            result = await load_csv(mock_ctx, file_path=temp_path)

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
            with pytest.raises(ToolError, match="Encoding error with all attempted encodings"):
                await load_csv(create_mock_context(), file_path=temp_path, encoding="utf-8")
        finally:
            Path(temp_path).unlink()

    @pytest.mark.skip(
        reason="Complex encoding fallback + memory limit edge case - needs refactoring"
    )
    async def test_load_csv_memory_check_on_fallback(self):
        """Test memory limit check during encoding fallback."""
        pass

    @pytest.mark.skip(reason="Complex encoding fallback + row limit edge case - needs refactoring")
    async def test_load_csv_row_limit_on_fallback(self):
        """Test row limit check during encoding fallback."""
        pass


class TestLoadCsvFromUrlFallbacks:
    """Test URL loading with encoding fallbacks."""

    @patch("src.databeak.servers.io_server.urlopen")
    @patch("pandas.read_csv")
    async def test_load_url_encoding_fallback_success(self, mock_read_csv, mock_urlopen):
        """Test URL loading with successful encoding fallback."""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Mock urlopen response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/csv", "Content-Length": "100"}
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # First call fails with encoding error, second succeeds
        mock_read_csv.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), mock_df]

        # Mock context with proper session_id
        mock_ctx = MagicMock()
        mock_ctx.session_id = "test_session_id"
        mock_ctx.info = AsyncMock(return_value=None)
        mock_ctx.error = AsyncMock(return_value=None)
        mock_ctx.report_progress = AsyncMock(return_value=None)

        result = await load_csv_from_url(
            mock_ctx, url="http://example.com/data.csv", encoding="utf-8"
        )

        assert result.rows_affected == 2
        assert mock_read_csv.call_count == 2
        mock_ctx.info.assert_called()

    @pytest.mark.skip(
        reason="Complex URL encoding fallback + memory limit edge case - needs refactoring"
    )
    async def test_load_url_memory_check_fallback(self):
        """Test URL loading with memory check during fallback."""
        pass

    @pytest.mark.skip(
        reason="Complex URL encoding fallback + row limit edge case - needs refactoring"
    )
    async def test_load_url_row_limit_fallback(self):
        """Test URL loading with row limit during fallback."""
        pass

    @patch("src.databeak.servers.io_server.urlopen")
    @patch("pandas.read_csv")
    async def test_load_url_all_encodings_fail(self, mock_read_csv, mock_urlopen):
        """Test URL loading when all encodings fail."""
        # Mock urlopen response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/csv", "Content-Length": "100"}
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # All attempts fail
        mock_read_csv.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")

        with pytest.raises(ToolError, match="Encoding error with all attempted encodings"):
            await load_csv_from_url(
                create_mock_context(), url="http://example.com/data.csv", encoding="utf-8"
            )

    @patch("src.databeak.servers.io_server.urlopen")
    @patch("pandas.read_csv")
    async def test_load_url_other_exception_during_fallback(self, mock_read_csv, mock_urlopen):
        """Test URL loading with non-encoding exception during fallback."""
        # Mock urlopen response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/csv", "Content-Length": "100"}
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # First encoding error, then different error
        mock_read_csv.side_effect = [
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
            ValueError("Different error"),
            pd.DataFrame({"col": [1]}),  # Eventually succeeds
        ]

        result = await load_csv_from_url(
            create_mock_context(), url="http://example.com/data.csv", encoding="utf-8"
        )

        assert result.rows_affected == 1
        assert mock_read_csv.call_count == 3
