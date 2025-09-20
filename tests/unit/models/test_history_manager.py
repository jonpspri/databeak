"""Unit tests for HistoryManager module."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.databeak.models.history_manager import (
    HistoryManager,
    HistoryStorage,
    OperationHistory,
)


class TestOperationHistory:
    """Test the OperationHistory class."""

    def test_operation_history_init(self):
        """Test OperationHistory initialization."""
        timestamp = datetime.now(UTC)
        details = {"operation": "filter", "condition": "age > 25"}
        metadata = {"user": "test_user"}

        op_history = OperationHistory(
            operation_id="test_op_123",
            operation_type="filter",
            timestamp=timestamp,
            details=details,
            metadata=metadata,
        )

        assert op_history.operation_id == "test_op_123"
        assert op_history.operation_type == "filter"
        assert op_history.timestamp == timestamp
        assert op_history.details == details
        assert op_history.metadata == metadata
        assert op_history.data_snapshot is None

    def test_operation_history_init_with_snapshot(self):
        """Test OperationHistory initialization with data snapshot."""
        timestamp = datetime.now(UTC)
        details = {"operation": "filter"}
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        op_history = OperationHistory(
            operation_id="test_op_123",
            operation_type="filter",
            timestamp=timestamp,
            details=details,
            data_snapshot=df,
        )

        assert op_history.data_snapshot is not None
        pd.testing.assert_frame_equal(op_history.data_snapshot, df)

    def test_operation_history_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now(UTC)
        details = {"operation": "filter"}
        metadata = {"user": "test_user"}

        op_history = OperationHistory(
            operation_id="test_op_123",
            operation_type="filter",
            timestamp=timestamp,
            details=details,
            metadata=metadata,
        )

        result = op_history.to_dict()

        assert result["operation_id"] == "test_op_123"
        assert result["operation_type"] == "filter"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["details"] == details
        assert result["metadata"] == metadata
        assert result["has_snapshot"] is False

    def test_operation_history_to_dict_with_snapshot(self):
        """Test conversion to dictionary with snapshot."""
        timestamp = datetime.now(UTC)
        details = {"operation": "filter"}
        df = pd.DataFrame({"col1": [1, 2, 3]})

        op_history = OperationHistory(
            operation_id="test_op_123",
            operation_type="filter",
            timestamp=timestamp,
            details=details,
            data_snapshot=df,
        )

        result = op_history.to_dict()
        assert result["has_snapshot"] is True

    def test_operation_history_from_dict(self):
        """Test creation from dictionary."""
        timestamp = datetime.now(UTC)
        data = {
            "operation_id": "test_op_123",
            "operation_type": "filter",
            "timestamp": timestamp.isoformat(),
            "details": {"operation": "filter"},
            "metadata": {"user": "test_user"},
        }

        op_history = OperationHistory.from_dict(data)

        assert op_history.operation_id == "test_op_123"
        assert op_history.operation_type == "filter"
        assert op_history.timestamp == timestamp
        assert op_history.details == {"operation": "filter"}
        assert op_history.metadata == {"user": "test_user"}
        assert op_history.data_snapshot is None

    def test_operation_history_from_dict_with_snapshot(self):
        """Test creation from dictionary with snapshot."""
        timestamp = datetime.now(UTC)
        df = pd.DataFrame({"col1": [1, 2, 3]})
        data = {
            "operation_id": "test_op_123",
            "operation_type": "filter",
            "timestamp": timestamp.isoformat(),
            "details": {"operation": "filter"},
        }

        op_history = OperationHistory.from_dict(data, data_snapshot=df)

        assert op_history.data_snapshot is not None
        pd.testing.assert_frame_equal(op_history.data_snapshot, df)

    def test_operation_history_from_dict_no_metadata(self):
        """Test creation from dictionary without metadata."""
        timestamp = datetime.now(UTC)
        data = {
            "operation_id": "test_op_123",
            "operation_type": "filter",
            "timestamp": timestamp.isoformat(),
            "details": {"operation": "filter"},
        }

        op_history = OperationHistory.from_dict(data)
        assert op_history.metadata == {}


class TestHistoryManagerInit:
    """Test HistoryManager initialization."""

    def test_init_memory_storage(self):
        """Test initialization with memory storage."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

        assert manager.session_id == "test_session_123"
        assert manager.storage_type == HistoryStorage.MEMORY
        assert manager.max_history == 100
        assert manager.enable_snapshots is True
        assert manager.snapshot_interval == 5
        assert manager.history == []
        assert manager.current_index == -1
        assert manager.redo_stack == []

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            max_history=50,
            enable_snapshots=False,
            snapshot_interval=10,
        )

        assert manager.max_history == 50
        assert manager.enable_snapshots is False
        assert manager.snapshot_interval == 10

    def test_init_history_dir_default(self):
        """Test initialization with default history directory."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

        expected_dir = str(Path.cwd() / ".csv_history")
        assert manager.history_dir == expected_dir

    def test_init_history_dir_custom(self):
        """Test initialization with custom history directory."""
        custom_dir = "/custom/history/path"
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            history_dir=custom_dir,
        )

        assert manager.history_dir == custom_dir

    @patch("pathlib.Path.mkdir")
    @patch("src.databeak.models.history_manager.HistoryManager._load_history")
    def test_init_json_storage_creates_dir(self, mock_load, mock_mkdir):
        """Test initialization with JSON storage creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_load.assert_called_once()
            assert manager.storage_type == HistoryStorage.JSON

    @patch("pathlib.Path.mkdir")
    @patch("src.databeak.models.history_manager.HistoryManager._load_history")
    def test_init_pickle_storage_creates_dir(self, mock_load, mock_mkdir):
        """Test initialization with pickle storage creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.PICKLE,
                history_dir=temp_dir,
            )

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_load.assert_called_once()
            assert manager.storage_type == HistoryStorage.PICKLE


class TestHistoryManagerPathMethods:
    """Test HistoryManager path generation methods."""

    def test_get_history_file_path_default_extension(self):
        """Test history file path with default extension."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            history_dir="/test/dir",
        )

        path = manager._get_history_file_path()
        expected = "/test/dir/history_test_session_123.json"
        assert path == expected

    def test_get_history_file_path_custom_extension(self):
        """Test history file path with custom extension."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            history_dir="/test/dir",
        )

        path = manager._get_history_file_path("pkl")
        expected = "/test/dir/history_test_session_123.pkl"
        assert path == expected

    @patch("pathlib.Path.mkdir")
    def test_get_snapshot_file_path(self, mock_mkdir):
        """Test snapshot file path generation."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            history_dir="/test/dir",
        )

        path = manager._get_snapshot_file_path("operation_123")
        expected = "/test/dir/snapshots/test_session_123/snapshot_operation_123.pkl"
        assert path == expected

        # Verify snapshot directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestHistoryManagerPersistenceJSON:
    """Test HistoryManager JSON persistence operations."""

    @pytest.fixture
    def temp_manager(self):
        """Create a temporary HistoryManager with JSON storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )
            yield manager, temp_dir

    def test_load_history_json_no_file(self, temp_manager):
        """Test loading history when no file exists."""
        manager, _temp_dir = temp_manager

        # Directly call _load_history (it's called in __init__)
        manager._load_history()

        assert manager.history == []
        assert manager.current_index == -1

    def test_save_and_load_history_json(self, temp_manager):
        """Test saving and loading history with JSON storage."""
        manager, temp_dir = temp_manager

        # Add operations to history
        operation1_id = manager.add_operation(
            operation_type="filter",
            details={"column": "age", "condition": "> 25"},
            metadata={"user": "test_user"},
        )

        operation2_id = manager.add_operation(
            operation_type="sort",
            details={"column": "name", "ascending": True},
        )

        # Create a new manager to test loading
        new_manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.JSON,
            history_dir=temp_dir,
        )

        assert len(new_manager.history) == 2
        assert new_manager.current_index == 1

        # Verify operation details
        assert new_manager.history[0].operation_id == operation1_id
        assert new_manager.history[0].operation_type == "filter"
        assert new_manager.history[0].details == {"column": "age", "condition": "> 25"}
        assert new_manager.history[0].metadata == {"user": "test_user"}

        assert new_manager.history[1].operation_id == operation2_id
        assert new_manager.history[1].operation_type == "sort"
        assert new_manager.history[1].details == {"column": "name", "ascending": True}

    def test_save_and_load_history_with_snapshots(self, temp_manager):
        """Test saving and loading history with data snapshots."""
        manager, temp_dir = temp_manager

        # Create test data
        df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df2 = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})

        # Add operations with snapshots - first operation always gets snapshot
        manager.add_operation(
            operation_type="filter",
            details={"column": "col1"},
            current_data=df1,
        )

        # Force second operation to also get snapshot by adding 4 more operations
        # to reach snapshot interval (every 5th operation)
        for i in range(4):
            manager.add_operation(f"dummy_op_{i}", {})

        manager.add_operation(
            operation_type="sort",
            details={"column": "col2"},
            current_data=df2,
        )

        # Create a new manager to test loading
        new_manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.JSON,
            history_dir=temp_dir,
        )

        assert len(new_manager.history) == 6  # 1 + 4 dummy + 1 = 6

        # Verify snapshots were loaded correctly
        assert new_manager.history[0].data_snapshot is not None
        pd.testing.assert_frame_equal(new_manager.history[0].data_snapshot, df1)

        assert new_manager.history[5].data_snapshot is not None  # 6th operation (index 5)
        pd.testing.assert_frame_equal(new_manager.history[5].data_snapshot, df2)

    def test_load_history_json_corrupted_file(self, temp_manager):
        """Test loading history with corrupted JSON file."""
        manager, temp_dir = temp_manager

        # Create corrupted JSON file
        history_file = Path(temp_dir) / "history_test_session_123.json"
        history_file.write_text("{ invalid json")

        # Should handle error gracefully
        with patch("src.databeak.models.history_manager.logger.error") as mock_logger:
            manager._load_history()
            mock_logger.assert_called_once()

        assert manager.history == []
        assert manager.current_index == -1

    def test_load_history_json_missing_snapshot_file(self, temp_manager):
        """Test loading history when snapshot file is missing."""
        manager, temp_dir = temp_manager

        # Create history file with reference to missing snapshot
        history_data = {
            "session_id": "test_session_123",
            "history": [
                {
                    "operation_id": "op_123",
                    "operation_type": "filter",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "details": {"column": "age"},
                    "metadata": {},
                    "has_snapshot": True,
                },
            ],
            "current_index": 0,
        }

        history_file = Path(temp_dir) / "history_test_session_123.json"
        history_file.write_text(json.dumps(history_data))

        # Load should succeed but without snapshot
        manager._load_history()

        assert len(manager.history) == 1
        assert manager.history[0].data_snapshot is None

    @patch("src.databeak.models.history_manager.logger.error")
    def test_save_history_json_write_error(self, mock_logger, temp_manager):
        """Test error handling during JSON save operation."""
        manager, temp_dir = temp_manager

        # Make directory read-only to cause write error
        Path(temp_dir).chmod(0o444)

        try:
            manager._save_history()
            mock_logger.assert_called_once()
        finally:
            # Restore permissions for cleanup
            Path(temp_dir).chmod(0o755)


class TestHistoryManagerPersistencePickle:
    """Test HistoryManager pickle persistence operations."""

    @pytest.fixture
    def temp_manager(self):
        """Create a temporary HistoryManager with pickle storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.PICKLE,
                history_dir=temp_dir,
            )
            yield manager, temp_dir

    def test_save_and_load_history_pickle(self, temp_manager):
        """Test saving and loading history with pickle storage."""
        manager, temp_dir = temp_manager

        # Create test data
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # Add operation with snapshot
        operation_id = manager.add_operation(
            operation_type="filter",
            details={"column": "col1"},
            current_data=df,
            metadata={"user": "test_user"},
        )

        # Create a new manager to test loading
        new_manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.PICKLE,
            history_dir=temp_dir,
        )

        assert len(new_manager.history) == 1
        assert new_manager.current_index == 0

        # Verify operation details
        assert new_manager.history[0].operation_id == operation_id
        assert new_manager.history[0].operation_type == "filter"
        assert new_manager.history[0].details == {"column": "col1"}
        assert new_manager.history[0].metadata == {"user": "test_user"}

        # Verify snapshot
        assert new_manager.history[0].data_snapshot is not None
        pd.testing.assert_frame_equal(new_manager.history[0].data_snapshot, df)

    def test_load_history_pickle_no_file(self, temp_manager):
        """Test loading history when no pickle file exists."""
        manager, _temp_dir = temp_manager

        manager._load_history()

        assert manager.history == []
        assert manager.current_index == -1

    def test_load_history_pickle_corrupted_file(self, temp_manager):
        """Test loading history with corrupted pickle file."""
        manager, temp_dir = temp_manager

        # Create corrupted pickle file
        history_file = Path(temp_dir) / "history_test_session_123.pkl"
        history_file.write_bytes(b"corrupted pickle data")

        # Should handle error gracefully
        with patch("src.databeak.models.history_manager.logger.error") as mock_logger:
            manager._load_history()
            mock_logger.assert_called_once()

        assert manager.history == []
        assert manager.current_index == -1

    @patch("src.databeak.models.history_manager.logger.error")
    def test_save_history_pickle_write_error(self, mock_logger, temp_manager):
        """Test error handling during pickle save operation."""
        manager, temp_dir = temp_manager

        # Make directory read-only to cause write error
        Path(temp_dir).chmod(0o444)

        try:
            manager._save_history()
            mock_logger.assert_called_once()
        finally:
            # Restore permissions for cleanup
            Path(temp_dir).chmod(0o755)


class TestHistoryManagerOperations:
    """Test HistoryManager operation management."""

    @pytest.fixture
    def manager(self):
        """Create a HistoryManager for testing."""
        return HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

    def test_add_operation_basic(self, manager):
        """Test adding a basic operation."""
        operation_id = manager.add_operation(
            operation_type="filter",
            details={"column": "age", "condition": "> 25"},
        )

        assert len(manager.history) == 1
        assert manager.current_index == 0
        assert manager.redo_stack == []

        operation = manager.history[0]
        assert operation.operation_id == operation_id
        assert operation.operation_type == "filter"
        assert operation.details == {"column": "age", "condition": "> 25"}
        assert operation.metadata == {}
        assert operation.data_snapshot is None

    def test_add_operation_with_metadata(self, manager):
        """Test adding operation with metadata."""
        manager.add_operation(
            operation_type="sort",
            details={"column": "name"},
            metadata={"user": "test_user", "session": "web"},
        )

        operation = manager.history[0]
        assert operation.metadata == {"user": "test_user", "session": "web"}

    def test_add_operation_with_snapshot(self, manager):
        """Test adding operation with data snapshot."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        manager.add_operation(
            operation_type="transform",
            details={"operation": "multiply"},
            current_data=df,
        )

        operation = manager.history[0]
        assert operation.data_snapshot is not None
        pd.testing.assert_frame_equal(operation.data_snapshot, df)
        # Verify it's a copy, not the same object
        assert operation.data_snapshot is not df

    def test_add_operation_snapshot_interval(self, manager):
        """Test snapshot creation based on snapshot interval."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        # First operation should create snapshot (index 0)
        manager.add_operation("op1", {}, df)
        assert manager.history[0].data_snapshot is not None

        # Operations 1-4 should not create snapshots (not at interval)
        for i in range(1, 5):
            manager.add_operation(f"op{i + 1}", {}, df)
            assert manager.history[i].data_snapshot is None

        # Operation 5 should create snapshot (index 5, multiple of interval)
        manager.add_operation("op6", {}, df)
        assert manager.history[5].data_snapshot is not None

    def test_add_operation_snapshots_disabled(self):
        """Test adding operations with snapshots disabled."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            enable_snapshots=False,
        )

        df = pd.DataFrame({"col1": [1, 2, 3]})

        manager.add_operation("filter", {}, df)
        assert manager.history[0].data_snapshot is None

    def test_add_operation_clears_redo_stack(self, manager):
        """Test that adding operation clears redo stack."""
        # Add operations
        manager.add_operation("op1", {})
        manager.add_operation("op2", {})

        # Undo to populate redo stack
        manager.undo()
        assert len(manager.redo_stack) == 1

        # Add new operation should clear redo stack
        manager.add_operation("op3", {})
        assert manager.redo_stack == []

    def test_add_operation_removes_future_history(self, manager):
        """Test that adding operation removes future history after undo."""
        # Add three operations
        manager.add_operation("op1", {})
        manager.add_operation("op2", {})
        manager.add_operation("op3", {})

        # Undo twice
        manager.undo()
        manager.undo()
        assert manager.current_index == 0
        assert len(manager.history) == 3

        # Add new operation should remove future history
        manager.add_operation("new_op", {})
        assert len(manager.history) == 2  # op1 + new_op
        assert manager.current_index == 1

    def test_add_operation_max_history_trimming(self):
        """Test history trimming when max history is reached."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
            max_history=3,
        )

        # Add operations up to max
        for i in range(5):
            manager.add_operation(f"op{i}", {})

        # Should only keep last 3 operations
        assert len(manager.history) == 3
        assert manager.current_index == 2
        assert manager.history[0].operation_type == "op2"
        assert manager.history[1].operation_type == "op3"
        assert manager.history[2].operation_type == "op4"

    @patch("pathlib.Path.unlink")
    def test_add_operation_cleanup_old_snapshots(self, mock_unlink):
        """Test cleanup of old snapshot files when history is trimmed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
                max_history=2,
            )

            df = pd.DataFrame({"col1": [1, 2, 3]})

            # Add operations with snapshots
            manager.add_operation("op1", {}, df)  # Will create snapshot
            manager.add_operation("op2", {}, df)
            manager.add_operation("op3", {}, df)  # Should trigger cleanup

            # Verify old snapshot was cleaned up
            mock_unlink.assert_called()


class TestHistoryManagerUndoRedo:
    """Test HistoryManager undo/redo functionality."""

    @pytest.fixture
    def manager_with_history(self):
        """Create a HistoryManager with some history."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

        # Add some operations with snapshots
        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        df2 = pd.DataFrame({"col1": [4, 5, 6]})
        df3 = pd.DataFrame({"col1": [7, 8, 9]})

        manager.add_operation("op1", {}, df1)
        manager.add_operation("op2", {}, df2)
        manager.add_operation("op3", {}, df3)

        return manager

    def test_can_undo_empty_history(self):
        """Test can_undo with empty history."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)
        assert manager.can_undo() is False

    def test_can_undo_with_history(self, manager_with_history):
        """Test can_undo with operations in history."""
        assert manager_with_history.can_undo() is True

    def test_can_redo_no_undo(self, manager_with_history):
        """Test can_redo when no undo has been performed."""
        assert manager_with_history.can_redo() is False

    def test_can_redo_after_undo(self, manager_with_history):
        """Test can_redo after undo operation."""
        manager_with_history.undo()
        assert manager_with_history.can_redo() is True

    def test_undo_basic(self, manager_with_history):
        """Test basic undo operation."""
        initial_index = manager_with_history.current_index

        undone_op, snapshot = manager_with_history.undo()

        assert manager_with_history.current_index == initial_index - 1
        assert len(manager_with_history.redo_stack) == 1
        assert undone_op.operation_type == "op3"
        assert snapshot is not None  # Should return op1's snapshot (most recent before current)
        pd.testing.assert_frame_equal(snapshot, pd.DataFrame({"col1": [1, 2, 3]}))

    def test_undo_no_available_snapshot(self):
        """Test undo when no snapshot is available."""
        manager = HistoryManager("test", HistoryStorage.MEMORY, enable_snapshots=False)

        # Add operation without snapshot
        manager.add_operation("op1", {})

        undone_op, snapshot = manager.undo()

        assert undone_op is not None
        assert undone_op.operation_type == "op1"
        assert snapshot is None

    def test_undo_cannot_undo(self):
        """Test undo when no operations to undo."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        undone_op, snapshot = manager.undo()

        assert undone_op is None
        assert snapshot is None

    def test_undo_multiple_times(self, manager_with_history):
        """Test multiple undo operations."""
        # Undo first operation
        undone_op1, _ = manager_with_history.undo()
        assert undone_op1.operation_type == "op3"
        assert manager_with_history.current_index == 1

        # Undo second operation
        undone_op2, _ = manager_with_history.undo()
        assert undone_op2.operation_type == "op2"
        assert manager_with_history.current_index == 0

        # Undo third operation
        undone_op3, _ = manager_with_history.undo()
        assert undone_op3.operation_type == "op1"
        assert manager_with_history.current_index == -1

    def test_redo_basic(self, manager_with_history):
        """Test basic redo operation."""
        # First undo an operation
        manager_with_history.undo()
        initial_index = manager_with_history.current_index

        redone_op, _snapshot = manager_with_history.redo()

        assert manager_with_history.current_index == initial_index + 1
        assert len(manager_with_history.redo_stack) == 0
        assert redone_op.operation_type == "op3"
        # Snapshot may or may not be available depending on snapshot interval
        # Don't assert on snapshot availability in this test

    def test_redo_cannot_redo(self):
        """Test redo when no operations to redo."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        redone_op, snapshot = manager.redo()

        assert redone_op is None
        assert snapshot is None

    def test_redo_multiple_times(self, manager_with_history):
        """Test multiple redo operations."""
        # Undo all operations
        manager_with_history.undo()
        manager_with_history.undo()
        manager_with_history.undo()
        assert manager_with_history.current_index == -1

        # Redo operations back
        redone_op1, _ = manager_with_history.redo()
        assert redone_op1.operation_type == "op1"
        assert manager_with_history.current_index == 0

        redone_op2, _ = manager_with_history.redo()
        assert redone_op2.operation_type == "op2"
        assert manager_with_history.current_index == 1

        redone_op3, _ = manager_with_history.redo()
        assert redone_op3.operation_type == "op3"
        assert manager_with_history.current_index == 2

    def test_undo_redo_cycle(self, manager_with_history):
        """Test undo followed by redo returns to original state."""
        initial_index = manager_with_history.current_index

        # Undo and then redo
        manager_with_history.undo()
        manager_with_history.redo()

        assert manager_with_history.current_index == initial_index
        assert manager_with_history.can_undo() is True
        assert manager_with_history.can_redo() is False

    @patch("src.databeak.models.history_manager.HistoryManager._save_history")
    def test_undo_saves_state_persistent_storage(self, mock_save):
        """Test that undo saves state for persistent storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            manager.add_operation("op1", {})
            mock_save.reset_mock()

            manager.undo()
            mock_save.assert_called_once()

    @patch("src.databeak.models.history_manager.HistoryManager._save_history")
    def test_redo_saves_state_persistent_storage(self, mock_save):
        """Test that redo saves state for persistent storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            manager.add_operation("op1", {})
            manager.undo()
            mock_save.reset_mock()

            manager.redo()
            mock_save.assert_called_once()


class TestHistoryManagerQueries:
    """Test HistoryManager query methods."""

    @pytest.fixture
    def manager_with_history(self):
        """Create a HistoryManager with test history."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

        df = pd.DataFrame({"col1": [1, 2, 3]})

        manager.add_operation("filter", {"column": "age"}, df)
        manager.add_operation("sort", {"column": "name"})
        manager.add_operation("group", {"column": "dept"}, df)

        return manager

    def test_get_history_all(self, manager_with_history):
        """Test getting all history."""
        history = manager_with_history.get_history()

        assert len(history) == 3

        # Check first entry
        assert history[0]["operation_type"] == "filter"
        assert history[0]["index"] == 0
        assert history[0]["is_current"] is False
        assert history[0]["can_restore"] is True

        # Check current entry - group operation at index 2 won't have snapshot
        # because snapshots are only taken at first operation and every 5th
        assert history[2]["operation_type"] == "group"
        assert history[2]["index"] == 2
        assert history[2]["is_current"] is True
        assert history[2]["can_restore"] is False  # No snapshot for this operation

        # Check entry without snapshot (index 1 - sort operation)
        assert history[1]["can_restore"] is False

    def test_get_history_limited(self, manager_with_history):
        """Test getting limited history."""
        history = manager_with_history.get_history(limit=2)

        assert len(history) == 2
        assert history[0]["operation_type"] == "sort"
        assert history[0]["index"] == 1
        assert history[1]["operation_type"] == "group"
        assert history[1]["index"] == 2

    def test_get_history_limit_larger_than_history(self, manager_with_history):
        """Test getting history with limit larger than available."""
        history = manager_with_history.get_history(limit=10)

        assert len(history) == 3  # Should return all available

    def test_get_history_empty(self):
        """Test getting history when empty."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        history = manager.get_history()
        assert history == []

    def test_get_operation_found(self, manager_with_history):
        """Test getting existing operation by ID."""
        operation_id = manager_with_history.history[1].operation_id

        operation = manager_with_history.get_operation(operation_id)

        assert operation is not None
        assert operation.operation_type == "sort"
        assert operation.operation_id == operation_id

    def test_get_operation_not_found(self, manager_with_history):
        """Test getting non-existing operation by ID."""
        operation = manager_with_history.get_operation("nonexistent_id")
        assert operation is None

    def test_get_operation_empty_history(self):
        """Test getting operation from empty history."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)
        operation = manager.get_operation("any_id")
        assert operation is None


class TestHistoryManagerRestore:
    """Test HistoryManager restore functionality."""

    @pytest.fixture
    def manager_with_snapshots(self):
        """Create a HistoryManager with snapshot history."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        pd.DataFrame({"col1": [4, 5, 6]})
        df3 = pd.DataFrame({"col1": [7, 8, 9]})

        op1_id = manager.add_operation("op1", {}, df1)
        op2_id = manager.add_operation("op2", {})  # No snapshot
        op3_id = manager.add_operation("op3", {}, df3)

        return manager, op1_id, op2_id, op3_id

    def test_restore_to_operation_with_snapshot(self, manager_with_snapshots):
        """Test restoring to operation that has a snapshot."""
        manager, op1_id, _op2_id, _op3_id = manager_with_snapshots

        restored_data = manager.restore_to_operation(op1_id)

        assert restored_data is not None
        pd.testing.assert_frame_equal(restored_data, pd.DataFrame({"col1": [1, 2, 3]}))
        assert manager.current_index == 0
        assert manager.redo_stack == []  # Should be cleared

    def test_restore_to_operation_without_direct_snapshot(self, manager_with_snapshots):
        """Test restoring to operation without direct snapshot."""
        manager, _op1_id, op2_id, _op3_id = manager_with_snapshots

        # Restore to op2 (no snapshot) should find nearest previous snapshot (op1)
        restored_data = manager.restore_to_operation(op2_id)

        assert restored_data is not None
        pd.testing.assert_frame_equal(restored_data, pd.DataFrame({"col1": [1, 2, 3]}))
        assert manager.current_index == 1

    def test_restore_to_operation_not_found(self, manager_with_snapshots):
        """Test restoring to non-existing operation."""
        manager, _op1_id, _op2_id, _op3_id = manager_with_snapshots

        with patch("src.databeak.models.history_manager.logger.error") as mock_logger:
            restored_data = manager.restore_to_operation("nonexistent_id")

            assert restored_data is None
            mock_logger.assert_called_once()

    def test_restore_to_operation_no_snapshots_available(self):
        """Test restoring when no snapshots are available."""
        manager = HistoryManager("test", HistoryStorage.MEMORY, enable_snapshots=False)

        op_id = manager.add_operation("op1", {})

        with patch("src.databeak.models.history_manager.logger.error") as mock_logger:
            restored_data = manager.restore_to_operation(op_id)

            assert restored_data is None
            mock_logger.assert_called_once()

    @patch("src.databeak.models.history_manager.HistoryManager._save_history")
    def test_restore_saves_state_persistent_storage(self, mock_save, manager_with_snapshots):
        """Test that restore saves state for persistent storage."""
        manager, op1_id, _, _ = manager_with_snapshots
        manager.storage_type = HistoryStorage.JSON  # Change to persistent storage

        manager.restore_to_operation(op1_id)
        mock_save.assert_called_once()


class TestHistoryManagerCleanup:
    """Test HistoryManager cleanup operations."""

    @pytest.fixture
    def temp_manager_with_files(self):
        """Create a HistoryManager with persistent files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            df = pd.DataFrame({"col1": [1, 2, 3]})
            manager.add_operation("op1", {}, df)
            manager.add_operation("op2", {}, df)

            yield manager, temp_dir

    def test_clear_history_memory(self):
        """Test clearing history with memory storage."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        manager.add_operation("op1", {})
        manager.add_operation("op2", {})
        manager.undo()

        manager.clear_history()

        assert manager.history == []
        assert manager.redo_stack == []
        assert manager.current_index == -1

    def test_clear_history_with_files(self, temp_manager_with_files):
        """Test clearing history removes persistent files."""
        manager, temp_dir = temp_manager_with_files

        # Verify files exist before clearing
        history_file = Path(temp_dir) / "history_test_session_123.json"
        snapshot_dir = Path(temp_dir) / "snapshots" / "test_session_123"
        assert history_file.exists()
        assert snapshot_dir.exists()

        manager.clear_history()

        # Verify files are removed
        assert not history_file.exists()
        assert not snapshot_dir.exists()

        # Verify memory is cleared
        assert manager.history == []
        assert manager.redo_stack == []
        assert manager.current_index == -1

    def test_clear_history_cleans_both_json_and_pkl(self):
        """Test clearing history removes both JSON and pickle files if they exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create both JSON and pickle files
            json_file = Path(temp_dir) / "history_test_session_123.json"
            pkl_file = Path(temp_dir) / "history_test_session_123.pkl"

            json_file.write_text("{}")
            pkl_file.write_bytes(b"test")

            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            manager.clear_history()

            assert not json_file.exists()
            assert not pkl_file.exists()

    def test_clear_history_handles_missing_files(self):
        """Test clearing history handles missing files gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test_session_123",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            # Should not raise exception even if no files exist
            manager.clear_history()

            assert manager.history == []


class TestHistoryManagerExport:
    """Test HistoryManager export functionality."""

    @pytest.fixture
    def manager_with_history(self):
        """Create a HistoryManager with test history."""
        manager = HistoryManager(
            session_id="test_session_123",
            storage_type=HistoryStorage.MEMORY,
        )

        df = pd.DataFrame({"col1": [1, 2, 3]})

        manager.add_operation(
            "filter",
            {"column": "age", "condition": "> 25"},
            df,
            {"user": "test_user"},
        )
        manager.add_operation("sort", {"column": "name", "ascending": True})

        return manager

    def test_export_history_json(self, manager_with_history):
        """Test exporting history to JSON format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            file_path = f.name

        try:
            result = manager_with_history.export_history(file_path, "json")

            assert result is True

            # Verify file contents
            with Path(file_path).open() as f:
                data = json.load(f)

            assert data["session_id"] == "test_session_123"
            assert data["total_operations"] == 2
            assert data["current_position"] == 1
            assert len(data["operations"]) == 2

            # Check first operation
            op1 = data["operations"][0]
            assert op1["operation_type"] == "filter"
            assert op1["details"] == {"column": "age", "condition": "> 25"}
            assert op1["can_restore"] is True

            # Check second operation
            op2 = data["operations"][1]
            assert op2["operation_type"] == "sort"
            assert op2["details"] == {"column": "name", "ascending": True}
            assert op2["can_restore"] is False

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_export_history_csv(self, manager_with_history):
        """Test exporting history to CSV format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            file_path = f.name

        try:
            result = manager_with_history.export_history(file_path, "csv")

            assert result is True

            # Verify file contents
            df = pd.read_csv(file_path)

            assert len(df) == 2
            assert list(df.columns) == ["timestamp", "operation_type", "details", "has_snapshot"]

            # Check first row
            assert df.iloc[0]["operation_type"] == "filter"
            assert json.loads(df.iloc[0]["details"]) == {"column": "age", "condition": "> 25"}
            assert df.iloc[0]["has_snapshot"]  # Compare value, not identity

            # Check second row
            assert df.iloc[1]["operation_type"] == "sort"
            assert json.loads(df.iloc[1]["details"]) == {"column": "name", "ascending": True}
            assert not df.iloc[1]["has_snapshot"]  # Compare value, not identity

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_export_history_invalid_format(self, manager_with_history):
        """Test exporting history with invalid format."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = f.name

        try:
            # The export_history method doesn't explicitly handle unsupported formats
            # It will silently succeed but not write anything for unsupported formats
            result = manager_with_history.export_history(file_path, "xml")
            assert result is True  # Method returns True even for unsupported formats

            # File should exist but be empty or have minimal content
            assert Path(file_path).exists()
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("src.databeak.models.history_manager.logger.error")
    def test_export_history_write_error(self, mock_logger, manager_with_history):
        """Test export error handling."""
        # Try to write to invalid path
        invalid_path = "/invalid/path/that/does/not/exist.json"

        result = manager_with_history.export_history(invalid_path, "json")

        assert result is False
        mock_logger.assert_called_once()

    def test_export_history_empty_history(self):
        """Test exporting empty history."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            file_path = f.name

        try:
            result = manager.export_history(file_path, "json")

            assert result is True

            # Verify file contents
            with Path(file_path).open() as f:
                data = json.load(f)

            assert data["total_operations"] == 0
            assert data["operations"] == []

        finally:
            Path(file_path).unlink(missing_ok=True)


class TestHistoryManagerStatistics:
    """Test HistoryManager statistics functionality."""

    def test_get_statistics_empty_history(self):
        """Test statistics with empty history."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        stats = manager.get_statistics()

        expected = {
            "total_operations": 0,
            "operation_types": {},
            "first_operation": None,
            "last_operation": None,
            "snapshots_count": 0,
        }

        # Check subset of fields (some are dynamic)
        for key, value in expected.items():
            assert stats[key] == value

    def test_get_statistics_with_operations(self):
        """Test statistics with operations."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        df = pd.DataFrame({"col1": [1, 2, 3]})

        # Add various operations
        manager.add_operation("filter", {"column": "age"}, df)
        manager.add_operation("sort", {"column": "name"})
        manager.add_operation("filter", {"column": "dept"})
        manager.add_operation("group", {"column": "region"}, df)

        stats = manager.get_statistics()

        assert stats["total_operations"] == 4
        assert stats["current_position"] == 4
        assert stats["can_undo"] is True
        assert stats["can_redo"] is False
        assert stats["redo_stack_size"] == 0

        # Check operation type counts
        assert stats["operation_types"]["filter"] == 2
        assert stats["operation_types"]["sort"] == 1
        assert stats["operation_types"]["group"] == 1

        # Check snapshots count - first operation gets snapshot, then every 5th
        assert stats["snapshots_count"] == 1  # Only first operation gets snapshot in this case

        # Check timestamps are present
        assert stats["first_operation"] is not None
        assert stats["last_operation"] is not None

        # Check configuration
        assert stats["storage_type"] == "memory"
        assert stats["max_history"] == 100

    def test_get_statistics_after_undo(self):
        """Test statistics after undo operations."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        manager.add_operation("filter", {})
        manager.add_operation("sort", {})
        manager.undo()

        stats = manager.get_statistics()

        assert stats["total_operations"] == 2  # Total remains same
        assert stats["current_position"] == 1  # But position changes
        assert stats["can_undo"] is True
        assert stats["can_redo"] is True
        assert stats["redo_stack_size"] == 1

    def test_get_statistics_with_custom_settings(self):
        """Test statistics with custom manager settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HistoryManager(
                session_id="test",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
                max_history=50,
            )

            # Add an operation so we get the full statistics output
            manager.add_operation("test_op", {"key": "value"})

            stats = manager.get_statistics()

            assert stats["storage_type"] == "json"
            assert stats["max_history"] == 50


@pytest.mark.parametrize(
    "storage_type",
    [HistoryStorage.MEMORY, HistoryStorage.JSON, HistoryStorage.PICKLE],
)
class TestHistoryManagerStorageTypes:
    """Test HistoryManager with different storage types."""

    def test_initialization_storage_type(self, storage_type):
        """Test initialization with different storage types."""
        if storage_type == HistoryStorage.MEMORY:
            manager = HistoryManager("test", storage_type)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = HistoryManager("test", storage_type, temp_dir)

        assert manager.storage_type == storage_type

    def test_add_operations_storage_type(self, storage_type):
        """Test adding operations with different storage types."""
        if storage_type == HistoryStorage.MEMORY:
            manager = HistoryManager("test", storage_type)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = HistoryManager("test", storage_type, temp_dir)

        df = pd.DataFrame({"col1": [1, 2, 3]})

        operation_id = manager.add_operation("filter", {"column": "age"}, df)

        assert len(manager.history) == 1
        assert manager.history[0].operation_id == operation_id
        assert manager.history[0].operation_type == "filter"

    def test_undo_redo_storage_type(self, storage_type):
        """Test undo/redo with different storage types."""
        if storage_type == HistoryStorage.MEMORY:
            manager = HistoryManager("test", storage_type)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = HistoryManager("test", storage_type, temp_dir)

        df = pd.DataFrame({"col1": [1, 2, 3]})

        manager.add_operation("filter", {}, df)
        manager.add_operation("sort", {}, df)

        # Test undo
        undone_op, snapshot = manager.undo()
        assert undone_op is not None
        assert undone_op.operation_type == "sort"
        assert manager.current_index == 0

        # Test redo
        redone_op, _snapshot = manager.redo()
        assert redone_op is not None
        assert redone_op.operation_type == "sort"
        assert manager.current_index == 1


class TestHistoryManagerEdgeCases:
    """Test HistoryManager edge cases and error conditions."""

    def test_operation_id_generation_uniqueness(self):
        """Test that operation IDs are unique."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        ids = set()
        for i in range(10):
            op_id = manager.add_operation(f"op{i}", {})
            assert op_id not in ids
            ids.add(op_id)

    def test_snapshot_copying_independence(self):
        """Test that snapshots are independent copies."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        df = pd.DataFrame({"col1": [1, 2, 3]})

        manager.add_operation("op1", {}, df)

        # Modify original DataFrame
        df.loc[0, "col1"] = 999

        # Snapshot should be unchanged
        snapshot = manager.history[0].data_snapshot
        assert snapshot is not None
        assert snapshot.loc[0, "col1"] == 1

    def test_concurrent_manager_instances(self):
        """Test multiple manager instances with same session ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first manager and add operations
            manager1 = HistoryManager(
                session_id="shared_session",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            manager1.add_operation("op1", {"test": "data"})

            # Create second manager - should load existing history
            manager2 = HistoryManager(
                session_id="shared_session",
                storage_type=HistoryStorage.JSON,
                history_dir=temp_dir,
            )

            assert len(manager2.history) == 1
            assert manager2.history[0].operation_type == "op1"

    def test_large_history_performance(self):
        """Test manager with large history (basic performance test)."""
        manager = HistoryManager("test", HistoryStorage.MEMORY, max_history=1000)

        # Add many operations
        for i in range(100):
            manager.add_operation(f"op{i}", {"index": i})

        # Should handle operations efficiently
        assert len(manager.history) == 100
        assert manager.current_index == 99

        # Test queries work with large history
        stats = manager.get_statistics()
        assert stats["total_operations"] == 100

        history = manager.get_history(limit=10)
        assert len(history) == 10

    def test_dataframe_with_special_types(self):
        """Test handling DataFrames with special data types."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        # Create DataFrame with various data types
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "datetime_col": pd.date_range("2023-01-01", periods=3),
            },
        )

        manager.add_operation("transform", {"type": "special"}, df)

        # Verify snapshot preserves data types
        snapshot = manager.history[0].data_snapshot
        assert snapshot is not None
        pd.testing.assert_frame_equal(snapshot, df)

    def test_operation_details_serialization(self):
        """Test complex operation details serialization."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        complex_details = {
            "filters": [{"column": "age", "op": ">", "value": 25}],
            "sorts": [{"column": "name", "ascending": True}],
            "nested": {"deep": {"value": [1, 2, 3]}},
            "special_chars": "Special: àáâãäå",
        }

        operation_id = manager.add_operation("complex_op", complex_details)

        operation = manager.get_operation(operation_id)
        assert operation is not None
        assert operation.details == complex_details

    @patch("src.databeak.models.history_manager.logger.info")
    def test_logging_operations(self, mock_logger):
        """Test that operations are properly logged."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        manager.add_operation("filter", {})
        mock_logger.assert_called()

        manager.undo()
        mock_logger.assert_called()

        manager.redo()
        mock_logger.assert_called()

    def test_history_with_none_values(self):
        """Test handling of None values in operation details."""
        manager = HistoryManager("test", HistoryStorage.MEMORY)

        # Test with None metadata
        manager.add_operation("op1", {"key": "value"}, metadata=None)
        assert manager.history[0].metadata == {}

        # Test with None in details
        manager.add_operation("op2", {"key": None, "other": "value"})
        assert manager.history[1].details == {"key": None, "other": "value"}
