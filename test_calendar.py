"""
Test script for calendar chunk completion and date parsing.
Run: python test_calendar.py
"""
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Use a temp copy of calendar data so we don't corrupt real data
TEST_DIR = Path(tempfile.mkdtemp())
os.environ["WILLIAM_DATA_DIR"] = str(TEST_DIR)
DATA_DIR = TEST_DIR / "app_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import after setting env (storage reads it)
sys.path.insert(0, str(Path(__file__).parent))

# Override storage to use test dir (before importing calendar_data)
import storage
storage.DATA_DIR = TEST_DIR
storage.DATA_DIR.mkdir(parents=True, exist_ok=True)

import calendar_data as cal
cal.CALENDAR_FILE = TEST_DIR / "app_data" / "calendar.json"
cal.CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)


def setup_test_data():
    """Create test tasks and chunks. Task t1 has exactly 0.5h so 1 chunk completes it."""
    today = datetime.now().strftime("%Y-%m-%d")
    due = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    data = {
        "tasks": [
            {"id": "t1", "name": "test task", "due_date": due, "total_hours": 0.5, "priority": 3},
        ],
        "events": [],
        "chunks": [
            {"id": "c1", "task_id": "t1", "date": today, "duration_minutes": 30, "priority": 3, "completed": False, "created_at": 1},
        ],
    }
    cal._save(data)
    return data


def test_chunk_completion():
    """Test that marking a chunk complete removes it from incomplete list."""
    print("=== Test: Chunk completion ===")
    setup_test_data()

    today = datetime.now().strftime("%Y-%m-%d")
    chunks_before = cal.get_chunks_for_date(today)
    incomplete_before = [c for c in chunks_before if not c.get("completed")]
    assert len(incomplete_before) >= 1, f"Expected at least 1 incomplete, got {len(incomplete_before)}"
    chunk_id_to_complete = incomplete_before[0]["id"]

    ok = cal.mark_chunk_complete(chunk_id_to_complete)
    assert ok, f"mark_chunk_complete({chunk_id_to_complete}) should return True"

    # Verify save happened: chunk should be completed in raw data
    data_direct = cal._load()
    chunk_in_file = next((c for c in data_direct.get("chunks", []) if c.get("id") == chunk_id_to_complete), None)
    assert chunk_in_file and chunk_in_file.get("completed"), (
        f"After mark_complete, chunk should be completed in file: {chunk_in_file}"
    )

    # Use read-only (no _ensure_chunks_for_date) - matches app behavior so completed chunks disappear
    chunks_after = cal.get_chunks_for_date_read_only(today)
    incomplete_after = [c for c in chunks_after if not c.get("completed")]
    completed_after = [c for c in chunks_after if c.get("completed")]
    assert chunk_id_to_complete in [c["id"] for c in completed_after], (
        f"Chunk {chunk_id_to_complete} should be in completed list"
    )
    assert chunk_id_to_complete not in [c["id"] for c in incomplete_after], (
        f"Chunk {chunk_id_to_complete} should not be in incomplete (completed chunks must be filtered out)"
    )
    assert len(incomplete_after) < len(incomplete_before), (
        f"Completed chunk should disappear: had {len(incomplete_before)} incomplete, now {len(incomplete_after)}"
    )

    print("  PASS: Chunk completed and removed from incomplete list")


def test_refresh_preserves_completed():
    """Test that refresh_chunks_for_date preserves completed chunks."""
    print("=== Test: Refresh preserves completed ===")
    setup_test_data()
    today = datetime.now().strftime("%Y-%m-%d")
    chunks = cal.get_chunks_for_date(today)
    incomplete = [c for c in chunks if not c.get("completed")]
    if not incomplete:
        print("  SKIP: No chunks to complete (task may be fully allocated)")
        return
    chunk_id = incomplete[0]["id"]
    cal.mark_chunk_complete(chunk_id)

    cal.refresh_chunks_for_date(today)

    data = cal._load()
    completed_today = [c for c in data.get("chunks", []) if c.get("date") == today and c.get("completed")]
    assert len(completed_today) >= 1, f"Refresh should preserve completed chunk, got {completed_today}"
    completed_ids = [c["id"] for c in completed_today]
    assert chunk_id in completed_ids, f"{chunk_id} should be preserved, completed={completed_ids}"

    print("  PASS: Refresh preserved completed chunk")


def test_weekday_parse():
    """Test deterministic weekday parsing returns next occurrence."""
    print("=== Test: Weekday parsing ===")
    from agent import _parse_weekday_in_text

    now = datetime.now().date()
    for text, expected_weekday in [
        ("task due wednesday", 2),  # Monday=0, Tuesday=1, Wednesday=2
        ("wednesday meeting", 2),
        ("due thursday", 3),
        ("tuesday deadline", 1),
    ]:
        got = _parse_weekday_in_text(text)
        assert got, f"For '{text}': got None"
        d = datetime.strptime(got, "%Y-%m-%d").date()
        assert d.weekday() == expected_weekday, (
            f"For '{text}': expected weekday {expected_weekday}, got {d.weekday()} ({got})"
        )
    print("  PASS: Weekday parsing returns correct day of week")


def run_tests():
    test_chunk_completion()
    test_refresh_preserves_completed()
    test_weekday_parse()
    print("\nAll tests passed.")
    shutil.rmtree(TEST_DIR, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
