"""
Calendar storage: tasks (goals with total work), events (time blocks), chunks (daily work units).
Chunks are auto-generated day-by-day; missed chunks roll over to the next day.
"""
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from storage import DATA_DIR


def _get_chunk_sizes(today_mins: int, task_name: str = "") -> List[int]:
    """Use LLM to suggest chunk sizes if available; else fall back to algorithmic split."""
    try:
        from agent import suggest_chunk_sizes
        suggested = suggest_chunk_sizes(today_mins, task_name)
        if suggested:
            return suggested
    except Exception:
        pass
    return _optimal_chunk_sizes(today_mins)


def get_chunks_for_date_read_only(date_str: str) -> List[Dict]:
    """Get chunks without running _ensure_chunks_for_date (for testing/display after completion)."""
    data = _load()
    chunks = [c for c in data.get("chunks", []) if c.get("date") == date_str]
    task_map = {t["id"]: t for t in data.get("tasks", [])}
    for c in chunks:
        t = task_map.get(c.get("task_id", ""), {})
        c["_task_name"] = t.get("name", "Task")
        c["_task_due"] = t.get("due_date", "")
    chunks.sort(key=lambda x: (-(x.get("priority") or 0), x.get("created_at", 0)))
    return chunks

CALENDAR_FILE = DATA_DIR / "calendar.json"

DEFAULT_CHUNK_MINUTES = 30
WORK_DAY_START = 9 * 60   # 9:00 in minutes from midnight
WORK_DAY_END = 18 * 60    # 18:00


def _load() -> dict:
    if not CALENDAR_FILE.exists():
        return {"tasks": [], "events": [], "chunks": []}
    return json.loads(CALENDAR_FILE.read_text())


def _save(data: dict) -> None:
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALENDAR_FILE.write_text(json.dumps(data, indent=2))


# ----- Tasks -----

def load_tasks() -> List[Dict]:
    return _load().get("tasks", [])


def add_task(name: str, due_date: str, total_hours: float, priority: int = 3) -> Dict:
    """Add a task. total_hours required."""
    data = _load()
    task = {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "due_date": due_date,
        "total_hours": float(total_hours),
        "priority": max(1, min(5, priority)),
        "created_at": int(time.time()),
    }
    data.setdefault("tasks", []).append(task)
    _save(data)
    return task


def get_active_tasks(today: str) -> List[Dict]:
    """Tasks not past due, sorted by due_date asc, priority desc."""
    tasks = [t for t in load_tasks() if (t.get("due_date") or "") >= today]
    tasks.sort(key=lambda x: (x.get("due_date", ""), -(x.get("priority") or 0)))
    return tasks


def get_tasks_due_on_date(date_str: str) -> List[Dict]:
    """Tasks that are due on this exact date (for calendar display)."""
    return [t for t in load_tasks() if t.get("due_date") == date_str]


def delete_task(task_id: str) -> bool:
    data = _load()
    data["tasks"] = [t for t in data.get("tasks", []) if t.get("id") != task_id]
    data["chunks"] = [c for c in data.get("chunks", []) if c.get("task_id") != task_id]
    _save(data)
    return True


# ----- Events -----

def load_events() -> List[Dict]:
    return _load().get("events", [])


def add_event(name: str, date: str, start_time: str, end_time: str) -> Dict:
    """start_time, end_time: "HH:MM" 24h."""
    data = _load()
    ev = {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
    }
    data.setdefault("events", []).append(ev)
    _save(data)
    return ev


def get_events_for_date(date: str) -> List[Dict]:
    return [e for e in load_events() if e.get("date") == date]


def delete_event(event_id: str) -> bool:
    data = _load()
    data["events"] = [e for e in data.get("events", []) if e.get("id") != event_id]
    _save(data)
    return True


# ----- Chunks -----

def load_chunks() -> List[Dict]:
    return _load().get("chunks", [])


def _chunks_for_task(chunks: List[Dict], task_id: str) -> List[Dict]:
    return [c for c in chunks if c.get("task_id") == task_id]


def _hours_completed_for_task(chunks: List[Dict], task_id: str) -> float:
    completed = [c for c in _chunks_for_task(chunks, task_id) if c.get("completed")]
    return sum((c.get("duration_minutes") or 0) / 60.0 for c in completed)


def _optimal_chunk_sizes(total_mins: int) -> List[int]:
    """
    Split total_mins into chunks of 15, 25, 30, 45, or 60 min.
    Prefer fewer, longer chunks for deep work; use 25-30 for shorter sessions.
    """
    if total_mins <= 0:
        return []
    preferred = [60, 45, 30, 25, 15]
    chunks = []
    remaining = total_mins
    while remaining > 0:
        used = False
        for size in preferred:
            if size <= remaining:
                chunks.append(size)
                remaining -= size
                used = True
                break
        if not used:
            # Merge remainder into last chunk
            if chunks:
                chunks[-1] += remaining
            else:
                chunks.append(remaining)
            break
    return chunks


def _overdue_chunks(chunks: List[Dict], before_date: str) -> List[Dict]:
    """Incomplete chunks from before this date."""
    return [
        c for c in chunks
        if not c.get("completed") and (c.get("date") or "") < before_date
    ]


def _ensure_chunks_for_date(date_str: str) -> None:
    """
    Generate and store chunks for date_str if not already present.
    Considers: tasks, events, overdue chunks, priority.
    """
    data = _load()
    tasks = data.get("tasks", [])
    events = data.get("events", [])
    chunks = data.get("chunks", [])

    today = time.strftime("%Y-%m-%d")
    if date_str < today:
        return  # Don't generate for past days

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return

    # Active tasks with work remaining
    active = [t for t in tasks if (t.get("due_date") or "") >= date_str]
    task_map = {t["id"]: t for t in active}

    # Overdue: incomplete chunks from before this date
    overdue = _overdue_chunks(chunks, date_str)
    overdue_by_task: Dict[str, list] = {}
    for c in overdue:
        tid = c.get("task_id", "")
        if tid not in overdue_by_task:
            overdue_by_task[tid] = []
        overdue_by_task[tid].append(c)

    # Events on this day (blocked time)
    day_events = [e for e in events if e.get("date") == date_str]

    # Build list of chunks to create
    new_chunks: List[Dict] = []
    existing_for_date = [c for c in chunks if c.get("date") == date_str]

    for task in active:
        if task.get("completed"):
            continue
        tid = task["id"]
        total = task.get("total_hours") or 0
        if total <= 0:
            continue
        completed = _hours_completed_for_task(chunks + new_chunks, tid)
        remaining = total - completed
        if remaining <= 0:
            continue

        try:
            due = datetime.strptime(task["due_date"], "%Y-%m-%d")
            days_left = (due.date() - date_obj.date()).days
        except (ValueError, TypeError):
            days_left = 1
        if days_left < 0:
            continue

        # Overdue work for this task (minutes)
        overdue_mins = sum(
            (c.get("duration_minutes") or 0)
            for c in overdue_by_task.get(tid, [])
        )
        # Today's fair share + catch-up
        if days_left > 0:
            fair_share_hours = remaining / (days_left + 1)
        else:
            fair_share_hours = remaining
        today_hours = fair_share_hours + (overdue_mins / 60.0)
        today_hours = min(today_hours, remaining)

        today_mins = max(15, int(round(today_hours * 60 / 15) * 15))
        today_mins = min(today_mins, int(remaining * 60))

        if today_mins <= 0:
            continue

        # One chunk per task per day (no multiple chunks)
        priority = task.get("priority") or 3
        chunk_mins = min(60, max(15, today_mins))  # clamp to 15-60 for display
        ch = {
            "id": str(uuid.uuid4()),
            "task_id": tid,
            "date": date_str,
            "duration_minutes": chunk_mins,
            "priority": priority,
            "completed": False,
            "created_at": int(time.time()),
        }
        new_chunks.append(ch)

    if new_chunks:
        # Keep completed chunks for this date; replace incomplete with fresh allocation
        existing = [c for c in data.get("chunks", []) if c.get("date") == date_str]
        completed = [c for c in existing if c.get("completed")]
        others = [c for c in data.get("chunks", []) if c.get("date") != date_str]
        data["chunks"] = others + completed + new_chunks
        _save(data)


def refresh_chunks_for_date(date_str: str) -> None:
    """Clear INCOMPLETE chunks and regenerate for this date. Preserves completed chunks."""
    data = _load()
    # Keep: chunks for other dates + completed chunks for this date (don't wipe completion)
    data["chunks"] = [
        c for c in data.get("chunks", [])
        if c.get("date") != date_str or c.get("completed")
    ]
    _save(data)
    _ensure_chunks_for_date(date_str)


def get_chunks_for_date(date_str: str) -> List[Dict]:
    """Get chunks for date, ensuring they exist. Returns chunks with task info."""
    _ensure_chunks_for_date(date_str)
    data = _load()
    chunks = [c for c in data.get("chunks", []) if c.get("date") == date_str]
    task_map = {t["id"]: t for t in data.get("tasks", [])}
    for c in chunks:
        t = task_map.get(c.get("task_id", ""), {})
        c["_task_name"] = t.get("name", "Task")
        c["_task_due"] = t.get("due_date", "")
    chunks.sort(key=lambda x: (-(x.get("priority") or 0), x.get("created_at", 0)))
    return chunks


def mark_chunk_complete(chunk_id: str, actual_minutes: Optional[int] = None) -> bool:
    """Mark chunk complete. Optionally override duration_minutes with actual time spent."""
    data = _load()
    for c in data.get("chunks", []):
        if c.get("id") == chunk_id:
            if actual_minutes is not None:
                c["duration_minutes"] = max(1, actual_minutes)
            c["completed"] = True
            c["completed_at"] = int(time.time())
            _save(data)
            return True
    return False


def update_task_remaining_estimate(task_id: str, remaining_hours: float) -> bool:
    """Update task total_hours to completed + remaining (re-estimate remaining work)."""
    data = _load()
    task = next((t for t in data.get("tasks", []) if t.get("id") == task_id), None)
    if not task:
        return False
    completed = _hours_completed_for_task(data.get("chunks", []), task_id)
    task["total_hours"] = max(0.1, completed + float(remaining_hours))
    _save(data)
    return True


def mark_task_complete(task_id: str) -> int:
    """Mark entire task as complete: set task.completed=True and mark all chunks completed. Returns 1 if task found."""
    data = _load()
    task = next((t for t in data.get("tasks", []) if t.get("id") == task_id), None)
    if not task:
        return 0
    task["completed"] = True
    task["completed_at"] = int(time.time())
    for c in data.get("chunks", []):
        if c.get("task_id") == task_id and not c.get("completed"):
            c["completed"] = True
            c["completed_at"] = int(time.time())
    _save(data)
    return 1


def get_task_completion_pct(task_id: str) -> Optional[int]:
    """Return completion percentage (0-100) for a task, or None if not found. Task-level completed=100%."""
    data = _load()
    task = next((t for t in data.get("tasks", []) if t.get("id") == task_id), None)
    if not task:
        return None
    if task.get("completed"):
        return 100
    total = task.get("total_hours") or 0
    if total <= 0:
        return 0
    chunks = data.get("chunks", [])
    completed = _hours_completed_for_task(chunks, task_id)
    pct = int(round(100 * completed / total))
    return min(100, max(0, pct))


def delete_chunk(chunk_id: str) -> bool:
    data = _load()
    before = len(data.get("chunks", []))
    data["chunks"] = [c for c in data.get("chunks", []) if c.get("id") != chunk_id]
    if len(data["chunks"]) < before:
        _save(data)
        return True
    return False


def cleanup_past_due() -> int:
    """Remove tasks past due. Return count removed."""
    today = time.strftime("%Y-%m-%d")
    data = _load()
    before = len(data.get("tasks", []))
    data["tasks"] = [t for t in data.get("tasks", []) if (t.get("due_date") or "") >= today]
    # Remove chunks for deleted tasks
    kept_ids = {t["id"] for t in data["tasks"]}
    data["chunks"] = [c for c in data.get("chunks", []) if c.get("task_id") in kept_ids]
    _save(data)
    return before - len(data["tasks"])


def cleanup_past_events() -> int:
    """Remove events past their date. Return count removed."""
    today = time.strftime("%Y-%m-%d")
    data = _load()
    before = len(data.get("events", []))
    data["events"] = [e for e in data.get("events", []) if (e.get("date") or "") >= today]
    _save(data)
    return before - len(data["events"])


# ----- Migration from old tasks.json -----

def migrate_from_tasks() -> int:
    """Migrate old tasks.json to new calendar format. Returns count migrated."""
    old_file = DATA_DIR / "tasks.json"
    if not old_file.exists():
        return 0
    try:
        old = json.loads(old_file.read_text())
        if not isinstance(old, list):
            return 0
        data = _load()
        count = 0
        for t in old:
            if not t.get("due_date") or not t.get("duration_hours"):
                continue
            if t.get("completed"):
                continue
            due = t.get("due_date", "")
            if due < time.strftime("%Y-%m-%d"):
                continue
            add_task(
                name=t.get("name", "Migrated task"),
                due_date=due,
                total_hours=float(t.get("duration_hours", 1)),
                priority=t.get("priority") or 3,
            )
            count += 1
        if count > 0:
            old_file.rename(old_file.with_suffix(".json.bak"))
        return count
    except Exception:
        return 0
