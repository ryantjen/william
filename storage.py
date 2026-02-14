# storage.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path("./app_data")
DATA_DIR.mkdir(exist_ok=True)

PROJECTS_FILE = DATA_DIR / "projects.json"


def load_projects() -> List[Dict[str, Any]]:
    """
    Load projects as list of dicts: [{"name": "...", "goal": "..." or None}, ...]
    Handles migration from old format (list of strings).
    """
    if not PROJECTS_FILE.exists():
        default = [{"name": "General", "goal": None}]
        PROJECTS_FILE.write_text(json.dumps(default, indent=2))
        return default
    
    data = json.loads(PROJECTS_FILE.read_text())
    
    # Migration: if old format (list of strings), convert to new format
    if data and isinstance(data[0], str):
        data = [{"name": name, "goal": None} for name in data]
        save_projects_raw(data)
    
    return data


def save_projects_raw(projects: List[Dict[str, Any]]) -> None:
    """Save projects list directly (for internal use)."""
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2))


def save_projects(projects: List[Dict[str, Any]]) -> None:
    """Save projects, sorting by name and deduplicating."""
    # Deduplicate by name, keeping first occurrence
    seen = set()
    unique = []
    for p in projects:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)
    # Sort by name
    unique.sort(key=lambda x: x["name"])
    save_projects_raw(unique)


def get_project_names(projects: List[Dict[str, Any]]) -> List[str]:
    """Get just the project names from the projects list."""
    return [p["name"] for p in projects]


def get_project_goal(projects: List[Dict[str, Any]], name: str) -> Optional[str]:
    """Get the goal for a specific project."""
    for p in projects:
        if p["name"] == name:
            return p.get("goal")
    return None


def set_project_goal(projects: List[Dict[str, Any]], name: str, goal: Optional[str]) -> List[Dict[str, Any]]:
    """Set the goal for a specific project. Returns updated projects list."""
    for p in projects:
        if p["name"] == name:
            p["goal"] = goal if goal and goal.strip() else None
            break
    return projects


def rename_project(projects: List[Dict[str, Any]], old_name: str, new_name: str) -> List[Dict[str, Any]]:
    """
    Rename a project. Updates projects list, chat file, and ingested records.
    Caller must also update ChromaDB memories via memory.merge_projects(old_name, new_name).
    Returns updated projects list.
    """
    if old_name == new_name:
        return projects
    if not any(p["name"] == old_name for p in projects):
        return projects
    if any(p["name"] == new_name for p in projects):
        return projects  # New name already exists; caller should validate

    # Update projects list
    for p in projects:
        if p["name"] == old_name:
            p["name"] = new_name
            break

    # Migrate chat: load from old, save to new, delete old
    old_chat = load_chat(old_name)
    save_chat(new_name, old_chat)
    delete_chat(old_name)

    # Migrate ingested records
    ingested = load_ingested()
    if old_name in ingested:
        ingested[new_name] = ingested.pop(old_name)
        save_ingested(ingested)

    return projects


def chat_path(project: str) -> Path:
    safe = "".join(c for c in project if c.isalnum() or c in (" ", "_", "-")).strip()
    safe = safe.replace(" ", "_") or "General"
    return DATA_DIR / f"chat_{safe}.json"


def load_chat(project: str) -> List[Dict[str, Any]]:
    p = chat_path(project)
    if not p.exists():
        p.write_text(json.dumps([], indent=2))
    return json.loads(p.read_text())


def save_chat(project: str, messages: List[Dict[str, Any]]) -> None:
    p = chat_path(project)
    p.write_text(json.dumps(messages, indent=2))


def delete_chat(project: str) -> bool:
    """Delete the chat history file for a project. Returns True if deleted."""
    p = chat_path(project)
    if p.exists():
        p.unlink()
        return True
    return False


INGESTED_FILE = DATA_DIR / "ingested_files.json"

def load_ingested() -> dict:
    if not INGESTED_FILE.exists():
        INGESTED_FILE.write_text("{}")
    return json.loads(INGESTED_FILE.read_text())

def save_ingested(obj: dict) -> None:
    INGESTED_FILE.write_text(json.dumps(obj, indent=2))

def clear_ingested_for_project(project: str) -> int:
    """Clear all ingestion records for a project. Returns count cleared."""
    ingested = load_ingested()
    if project in ingested:
        count = len(ingested[project])
        del ingested[project]
        save_ingested(ingested)
        return count
    return 0

def clear_ingested_file(project: str, file_hash: str) -> bool:
    """Clear ingestion record for a specific file. Returns True if cleared."""
    ingested = load_ingested()
    if project in ingested and file_hash in ingested[project]:
        del ingested[project][file_hash]
        save_ingested(ingested)
        return True
    return False