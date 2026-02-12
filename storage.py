# storage.py
import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path("./app_data")
DATA_DIR.mkdir(exist_ok=True)

PROJECTS_FILE = DATA_DIR / "projects.json"


def load_projects() -> List[str]:
    if not PROJECTS_FILE.exists():
        PROJECTS_FILE.write_text(json.dumps(["General"], indent=2))
    return json.loads(PROJECTS_FILE.read_text())


def save_projects(projects: List[str]) -> None:
    PROJECTS_FILE.write_text(json.dumps(sorted(set(projects)), indent=2))


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


INGESTED_FILE = DATA_DIR / "ingested_files.json"

def load_ingested() -> dict:
    if not INGESTED_FILE.exists():
        INGESTED_FILE.write_text("{}")
    return json.loads(INGESTED_FILE.read_text())

def save_ingested(obj: dict) -> None:
    INGESTED_FILE.write_text(json.dumps(obj, indent=2))