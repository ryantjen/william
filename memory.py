from __future__ import annotations
from typing import Optional, List, Dict, Any
import time
import hashlib

import chromadb
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY, EMBEDDING_MODEL

chroma_client = chromadb.PersistentClient(path="./chroma_storage")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)

collection = chroma_client.get_or_create_collection(
    name="research_memory",
    embedding_function=openai_ef
)

def _make_id(text: str, project: Optional[str], salt: str) -> str:
    """Generate a unique ID for a memory using SHA256."""
    base = f"{project or ''}|{salt}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(base).hexdigest()

def store_memory(text: str, metadata: dict | None = None, project: str | None = None) -> str:
    """
    Store one memory and return its id.
    """
    final_metadata = dict(metadata or {})
    if project:
        final_metadata["project"] = project
    final_metadata.setdefault("created_at", int(time.time()))

    mem_id = _make_id(text, project, salt=str(final_metadata.get("created_at", "")))

    collection.add(
        documents=[text],
        metadatas=[final_metadata],
        ids=[mem_id]
    )
    return mem_id

def store_memories(texts: List[str], metadatas: List[dict], project: str | None = None) -> List[str]:
    """
    Batch store many memories; returns list of ids.
    """
    if not texts:
        return []

    now = int(time.time())
    ids = []
    final_metas = []

    for i, (t, md) in enumerate(zip(texts, metadatas)):
        m = dict(md or {})
        if project:
            m["project"] = project
        m.setdefault("created_at", now)
        mid = _make_id(t, project, salt=f"{now}:{i}")
        ids.append(mid)
        final_metas.append(m)

    collection.add(documents=texts, metadatas=final_metas, ids=ids)
    return ids

def retrieve_memory(query: str, project: str | None = None, n_results: int = 4):
    """
    Retrieve memories filtered by project if provided.
    """
    where_filter = {"project": project} if project else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    return results.get("documents", [[]])[0]

def retrieve_hybrid_memory(query: str, project: str, n_project: int = 4, n_global: int = 3):
    """
    Hybrid: pull project-specific + global memories and combine (project first).
    """
    project_docs = retrieve_memory(query, project=project, n_results=n_project)
    global_docs = retrieve_memory(query, project=None, n_results=n_global)

    # de-duplicate while preserving order
    seen = set()
    combined = []
    for d in (project_docs + global_docs):
        if d and d not in seen:
            seen.add(d)
            combined.append(d)
    return combined

def list_memories(
    project: str | None = None,
    where_extra: dict | None = None,
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    Fetch memories with optional metadata filters. Returns list of dicts:
    {id, text, metadata}
    """
    where = {}
    if project:
        where["project"] = project
    if where_extra:
        where.update(where_extra)

    res = collection.get(where=where if where else None, include=["documents", "metadatas"])

    items = []
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    for mid, doc, md in zip(ids, docs, metas):
        items.append({"id": mid, "text": doc, "metadata": md or {}})

    # newest first
    items.sort(key=lambda x: x["metadata"].get("created_at", 0), reverse=True)
    return items[:limit]

def delete_memory(mem_id: str):
    collection.delete(ids=[mem_id])

def count_memories(project: str | None = None, where_extra: dict | None = None) -> int:
    where = {}
    if project:
        where["project"] = project
    if where_extra:
        where.update(where_extra)
    res = collection.get(where=where if where else None, include=[])
    return len(res.get("ids", []))