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

def is_duplicate(text: str, project: str | None = None, similarity_threshold: float = 0.85) -> bool:
    """
    Check if a similar memory already exists using semantic similarity.
    Returns True if a duplicate is found (similarity > threshold).
    """
    try:
        # Query for the most similar existing memory
        results = collection.query(
            query_texts=[text],
            n_results=1,
            where={"project": project} if project else None,
            include=["distances"]
        )
        
        # ChromaDB returns distances (lower = more similar for cosine)
        # Distance of 0 = identical, distance of 2 = opposite
        # Convert to similarity: similarity = 1 - (distance / 2)
        distances = results.get("distances", [[]])
        if distances and distances[0]:
            distance = distances[0][0]
            similarity = 1 - (distance / 2)
            return similarity >= similarity_threshold
        
        return False
    except Exception:
        # If check fails, assume not duplicate (safer to store than lose)
        return False

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

def store_memory_if_unique(text: str, metadata: dict | None = None, project: str | None = None, similarity_threshold: float = 0.85) -> str | None:
    """
    Store a memory only if a similar one doesn't already exist.
    Returns the memory id if stored, None if duplicate was found.
    """
    if is_duplicate(text, project=project, similarity_threshold=similarity_threshold):
        return None
    return store_memory(text, metadata, project)

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

def store_memories_if_unique(texts: List[str], metadatas: List[dict], project: str | None = None, similarity_threshold: float = 0.85) -> tuple[List[str], int]:
    """
    Batch store memories, skipping duplicates.
    Returns (list of ids stored, count of duplicates skipped).
    """
    if not texts:
        return [], 0

    ids_stored = []
    duplicates_skipped = 0
    now = int(time.time())

    for i, (t, md) in enumerate(zip(texts, metadatas)):
        if is_duplicate(t, project=project, similarity_threshold=similarity_threshold):
            duplicates_skipped += 1
            continue
        
        m = dict(md or {})
        if project:
            m["project"] = project
        m.setdefault("created_at", now)
        mid = _make_id(t, project, salt=f"{now}:{i}")
        
        collection.add(documents=[t], metadatas=[m], ids=[mid])
        ids_stored.append(mid)

    return ids_stored, duplicates_skipped

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


def retrieve_memory_with_metadata(query: str, project: str | None = None, n_results: int = 4):
    """
    Retrieve memories with full metadata for citation tracking.
    Returns list of dicts: [{id, text, name, type, source}, ...]
    """
    where_filter = {"project": project} if project else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas"]
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    memories = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        mem_id = ids[i] if i < len(ids) else ""
        memories.append({
            "id": mem_id,
            "text": doc,
            "name": meta.get("name", ""),
            "type": meta.get("type", "unknown"),
            "source": meta.get("source", "unknown"),
            "confidence": int(meta.get("confidence", 0)),
            "created_at": int(meta.get("created_at", 0)),
        })
    return memories

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
    limit: int = 200,
    global_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch memories with optional metadata filters. Returns list of dicts:
    {id, text, metadata}
    If global_only=True, returns only memories without a project (global memories).
    """
    if global_only:
        res = collection.get(where=None, include=["documents", "metadatas"])
        items = []
        for mid, doc, md in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
            meta = md or {}
            if "project" not in meta:
                items.append({"id": mid, "text": doc, "metadata": meta})
        if where_extra:
            for k, v in where_extra.items():
                items = [it for it in items if (it["metadata"] or {}).get(k) == v]
        items.sort(key=lambda x: x["metadata"].get("created_at", 0), reverse=True)
        return items[:limit]

    # Build where clause - use $and for multiple conditions
    conditions = []
    if project:
        conditions.append({"project": {"$eq": project}})
    if where_extra:
        for k, v in where_extra.items():
            conditions.append({k: {"$eq": v}})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    res = collection.get(where=where, include=["documents", "metadatas"])

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

def delete_memories(ids: List[str]) -> int:
    """Delete multiple memories by id. Returns count deleted."""
    ids = [i for i in ids if i]
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)

def set_memory_irrelevant(mem_id: str, irrelevant: bool) -> bool:
    """Mark or unmark a memory as irrelevant. Returns True if successful."""
    mem = get_memory_by_id(mem_id)
    if not mem:
        return False
    return update_memory(mem_id, mem["text"], {"irrelevant": irrelevant})

def update_memory(mem_id: str, new_text: str, new_metadata: dict = None) -> bool:
    """
    Update an existing memory's text and optionally metadata.
    ChromaDB doesn't support direct updates, so we delete and re-add.
    Returns True if successful, False if memory not found.
    """
    # Get existing memory
    res = collection.get(ids=[mem_id], include=["metadatas"])
    if not res.get("ids"):
        return False
    
    old_meta = res["metadatas"][0] if res.get("metadatas") else {}
    
    # Merge metadata (keep old values, update with new)
    final_meta = dict(old_meta)
    if new_metadata:
        final_meta.update(new_metadata)
    final_meta["last_edited"] = int(time.time())
    
    # Delete old and add new (keeps same ID)
    collection.delete(ids=[mem_id])
    collection.add(
        documents=[new_text],
        metadatas=[final_meta],
        ids=[mem_id]
    )
    return True

def get_memory_by_id(mem_id: str) -> Dict[str, Any] | None:
    """Get a single memory by its ID."""
    res = collection.get(ids=[mem_id], include=["documents", "metadatas"])
    if not res.get("ids"):
        return None
    return {
        "id": res["ids"][0],
        "text": res["documents"][0] if res.get("documents") else "",
        "metadata": res["metadatas"][0] if res.get("metadatas") else {}
    }

def delete_project_memories(project: str) -> int:
    """
    Delete all memories for a specific project.
    Returns count of memories deleted.
    """
    where = {"project": {"$eq": project}}
    res = collection.get(where=where, include=[])
    ids = res.get("ids", [])
    
    if ids:
        collection.delete(ids=ids)
    
    return len(ids)

def merge_projects(source_project: str, target_project: str) -> int:
    """
    Merge all memories from source_project into target_project.
    Updates the project metadata field. Returns count of memories merged.
    """
    where = {"project": {"$eq": source_project}}
    res = collection.get(where=where, include=["documents", "metadatas"])
    
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    
    if not ids:
        return 0
    
    # Update each memory's project tag
    for i, (mid, doc, md) in enumerate(zip(ids, docs, metas)):
        new_meta = dict(md or {})
        new_meta["project"] = target_project
        new_meta["merged_from"] = source_project  # Track merge history
        
        # ChromaDB doesn't support direct metadata updates, so we delete and re-add
        collection.delete(ids=[mid])
        collection.add(
            documents=[doc],
            metadatas=[new_meta],
            ids=[mid]
        )
    
    return len(ids)

def count_memories(project: str | None = None, where_extra: dict | None = None, global_only: bool = False) -> int:
    if global_only:
        res = collection.get(where=None, include=["metadatas"])
        items = [md or {} for md in (res.get("metadatas") or []) if "project" not in (md or {})]
        if where_extra:
            for k, v in where_extra.items():
                items = [m for m in items if m.get(k) == v]
        return len(items)

    # Build where clause - use $and for multiple conditions
    conditions = []
    if project:
        conditions.append({"project": {"$eq": project}})
    if where_extra:
        for k, v in where_extra.items():
            conditions.append({k: {"$eq": v}})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    res = collection.get(where=where, include=[])
    return len(res.get("ids", []))

def get_all_embeddings(project: str | None = None) -> Dict[str, Any]:
    """
    Fetch all memories with their embeddings for visualization.
    Returns dict with: ids, texts, metadatas, embeddings
    """
    where = {"project": {"$eq": project}} if project else None
    
    res = collection.get(
        where=where,
        include=["documents", "metadatas", "embeddings"]
    )
    
    return {
        "ids": res.get("ids", []),
        "texts": res.get("documents", []),
        "metadatas": res.get("metadatas", []),
        "embeddings": res.get("embeddings", [])
    }