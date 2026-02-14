from __future__ import annotations
from typing import Optional, List, Dict, Any
import math
import time
import hashlib
from pathlib import Path

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


def _build_where(project: Optional[str], where_extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build ChromaDB where clause from project and extra metadata filters."""
    conditions = []
    if project:
        conditions.append({"project": {"$eq": project}})
    if where_extra:
        for k, v in where_extra.items():
            conditions.append({k: {"$eq": v}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


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

# Re-ranking constants (tuned for low latency: small weights, simple math)
_OVER_FETCH = 2  # Fetch 2x then re-rank down (minimal extra cost)
_RECENCY_HALF_LIFE_DAYS = 60
_RECENCY_WEIGHT = 0.15
_IMPORTANCE_WEIGHT = 0.08


def retrieve_memory_with_metadata(query: str, project: str | None = None, n_results: int = 4):
    """
    Retrieve memories with full metadata for citation tracking.
    Over-fetches, re-ranks by similarity + recency + importance, returns top n_results.
    Returns list of dicts: [{id, text, name, type, source, importance, created_at}, ...]
    """
    where_filter = {"project": project} if project else None
    fetch_n = min(n_results * _OVER_FETCH, 50)  # Cap to avoid huge pulls
    
    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    now = int(time.time())
    scored = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        mem_id = ids[i] if i < len(ids) else ""
        dist = distances[i] if i < len(distances) else 1.0
        
        # Similarity: cosine distance 0=identical, 2=opposite
        similarity = max(0.0, 1.0 - (dist / 2.0))
        
        # Recency: use last_accessed_at if set, else created_at
        ts = meta.get("last_accessed_at") or meta.get("created_at", now)
        age_days = (now - ts) / 86400.0
        recency_bonus = math.exp(-age_days / _RECENCY_HALF_LIFE_DAYS)
        
        # Importance: 1-5 -> [0, 1], default 3
        imp = int(meta.get("importance", 3))
        imp = max(1, min(5, imp))
        importance_bonus = (imp - 1) / 4.0
        
        score = (
            similarity
            + _RECENCY_WEIGHT * recency_bonus
            + _IMPORTANCE_WEIGHT * importance_bonus
        )
        scored.append((score, mem_id, doc, meta))
    
    scored.sort(key=lambda x: -x[0])
    
    memories = []
    for _, mem_id, doc, meta in scored[:n_results]:
        memories.append({
            "id": mem_id,
            "text": doc,
            "name": meta.get("name", ""),
            "type": meta.get("type", "unknown"),
            "source": meta.get("source", "unknown"),
            "importance": int(meta.get("importance", 3)),
            "confidence": int(meta.get("confidence", 0)),
            "created_at": int(meta.get("created_at", 0)),
        })
    return memories


def touch_memories(mem_ids: List[str]) -> None:
    """
    Update last_accessed_at for cited memories. Call from a background thread
    so it does not add latency to the main request.
    """
    if not mem_ids:
        return
    mem_ids = list(dict.fromkeys(mem_ids))  # dedupe preserving order
    now = int(time.time())
    res = collection.get(ids=mem_ids, include=["documents", "metadatas"])
    for i, mid in enumerate(res.get("ids", [])):
        doc = (res.get("documents") or [])[i] if i < len(res.get("documents", [])) else ""
        meta = dict((res.get("metadatas") or [])[i] if i < len(res.get("metadatas", [])) else {})
        meta["last_accessed_at"] = now
        try:
            collection.delete(ids=[mid])
            collection.add(documents=[doc], metadatas=[meta], ids=[mid])
        except Exception:
            pass  # Don't fail background touch

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

    where = _build_where(project, where_extra)
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

    where = _build_where(project, where_extra)
    res = collection.get(where=where, include=[])
    return len(res.get("ids", []))


def get_memory_stats() -> Dict[str, Any]:
    """
    Return storage health stats: total_memories, disk_size_mb, embedding_model.
    """
    total = len(collection.get(where=None, include=[]).get("ids", []))
    chroma_path = Path("./chroma_storage")
    disk_bytes = 0
    if chroma_path.exists():
        disk_bytes = sum(f.stat().st_size for f in chroma_path.rglob("*") if f.is_file())
    from config import EMBEDDING_MODEL
    # Use GB for >= 1 GB, else MB
    if disk_bytes >= 1024**3:
        disk_display = f"{disk_bytes / 1024**3:.2f} GB"
    else:
        disk_display = f"{disk_bytes / 1024**2:.2f} MB"
    return {
        "total_memories": total,
        "disk_display": disk_display,
        "embedding_model": EMBEDDING_MODEL,
    }


def prune_low_importance(
    project: str | None = None,
    importance_below: int = 2,
    limit: int = 100,
    global_only: bool = False,
) -> int:
    """
    Delete memories with importance <= importance_below.
    Returns count deleted.
    """
    items = list_memories(project=project, limit=2000, global_only=global_only)
    to_delete = [
        it["id"]
        for it in items
        if (it.get("metadata") or {}).get("importance", 3) <= importance_below
    ][:limit]
    if not to_delete:
        return 0
    delete_memories(to_delete)
    return len(to_delete)


def prune_neglected(
    project: str | None = None,
    importance_below: int = 2,
    older_than_days: int = 365,
    limit: int = 100,
    global_only: bool = False,
) -> int:
    """
    Delete low-importance memories that are old and never cited (no last_accessed_at).
    Safer than prune_low_importance - only removes memories that are likely stale.
    Returns count deleted.
    """
    items = list_memories(project=project, limit=2000, global_only=global_only)
    cutoff = int(time.time()) - (older_than_days * 86400)
    to_delete = []
    for it in items:
        md = it.get("metadata") or {}
        imp = md.get("importance", 3)
        created = md.get("created_at", 0)
        last_acc = md.get("last_accessed_at")
        if imp <= importance_below and created < cutoff and last_acc is None:
            to_delete.append(it["id"])
            if len(to_delete) >= limit:
                break
    if not to_delete:
        return 0
    delete_memories(to_delete)
    return len(to_delete)


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