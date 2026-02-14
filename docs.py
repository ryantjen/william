"""
Documentation content for each tab. Use in help popovers and the Docs tab.
"""

# Short summaries for help popovers (one per tab)
CHAT_HELP = """
**Chat** — Conversational AI with your project context. William uses retrieved memories to inform responses.

- **run:** — Execute Python code directly.
- **nlrun:** — Natural language → code generation and execution.
- **papers:** — Search academic papers (Semantic Scholar).
- **Save to Memory** — Extract and store insights from assistant replies.
"""
FILES_HELP = """
**File ingestion** — Upload PDF, TXT, or CSV to add content to memory.

- Files are chunked and embedded via OpenAI.
- Duplicates are skipped (similarity threshold).
- Store as project-specific or global (shared across projects).
"""
ADD_MEMORY_HELP = """
**Add Memory from Text** — Paste notes, formulas, excerpts. The agent extracts and stores structured memories.

- Assign type (definition, theorem, formula, etc.), importance (1–5), and confidence.
- Duplicate detection prevents near-identical memories.
"""
MEMORY_HELP = """
**Memory Dashboard** — Browse, search, filter, and manage your memories.

- Filter by type, importance, date.
- Prune low-importance or neglected memories.
- See the **📖 Docs** button for full memory system documentation (storage, retrieval, decay, priority).
"""
MEMORY_MAP_HELP = """
**Memory Map** — Visualize memories as a 2D scatter plot by semantic similarity.

- UMAP reduces embedding dimensions. Closer points = more related.
- Color by type, source, or importance.
- Click points for details.
"""
CALENDAR_HELP = """
**Calendar** — Tasks (with hours), events, and daily work chunks.

- Add tasks/events via natural language. One chunk per task per day.
- Mark chunks done, set expected time remaining, or mark entire task complete.
- Past-due tasks and past events auto-delete.
"""
REVIEW_HELP = """
**Review Mode** — Flashcard-style review of memories.

- Load cards by type (definition, theorem, formula, etc.).
- Flip to reveal, mark correct or review again.
"""

# Full Memory system documentation
MEMORY_FULL_DOC = """
# Memory System: Full Technical Documentation

## Architecture

William uses **ChromaDB** (persistent, on disk at `./chroma_storage`) with a single collection `research_memory`. Each memory is:

1. **Text** — The actual content (e.g. "Central Limit Theorem: ...").
2. **Embedding** — A vector from OpenAI's embedding model (`config.EMBEDDING_MODEL`, typically `text-embedding-3-small`). ChromaDB computes this automatically via `OpenAIEmbeddingFunction`.
3. **Metadata** — A JSON object with: `project`, `type`, `importance`, `confidence`, `source`, `created_at`, `last_accessed_at`, etc.

ChromaDB stores vectors and runs similarity search (cosine distance). William adds a custom re-ranking layer on top.

---

## Storage: How Memories Are Stored

### ID Generation

Each memory gets a unique ID: `SHA256(project|salt|text)`.

- **project** — The project name (or empty for global).
- **salt** — For single store: `created_at`. For batch: `{timestamp}:{index}` so each item in a batch gets a different ID.
- **text** — The memory content.

This makes IDs deterministic: storing the same text twice in the same project with the same salt would yield the same ID (idempotent for batch inserts).

### Metadata Fields

| Field | Purpose |
|-------|---------|
| `project` | Which project the memory belongs to. Empty = global (shared). |
| `type` | e.g. definition, theorem, formula, function, example, insight, summary, pdf_chunk. |
| `importance` | 1–5. Affects retrieval ranking and pruning. Default 3. |
| `confidence` | 0–100. Set by extractor, not used in ranking. |
| `source` | e.g. chat, add_memory, pdf, txt, csv. |
| `created_at` | Unix timestamp when stored. |
| `last_accessed_at` | Unix timestamp when the memory was last cited. `None` if never cited. |
| `name` | Short label for display. |

### Deduplication (store_memory_if_unique)

Before storing, `is_duplicate()` runs:

1. ChromaDB queries for the 1 most similar memory in that project.
2. ChromaDB returns a **cosine distance** (0 = identical, 2 = opposite).
3. **Similarity** = `1 - (distance / 2)`, so 0–1 scale.
4. If similarity ≥ 0.85 (default), the memory is considered a duplicate and **not stored**.

So near-duplicate text (e.g. slightly rephrased) is skipped. Exact threshold is configurable.

### Batch vs Single Store

- **store_memory** — Stores one memory, no dedup check. Returns ID.
- **store_memory_if_unique** — Checks `is_duplicate`, stores only if new. Returns ID or `None`.
- **store_memories** — Batch store many, no dedup.
- **store_memories_if_unique** — Batch with dedup per item. Returns `(ids_stored, duplicates_skipped)`.

---

## Retrieval: Step-by-Step

### 1. Over-Fetch

When you ask for `n_results` (e.g. 4), William asks ChromaDB for **2× that many** (capped at 50):

- `fetch_n = min(n_results * 2, 50)`

So for `n_results=4`, ChromaDB returns 8 candidates. This gives re-ranking room to promote recency/importance.

### 2. ChromaDB Query

ChromaDB runs a **vector similarity search**:

- Your query text is embedded with the same OpenAI model.
- ChromaDB finds the `fetch_n` nearest neighbors by **cosine distance**.
- Returns: `documents`, `metadatas`, `ids`, `distances`.

### 3. Re-Ranking Score (Per Candidate)

For each candidate, a score is computed:

```
similarity     = max(0, 1 - (cosine_distance / 2))   # 0–1, higher = more relevant
age_days       = (now - last_accessed_or_created) / 86400
recency_bonus  = exp(-age_days / 60)                 # 60-day half-life
importance_bonus = (importance - 1) / 4              # importance 1–5 → 0–1

score = similarity + 0.15 * recency_bonus + 0.08 * importance_bonus
```

**Constants** (in `memory.py`):

- `_OVER_FETCH = 2`
- `_RECENCY_HALF_LIFE_DAYS = 60`
- `_RECENCY_WEIGHT = 0.15`
- `_IMPORTANCE_WEIGHT = 0.08`

### 4. Age and Recency

**Age** is based on:

- `last_accessed_at` if the memory was ever cited (via `touch_memories`).
- Otherwise `created_at`.

So a memory cited yesterday is "fresh" (high recency_bonus). A memory created 2 years ago and never cited is "stale" (low recency_bonus).

**Recency bonus** uses exponential decay:

- `recency_bonus = exp(-age_days / 60)`
- At 0 days: 1.0
- At 60 days: ~0.37 (half-life)
- At 120 days: ~0.14
- At 365 days: ~0.002

### 5. Importance Bonus

- Importance 1 → 0
- Importance 2 → 0.25
- Importance 3 → 0.5 (default)
- Importance 4 → 0.75
- Importance 5 → 1.0

This is a small additive bonus (× 0.08), so similarity dominates. Importance nudges high-value memories up.

### 6. Final Output

Candidates are sorted by `score` descending. The top `n_results` are returned as a list of dicts with `id`, `text`, `name`, `type`, `source`, `importance`, `confidence`, `created_at`.

---

## How the Chat Agent Uses Retrieval

When you send a message, the agent fetches memories in **parallel** (ThreadPoolExecutor):

1. **Project memories** — `retrieve_memory_with_metadata(query, project, 8)` — 8 results from your project.
2. **Global memories** — `retrieve_memory_with_metadata(query, None, 4)` — 4 results from global (no project).
3. **User preferences** — `retrieve_memory_with_metadata("user preferences communication style learning", None, 5)`.
4. **Agent traits** — `retrieve_memory_with_metadata("agent personality traits approach style", None, 5)`.

These are merged into the system prompt. When the assistant replies and cites memories, `touch_memories(cited_ids)` runs in a **background thread** so it doesn’t add latency. That updates `last_accessed_at` for each cited memory.

---

## Last-Accessed Tracking (touch_memories)

ChromaDB doesn’t support in-place metadata updates. So `touch_memories`:

1. Fetches each memory by ID (documents + metadatas).
2. Sets `meta["last_accessed_at"] = now`.
3. Deletes the old record and re-adds with the new metadata (same ID, same text).

This is done in a daemon thread so the user doesn’t wait. If it fails, the main flow is unaffected.

---

## Pruning

### prune_low_importance

Deletes memories with `importance ≤ threshold` (default 2). No age or access check. Use with caution.

### prune_neglected

Safer option. Deletes only if **all** of:

- `importance ≤ threshold` (default 2)
- `created_at` older than `older_than_days` (default 365)
- `last_accessed_at` is `None` (never cited)

So it only removes low-importance, old, never-used memories.

---

## Other Functions

| Function | Purpose |
|----------|---------|
| `list_memories` | Fetch by project + filters (type, etc.). No semantic search. Sorted by created_at desc. |
| `get_memory_by_id` | Fetch one memory by ID. |
| `update_memory` | Change text and/or metadata. Delete + re-add (ChromaDB limitation). |
| `delete_memory` / `delete_memories` | Remove by ID. |
| `delete_project_memories` | Remove all memories for a project. |
| `merge_projects` | Move memories from source project to target. |
| `count_memories` | Count by project and filters. |
| `get_memory_stats` | Total count, disk usage, embedding model. |
| `get_all_embeddings` | Fetch all embeddings for a project (for Memory Map UMAP). |
"""
