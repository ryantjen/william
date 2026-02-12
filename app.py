# app.py
import re
import time
import streamlit as st
import pandas as pd

from agent import ask_agent, execute_natural_language, extract_memories_from_exchange, extract_memories_from_text
from tools import run_python


def convert_latex_for_streamlit(text: str) -> str:
    """
    Convert various LaTeX delimiters to Streamlit-compatible format.
    Streamlit's st.markdown() only renders $...$ (inline) and $$...$$ (block).
    """
    # Convert \[...\] to $$...$$  (display/block math)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # Convert \(...\) to $...$ (inline math)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    return text

from storage import (
    load_projects, save_projects, get_project_names, get_project_goal, set_project_goal,
    load_chat, save_chat,
    load_ingested, save_ingested
)

from memory import (
    store_memory, store_memories, store_memories_if_unique,
    list_memories, delete_memory, update_memory, get_memory_by_id,
    delete_project_memories, merge_projects,
    count_memories, get_all_embeddings
)

from ingest_files import file_sha256, ingest_txt, ingest_csv, ingest_pdf

st.set_page_config(page_title="Statistical Research Copilot", layout="wide")
st.title("William")

# ---------- Sidebar: Projects ----------
projects = load_projects()  # List of {"name": ..., "goal": ...}
project_names = get_project_names(projects)

if "active_project" not in st.session_state:
    st.session_state.active_project = project_names[0] if project_names else "General"

st.sidebar.header("Projects")

active_project = st.sidebar.selectbox(
    "Active project",
    options=project_names,
    index=project_names.index(st.session_state.active_project)
    if st.session_state.active_project in project_names else 0
)
st.session_state.active_project = active_project

# Show and edit project goal
current_goal = get_project_goal(projects, active_project)
with st.sidebar.expander("🎯 Project Goal", expanded=bool(current_goal)):
    goal_input = st.text_area(
        "Goal (optional)",
        value=current_goal or "",
        placeholder="e.g., Build a CNN classifier for cats vs dogs, Learn Real Analysis Chapter 3...",
        height=80,
        key="goal_input"
    )
    if st.button("Save Goal"):
        projects = set_project_goal(projects, active_project, goal_input)
        save_projects(projects)
        st.success("Goal saved!")
        st.rerun()

with st.sidebar.expander("➕ Add a new project", expanded=False):
    new_project = st.text_input("Project name", placeholder="e.g., Sleep Deserts, PCA Study, Real Analysis HW")
    new_project_goal = st.text_area("Goal (optional)", placeholder="What do you want to achieve?", height=60, key="new_project_goal")
    if st.button("Create project"):
        name = (new_project or "").strip()
        if name and name not in project_names:
            projects.append({"name": name, "goal": new_project_goal.strip() if new_project_goal else None})
            save_projects(projects)
            st.session_state.active_project = name
            st.success(f"Created project: {name}")
            st.rerun()
        elif name in project_names:
            st.info("That project already exists.")
        else:
            st.warning("Enter a project name.")

with st.sidebar.expander("🗑️ Delete project", expanded=False):
    if len(project_names) <= 1:
        st.warning("Cannot delete the only project.")
    else:
        project_to_delete = st.selectbox(
            "Select project to delete",
            options=[p for p in project_names if p != "General"],  # Protect General
            key="delete_project_select"
        )
        
        mem_count = count_memories(project=project_to_delete) if project_to_delete else 0
        st.caption(f"This project has {mem_count} memories.")
        
        confirm_delete = st.checkbox(f"I understand this will permanently delete all {mem_count} memories", key="confirm_delete")
        
        if st.button("🗑️ Delete Project", type="primary", disabled=not confirm_delete):
            if project_to_delete and project_to_delete != "General":
                # Delete all memories for this project
                deleted_count = delete_project_memories(project_to_delete)
                
                # Remove from projects list
                projects = [p for p in projects if p["name"] != project_to_delete]
                save_projects(projects)
                
                # Clear chat history file
                save_chat(project_to_delete, [])
                
                # Switch to another project
                new_names = get_project_names(projects)
                st.session_state.active_project = new_names[0] if new_names else "General"
                
                st.success(f"Deleted project '{project_to_delete}' and {deleted_count} memories.")
                st.rerun()

with st.sidebar.expander("🔀 Merge projects", expanded=False):
    if len(project_names) < 2:
        st.warning("Need at least 2 projects to merge.")
    else:
        st.caption("Move all memories from source → target project.")
        
        source_project = st.selectbox(
            "Source project (will be deleted)",
            options=[p for p in project_names if p != "General"],
            key="merge_source"
        )
        
        target_options = [p for p in project_names if p != source_project]
        target_project = st.selectbox(
            "Target project (keeps memories)",
            options=target_options,
            key="merge_target"
        )
        
        source_count = count_memories(project=source_project) if source_project else 0
        st.caption(f"Will merge {source_count} memories from '{source_project}' into '{target_project}'.")
        
        confirm_merge = st.checkbox(f"I understand '{source_project}' will be deleted after merge", key="confirm_merge")
        
        if st.button("🔀 Merge Projects", type="primary", disabled=not confirm_merge or not source_project or not target_project):
            # Merge memories
            merged_count = merge_projects(source_project, target_project)
            
            # Remove source from projects list
            projects = [p for p in projects if p["name"] != source_project]
            save_projects(projects)
            
            # Clear source chat history
            save_chat(source_project, [])
            
            # Switch to target if we were on source
            if st.session_state.active_project == source_project:
                st.session_state.active_project = target_project
            
            st.success(f"Merged {merged_count} memories from '{source_project}' into '{target_project}'.")
            st.rerun()

st.sidebar.caption("Project name is used as the `project` tag in memory metadata.")

if st.sidebar.button("Clear chat for this project"):
    save_chat(active_project, [])
    st.session_state.chat = []
    st.success("Cleared chat UI history for this project.")
    st.rerun()

# ---------- Tabs ----------
tab_chat, tab_files, tab_add_memory, tab_memory, tab_map = st.tabs(["💬 Chat", "📁 Files", "➕ Add Memory", "🧠 Memory Dashboard", "🗺️ Memory Map"])

# ---------- Load chat for active project ----------
if "chat" not in st.session_state or st.session_state.get("chat_project") != active_project:
    st.session_state.chat = load_chat(active_project)
    st.session_state.chat_project = active_project

# =========================
# TAB: CHAT
# =========================
with tab_chat:
    st.subheader(f"Chat — {active_project}")

    # Render chat history + Save to Memory button for assistant messages
    for i, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            st.markdown(convert_latex_for_streamlit(msg["content"]))

            if msg["role"] == "assistant":
                # Find the preceding user message for this exchange
                user_input = ""
                if i > 0 and st.session_state.chat[i-1]["role"] == "user":
                    user_input = st.session_state.chat[i-1]["content"]
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("💾 Save to Memory", key=f"save_mem_{active_project}_{i}"):
                        with st.spinner("Extracting memories..."):
                            count = extract_memories_from_exchange(
                                user_input=user_input,
                                answer=msg["content"],
                                project=active_project
                            )
                        if count > 0:
                            st.success(f"Saved {count} memory(s).")
                        else:
                            st.info("No new memories extracted (may be duplicates or trivial).")

    # Display any pending figures from run:/nlrun: (saved before rerun)
    if st.session_state.get("pending_figs"):
        st.markdown("**Plots:**")
        for fig in st.session_state["pending_figs"]:
            st.pyplot(fig)
        # Clear after displaying
        del st.session_state["pending_figs"]

    user_text = st.chat_input("Use 'run:' for Python, 'nlrun:' for natural language → Python.")

    if user_text:
        # Save user message
        st.session_state.chat.append({"role": "user", "content": user_text})
        save_chat(active_project, st.session_state.chat)

        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_text)

        # Produce assistant response
        with st.chat_message("assistant"):

            # 1) Direct Python
            if user_text.startswith("run:"):
                code = user_text[len("run:"):].strip()
                out, figs = run_python(code)

                st.markdown("**Ran code:**")
                st.code(code, language="python")

                if out:
                    st.markdown("**Output:**")
                    st.code(out, language="text")

                if figs:
                    st.markdown("**Plots:**")
                    for fig in figs:
                        st.pyplot(fig)
                    # Save figures to session state for redisplay after rerun
                    st.session_state["pending_figs"] = figs

                # Include code in history so it displays on reload
                response_for_history = f"**Ran code:**\n```python\n{code}\n```\n\n**Output:**\n```\n{out}\n```" if out else f"**Ran code:**\n```python\n{code}\n```\n\n(no text output)"

            # 2) Natural language -> Python -> run
            elif user_text.startswith("nlrun:"):
                task = user_text[len("nlrun:"):].strip()

                # generate code
                _, code = execute_natural_language(task, project=active_project)

                # run it in-process so plots render
                run_out, figs = run_python(code)

                st.markdown("**Generated code:**")
                st.code(code, language="python")

                if run_out:
                    st.markdown("**Output:**")
                    st.code(run_out, language="text")

                if figs:
                    st.markdown("**Plots:**")
                    for fig in figs:
                        st.pyplot(fig)
                    # Save figures to session state for redisplay after rerun
                    st.session_state["pending_figs"] = figs

                # Include code in history so it displays on reload
                response_for_history = f"**Generated code:**\n```python\n{code}\n```\n\n**Output:**\n```\n{run_out}\n```"

            # 3) Normal chat
            else:
                # Pass chat history for context (exclude the just-added user message)
                history_for_context = st.session_state.chat[:-1]
                project_goal = get_project_goal(projects, active_project)
                response_for_history = ask_agent(user_text, project=active_project, chat_history=history_for_context, project_goal=project_goal)
                st.markdown(convert_latex_for_streamlit(response_for_history))

        # Save assistant message
        st.session_state.chat.append({"role": "assistant", "content": response_for_history})
        save_chat(active_project, st.session_state.chat)

        st.rerun()

# =========================
# TAB: FILES
# =========================
with tab_files:
    st.subheader(f"File ingestion — {active_project}")
    st.caption("Uploads are chunked and stored in Chroma with your project tag. Text-only PDFs supported (no OCR yet).")

    ingested = load_ingested()
    ingested.setdefault(active_project, {})

    uploaded = st.file_uploader("Upload PDF/TXT/CSV", type=["pdf", "txt", "csv"])

    max_pages = st.slider("PDF max pages (safety cap)", 5, 200, 60, 5)

    store_as_global = st.checkbox("Store as global reference (no project tag)", value=False)
    target_project = None if store_as_global else active_project

    if uploaded:
        data = uploaded.read()
        h = file_sha256(data)

        st.write(f"**Selected:** {uploaded.name}")
        if h in ingested[active_project]:
            st.warning("This file was already ingested for this project (hash match).")
            st.json(ingested[active_project][h])

        if st.button("Ingest into memory"):
            try:
                with st.spinner("Extracting + chunking + storing..."):
                    name = uploaded.name
                    lower = name.lower()
                    n_chunks = 0
                    n_duplicates = 0

                    if lower.endswith(".txt"):
                        chunks, metas = ingest_txt(data, name)
                        ids_stored, n_duplicates = store_memories_if_unique(chunks, metas, project=target_project)
                        n_chunks = len(ids_stored)

                    elif lower.endswith(".csv"):
                        chunks, metas, df = ingest_csv(data, name)
                        ids_stored, n_duplicates = store_memories_if_unique(chunks, metas, project=target_project)
                        n_chunks = len(ids_stored)
                        st.dataframe(df.head(20))

                    elif lower.endswith(".pdf"):
                        chunks, metas, pages_with_text = ingest_pdf(data, name)
                        if pages_with_text == 0:
                            st.warning("No extractable text found. This PDF may be scanned or image-based.")
                        else:
                            ids_stored, n_duplicates = store_memories_if_unique(chunks, metas, project=target_project)
                            n_chunks = len(ids_stored)
                            dup_msg = f" ({n_duplicates} duplicates skipped)" if n_duplicates > 0 else ""
                            st.success(f"Ingested PDF: {pages_with_text} pages with text, {n_chunks} chunks stored{dup_msg}.")

                    else:
                        st.error("Unsupported file type.")
                        n_chunks = 0

                    # record ingestion metadata
                    ingested[active_project][h] = {
                        "name": name,
                        "ts": int(time.time()),
                        "chunks": n_chunks,
                        "duplicates_skipped": n_duplicates,
                        "stored_as": "global" if store_as_global else "project"
                    }
                    save_ingested(ingested)

                    # Show success message for non-PDF files (PDF has its own message above)
                    if n_chunks > 0 and not lower.endswith(".pdf"):
                        dup_msg = f" ({n_duplicates} duplicates skipped)" if n_duplicates > 0 else ""
                        st.success(f"Ingested {name}: {n_chunks} chunks saved{dup_msg}.")

            except Exception as e:
                st.error("Ingestion failed.")
                st.exception(e)

    st.markdown("### Ingested files registry (this project)")
    if ingested.get(active_project):
        st.json(ingested[active_project])
    else:
        st.info("No files ingested yet for this project.")

# =========================
# TAB: ADD MEMORY
# =========================
with tab_add_memory:
    st.subheader("Add Memory from Text")
    st.caption("Paste notes, formulas, excerpts, or any content. The agent will extract and store relevant memories.")

    # Project selection
    col1, col2 = st.columns([2, 1])
    with col1:
        store_to_options = [active_project, "(Global - all projects)"]
        store_to = st.selectbox("Store memories to:", store_to_options)
    
    target_project = None if store_to == "(Global - all projects)" else active_project

    # Show success message from previous extraction (if any)
    if st.session_state.get("add_memory_success"):
        count = st.session_state.add_memory_success
        st.success(f"Successfully extracted and stored {count} memory(s)!")
        st.balloons()
        st.session_state.add_memory_success = None  # Clear after showing

    # Check if we need to clear the text (must happen BEFORE widget creation)
    default_text = ""
    if st.session_state.get("clear_add_memory_text"):
        default_text = ""
        st.session_state.clear_add_memory_text = False
    elif "add_memory_text_value" in st.session_state:
        default_text = st.session_state.add_memory_text_value

    input_text = st.text_area(
        "Text to extract memories from:",
        value=default_text,
        height=300,
        placeholder="Paste your notes, formulas, theorems, or any content here...\n\nExample:\nThe Central Limit Theorem states that for a random sample of n observations from any population with mean μ and variance σ², the sampling distribution of the sample mean approaches a normal distribution as n → ∞.",
    )
    
    # Store current value for persistence
    st.session_state.add_memory_text_value = input_text

    # Extract button
    if st.button("🧠 Extract & Store Memories", type="primary", disabled=not input_text.strip()):
        with st.spinner("Analyzing text and extracting memories..."):
            count = extract_memories_from_text(input_text.strip(), project=target_project)
        
        if count > 0:
            # Store success info and set clear flag, then rerun
            st.session_state.add_memory_success = count
            st.session_state.clear_add_memory_text = True
            st.session_state.add_memory_text_value = ""
            st.rerun()
        else:
            st.warning("No memories extracted. The text may not contain notable information, or similar memories already exist.")

    st.divider()
    st.markdown("### Tips for best results")
    st.markdown("""
    - **Formulas**: Include the formula name and the equation. Use standard notation.
    - **Theorems**: State the theorem name, conditions, and conclusion.
    - **Insights**: Be specific about what you learned and why it matters.
    - **References**: Include author, title, and key points.
    """)

# =========================
# TAB: MEMORY DASHBOARD
# =========================
with tab_memory:
    st.subheader(f"Memory Dashboard — {active_project}")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        show_project_only = st.checkbox("Show only this project", value=True)
    with colB:
        mem_type = st.selectbox(
            "Type filter",
            ["(all)", "definition", "theorem", "formula", "function", "example", "insight", "assumption", "decision", "result", "reference", "methodology", "user_preference", "agent_trait", "pdf_chunk", "txt_chunk", "csv_chunk", "simulation"]
        )
    with colC:
        date_filter = st.selectbox(
            "Time range",
            ["All time", "Last 7 days", "Last 30 days", "Last 90 days", "Last year"]
        )
    with colD:
        limit = st.slider("Items to show", 10, 500, 100, 10)

    # Calculate cutoff timestamp for date filter
    date_cutoffs = {
        "All time": 0,
        "Last 7 days": int(time.time()) - 7 * 24 * 60 * 60,
        "Last 30 days": int(time.time()) - 30 * 24 * 60 * 60,
        "Last 90 days": int(time.time()) - 90 * 24 * 60 * 60,
        "Last year": int(time.time()) - 365 * 24 * 60 * 60,
    }
    date_cutoff = date_cutoffs.get(date_filter, 0)

    where_extra = {}
    if mem_type != "(all)":
        where_extra["type"] = mem_type

    proj = active_project if show_project_only else None
    total = count_memories(project=proj, where_extra=where_extra if where_extra else None)
    st.caption(f"Total memories (before date filter): {total}")

    query = st.text_input(
        "Search memories",
        "",
        placeholder="e.g., theorem, CLT, regression, pdf_chunk..."
    )
    st.caption("Searches text content, type, name, and source. Try: theorem, reference, insight, simulation, etc.")

    items = list_memories(project=proj, where_extra=where_extra if where_extra else None, limit=limit)

    # Apply date filter
    if date_cutoff > 0:
        items = [it for it in items if (it["metadata"] or {}).get("created_at", 0) >= date_cutoff]

    if query.strip():
        q = query.strip().lower()
        # Search in text AND metadata fields (type, name, source)
        def matches(it):
            text = (it["text"] or "").lower()
            md = it["metadata"] or {}
            mtype = (md.get("type") or "").lower()
            name = (md.get("name") or "").lower()
            source = (md.get("source") or "").lower()
            return q in text or q in mtype or q in name or q in source
        items = [it for it in items if matches(it)]
    
    st.caption(f"Showing {len(items)} memories")

    if not items:
        st.info("No memories found with current filters.")
    else:
        for it in items:
            md = it["metadata"] or {}
            created = md.get("created_at", "")
            mtype = md.get("type", "(no type)")
            name = md.get("name", "")
            text = it["text"] or ""

            # Generate a short preview title
            # Use first line or first 80 chars, whichever is shorter
            first_line = text.split("\n")[0].strip()
            preview = first_line[:80] + ("..." if len(first_line) > 80 else "")
            
            # Build expander title
            if name:
                title = f"📄 **{mtype}** — {name}"
            elif preview:
                title = f"📄 **{mtype}** — {preview}"
            else:
                title = f"📄 **{mtype}**"

            with st.expander(title, expanded=False):
                mem_id = it["id"]
                edit_key = f"editing_{mem_id}"
                
                # Check if we're in edit mode for this memory
                if st.session_state.get(edit_key, False):
                    # Edit mode
                    st.markdown("**Editing memory:**")
                    edited_text = st.text_area(
                        "Content",
                        value=text,
                        height=200,
                        key=f"edit_text_{mem_id}"
                    )
                    
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("💾 Save", key=f"save_{mem_id}"):
                            update_memory(mem_id, edited_text)
                            st.session_state[edit_key] = False
                            st.success("Memory updated!")
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_{mem_id}"):
                            st.session_state[edit_key] = False
                            st.rerun()
                else:
                    # View mode
                    st.markdown(convert_latex_for_streamlit(text))
                    
                    st.divider()
                    
                    # Metadata and actions
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("✏️ Edit", key=f"edit_{mem_id}"):
                            st.session_state[edit_key] = True
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{mem_id}"):
                            delete_memory(mem_id)
                            st.success("Deleted.")
                            st.rerun()
                    with col3:
                        edited_time = md.get("last_edited")
                        edit_info = f" · edited: {time.strftime('%Y-%m-%d', time.localtime(edited_time))}" if edited_time else ""
                        st.caption(f"id: {mem_id[:12]}... · created: {created}{edit_info}")
                    
                    with st.expander("View metadata", expanded=False):
                        st.json(md)

# =========================
# TAB: MEMORY MAP
# =========================
with tab_map:
    st.subheader("Memory Map")
    st.caption("Visualize your memories as a 2D map based on semantic similarity. Closer points = more related concepts. Click a point to see details.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        map_scope = st.radio("Scope", ["This project only", "All memories"], horizontal=True)
    with col2:
        color_by = st.selectbox("Color by", ["type", "source", "importance"])
    with col3:
        spread_factor = st.slider("Spread factor", 1.0, 3.0, 1.5, 0.1, help="Exponentially spread apart dissimilar memories")

    connection_threshold = st.slider("Connection distance", 0.0, 1.0, 0.3, 0.05, help="Draw lines between memories within this distance (0 = no lines)")

    map_project = active_project if map_scope == "This project only" else None

    # Initialize session state for map data
    if "map_data" not in st.session_state:
        st.session_state.map_data = None
    if "map_fig" not in st.session_state:
        st.session_state.map_fig = None

    if st.button("🗺️ Generate Memory Map", type="primary"):
        with st.spinner("Fetching embeddings and computing layout..."):
            data = get_all_embeddings(project=map_project)
            
            embeddings = data["embeddings"]
            texts = data["texts"]
            metadatas = data["metadatas"]
            
            if embeddings is None or len(embeddings) < 3:
                st.warning("Need at least 3 memories to generate a map. Add more memories first!")
            else:
                import numpy as np
                import plotly.graph_objects as go
                
                try:
                    from umap import UMAP
                except ImportError:
                    st.error("UMAP not installed. Run: pip install umap-learn")
                    st.stop()
                
                # Convert to numpy array
                embeddings_array = np.array(embeddings)
                
                # Reduce dimensions with UMAP
                n_neighbors = min(15, len(embeddings) - 1)
                reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', random_state=42)
                coords_2d = reducer.fit_transform(embeddings_array)
                
                # Apply exponential spread to push dissimilar points further apart
                # Center the coordinates first
                center = coords_2d.mean(axis=0)
                centered = coords_2d - center
                
                # Apply exponential transformation based on distance from center
                distances = np.linalg.norm(centered, axis=1, keepdims=True)
                max_dist = distances.max() if distances.max() > 0 else 1
                normalized_dist = distances / max_dist
                
                # Exponential spread: points further from center get pushed even further
                spread_multiplier = np.power(normalized_dist + 0.5, spread_factor)
                coords_spread = centered * spread_multiplier + center
                
                # Build color map for types
                type_colors = {}
                color_palette = [
                    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
                ]
                
                # Prepare data for plotting
                plot_data = []
                for i, (x, y) in enumerate(coords_spread):
                    md = metadatas[i] or {}
                    name = md.get("name", "")
                    mem_type = md.get("type", "unknown")
                    source = md.get("source", "unknown")
                    importance = md.get("importance", 3)
                    full_text = texts[i] or ""
                    text_preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
                    
                    # Clean text preview for hover (keep full_text intact for display)
                    text_preview_clean = text_preview.replace("\n", " ").replace("$", "")
                    
                    # Assign colors
                    color_key = mem_type if color_by == "type" else (source if color_by == "source" else str(importance))
                    if color_key not in type_colors:
                        type_colors[color_key] = color_palette[len(type_colors) % len(color_palette)]
                    
                    plot_data.append({
                        "x": x,
                        "y": y,
                        "name": name or text_preview_clean[:40],
                        "type": mem_type,
                        "source": source,
                        "importance": str(importance),
                        "preview": text_preview_clean,
                        "full_text": full_text,  # Keep full text for click expansion
                        "color_key": color_key,
                        "index": i  # Track original index
                    })
                
                # Create figure
                fig = go.Figure()
                
                # Add connection lines between nearby memories
                if connection_threshold > 0:
                    # Calculate pairwise distances in 2D space
                    edge_x = []
                    edge_y = []
                    
                    # Normalize coordinates to 0-1 range for threshold comparison
                    xs = np.array([p["x"] for p in plot_data])
                    ys = np.array([p["y"] for p in plot_data])
                    x_range = xs.max() - xs.min() if xs.max() != xs.min() else 1
                    y_range = ys.max() - ys.min() if ys.max() != ys.min() else 1
                    scale = max(x_range, y_range)
                    
                    for i in range(len(plot_data)):
                        for j in range(i + 1, len(plot_data)):
                            dx = (plot_data[i]["x"] - plot_data[j]["x"]) / scale
                            dy = (plot_data[i]["y"] - plot_data[j]["y"]) / scale
                            dist = np.sqrt(dx*dx + dy*dy)
                            
                            if dist < connection_threshold:
                                edge_x.extend([plot_data[i]["x"], plot_data[j]["x"], None])
                                edge_y.extend([plot_data[i]["y"], plot_data[j]["y"], None])
                    
                    if edge_x:
                        fig.add_trace(go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            mode='lines',
                            line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
                            hoverinfo='none',
                            showlegend=False
                        ))
                
                # Add scatter points grouped by color category
                for color_key, color in type_colors.items():
                    points = [p for p in plot_data if p["color_key"] == color_key]
                    if points:
                        fig.add_trace(go.Scatter(
                            x=[p["x"] for p in points],
                            y=[p["y"] for p in points],
                            mode='markers',
                            name=color_key,
                            marker=dict(
                                size=12,
                                color=color,
                                line=dict(width=1, color='DarkSlateGrey')
                            ),
                            text=[p["name"] for p in points],
                            hovertemplate="<b>%{text}</b><br>Type: " + 
                                         "<br>".join([p["type"] for p in points][:1]) + 
                                         "<extra></extra>",
                            customdata=[[p["type"], p["source"], p["preview"]] for p in points],
                            hoverinfo='text'
                        ))
                
                fig.update_layout(
                    title=f"Memory Map ({len(embeddings)} memories) - Click a point to see details",
                    xaxis_title="",
                    yaxis_title="",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=650,
                    legend=dict(title=color_by.capitalize()),
                    clickmode='event+select'
                )
                
                # Store in session state for persistence
                st.session_state.map_data = plot_data
                st.session_state.map_fig = fig
                st.session_state.map_count = len(embeddings)
                
                st.success(f"Map generated with {len(embeddings)} memories. Click any point to see full details.")
    
    # Display the map if we have one
    if st.session_state.get("map_fig") is not None:
        selection = st.plotly_chart(
            st.session_state.map_fig, 
            use_container_width=True,
            on_select="rerun",
            key="memory_map_chart"
        )
        
        # Handle click selection
        if selection and selection.selection and selection.selection.points:
            selected_point = selection.selection.points[0]
            
            # Find the matching memory from our data
            # The point index within its trace + trace offset
            curve_num = selected_point.get("curve_number", 0)
            point_idx = selected_point.get("point_index", 0)
            
            # Skip if it's the edge trace (curve 0 when edges exist)
            plot_data = st.session_state.map_data
            if plot_data:
                # Find point by coordinates (most reliable)
                sel_x = selected_point.get("x")
                sel_y = selected_point.get("y")
                
                selected_memory = None
                for p in plot_data:
                    if abs(p["x"] - sel_x) < 0.0001 and abs(p["y"] - sel_y) < 0.0001:
                        selected_memory = p
                        break
                
                if selected_memory:
                    st.markdown("---")
                    st.markdown("### Selected Memory")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Name:** {selected_memory['name']}")
                    with col2:
                        st.markdown(f"**Type:** {selected_memory['type']}")
                    with col3:
                        st.markdown(f"**Source:** {selected_memory['source']}")
                    
                    st.markdown("**Full Content:**")
                    # Use the LaTeX converter for proper rendering
                    st.markdown(convert_latex_for_streamlit(selected_memory['full_text']))
        
        # Show cluster summary
        if st.session_state.get("map_data"):
            st.markdown("---")
            st.markdown("### Memory Distribution")
            import pandas as pd
            df = pd.DataFrame(st.session_state.map_data)
            type_counts = df["type"].value_counts()
            st.dataframe(type_counts.reset_index().rename(columns={"index": "Type", "type": "Count"}), hide_index=True)
