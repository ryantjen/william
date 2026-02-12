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
    load_projects, save_projects,
    load_chat, save_chat,
    load_ingested, save_ingested
)

from memory import (
    store_memory, store_memories, store_memories_if_unique,
    list_memories, delete_memory, count_memories
)

from ingest_files import file_sha256, ingest_txt, ingest_csv, ingest_pdf

st.set_page_config(page_title="Statistical Research Copilot", layout="wide")
st.title("William")

# ---------- Sidebar: Projects ----------
projects = load_projects()
if "active_project" not in st.session_state:
    st.session_state.active_project = projects[0] if projects else "General"

st.sidebar.header("Projects")

active_project = st.sidebar.selectbox(
    "Active project",
    options=projects,
    index=projects.index(st.session_state.active_project)
    if st.session_state.active_project in projects else 0
)
st.session_state.active_project = active_project

with st.sidebar.expander("➕ Add a new project", expanded=False):
    new_project = st.text_input("Project name", placeholder="e.g., Sleep Deserts, PCA Study, Real Analysis HW")
    if st.button("Create project"):
        name = (new_project or "").strip()
        if name and name not in projects:
            projects.append(name)
            save_projects(projects)
            st.session_state.active_project = name
            st.success(f"Created project: {name}")
            st.rerun()
        elif name in projects:
            st.info("That project already exists.")
        else:
            st.warning("Enter a project name.")

st.sidebar.caption("Project name is used as the `project` tag in memory metadata.")

if st.sidebar.button("Clear chat for this project"):
    save_chat(active_project, [])
    st.session_state.chat = []
    st.success("Cleared chat UI history for this project.")
    st.rerun()

# ---------- Tabs ----------
tab_chat, tab_files, tab_add_memory, tab_memory = st.tabs(["💬 Chat", "📁 Files", "➕ Add Memory", "🧠 Memory Dashboard"])

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
                response_for_history = ask_agent(user_text, project=active_project, chat_history=history_for_context)
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

    # Text input
    input_text = st.text_area(
        "Text to extract memories from:",
        height=300,
        placeholder="Paste your notes, formulas, theorems, or any content here...\n\nExample:\nThe Central Limit Theorem states that for a random sample of n observations from any population with mean μ and variance σ², the sampling distribution of the sample mean approaches a normal distribution as n → ∞."
    )

    # Extract button
    if st.button("🧠 Extract & Store Memories", type="primary", disabled=not input_text.strip()):
        with st.spinner("Analyzing text and extracting memories..."):
            count = extract_memories_from_text(input_text.strip(), project=target_project)
        
        if count > 0:
            st.success(f"Successfully extracted and stored {count} memory(s)!")
            st.balloons()
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
            ["(all)", "user_preference", "agent_trait", "pdf_chunk", "txt_chunk", "csv_chunk", "simulation", "theorem", "insight", "assumption", "decision", "result", "reference", "formula", "methodology"]
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
                # Show full content with proper markdown/LaTeX rendering
                st.markdown(convert_latex_for_streamlit(text))
                
                st.divider()
                
                # Metadata and actions
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑️ Delete", key=f"del_{it['id']}"):
                        delete_memory(it["id"])
                        st.success("Deleted.")
                        st.rerun()
                with col2:
                    st.caption(f"id: {it['id'][:12]}... · created: {created}")
                
                with st.expander("View metadata", expanded=False):
                    st.json(md)
