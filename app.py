# app.py
import re
import time
import json
import streamlit as st
import pandas as pd

from agent import ask_agent, ask_agent_stream, execute_natural_language, extract_memories_from_exchange, extract_memories_from_text, summarize_conversation, search_papers, format_paper_results, parse_task_from_text, parse_event_from_text
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


STYLE_CONTEXT_TYPES = ("agent_trait", "user_preference", "user_goal", "user_knowledge")


def render_citations(citations: list) -> None:
    """
    Display content sources, excluding style context (agent_trait, user_preference).
    """
    content = [c for c in citations if c.get("type") not in STYLE_CONTEXT_TYPES]
    if not content:
        return
    with st.expander(f"📚 Sources ({len(content)} memories referenced)", expanded=False):
        for c in content:
            raw = c.get("name") or c.get("text", "") or ""
            name = raw[:50] + "..." if len(raw) > 50 else raw
            mtype = c.get("type", "unknown")
            st.markdown(f"- **{mtype}**: {name}")


from storage import (
    load_projects, save_projects, get_project_names, get_project_goal, set_project_goal,
    rename_project as storage_rename_project,
    load_chat, save_chat, delete_chat,
    load_ingested, save_ingested, clear_ingested_for_project, clear_ingested_file
)

from memory import (
    store_memory, store_memories, store_memory_if_unique, store_memories_if_unique,
    list_memories, delete_memory, delete_memories, update_memory, get_memory_by_id,
    delete_project_memories, merge_projects,
    count_memories, get_all_embeddings,
    get_memory_stats, prune_low_importance, prune_neglected
)

from ingest_files import file_sha256, ingest_txt, ingest_csv, ingest_pdf
from docs import (
    CHAT_HELP, FILES_HELP, ADD_MEMORY_HELP, MEMORY_HELP, MEMORY_MAP_HELP,
    CALENDAR_HELP, REVIEW_HELP, MEMORY_FULL_DOC
)
from calendar_data import (
    load_tasks, add_task, get_active_tasks, get_tasks_due_on_date, delete_task,
    load_events, add_event, get_events_for_date, delete_event,
    get_chunks_for_date, get_chunks_for_date_read_only, mark_chunk_complete, mark_task_complete, update_task_remaining_estimate, refresh_chunks_for_date, get_task_completion_pct,
    cleanup_past_due, cleanup_past_events, migrate_from_tasks,
)

st.set_page_config(page_title="William — Learning & Research Copilot", layout="wide")

# Header with Docs button in top-right
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("William")
with header_col2:
    with st.popover("📖 Docs", help="Full documentation"):
        doc_section = st.radio(
            "Section",
            ["Memory System", "Chat", "Files", "Add Memory", "Memory Dashboard", "Memory Map", "Calendar", "Review"],
            key="doc_section"
        )
        if doc_section == "Memory System":
            st.markdown(MEMORY_FULL_DOC)
        elif doc_section == "Chat":
            st.markdown(CHAT_HELP)
        elif doc_section == "Files":
            st.markdown(FILES_HELP)
        elif doc_section == "Add Memory":
            st.markdown(ADD_MEMORY_HELP)
        elif doc_section == "Memory Dashboard":
            st.markdown(MEMORY_HELP)
        elif doc_section == "Memory Map":
            st.markdown(MEMORY_MAP_HELP)
        elif doc_section == "Calendar":
            st.markdown(CALENDAR_HELP)
        else:
            st.markdown(REVIEW_HELP)

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

# Show and edit project goal (key scoped by project so switching projects shows correct goal)
current_goal = get_project_goal(projects, active_project)
with st.sidebar.expander("🎯 Project Goal", expanded=bool(current_goal)):
    goal_input = st.text_area(
        "Goal (optional)",
        value=current_goal or "",
        placeholder="e.g., Build a CNN classifier for cats vs dogs, Learn Real Analysis Chapter 3...",
        height=80,
        key=f"goal_input_{active_project}"
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

with st.sidebar.expander("✏️ Rename project", expanded=False):
    if len(project_names) < 1:
        st.warning("No projects to rename.")
    else:
        project_to_rename = st.selectbox(
            "Select project to rename",
            options=project_names,
            key="rename_project_select"
        )
        new_name = st.text_input(
            "New name",
            placeholder="Enter new project name",
            key="rename_new_name"
        )
        if st.button("Rename project"):
            name = (new_name or "").strip()
            if not name:
                st.warning("Enter a new name.")
            elif name == project_to_rename:
                st.info("New name is the same as current name.")
            elif name in project_names:
                st.info("A project with that name already exists.")
            else:
                projects = storage_rename_project(projects, project_to_rename, name)
                save_projects(projects)
                merge_projects(project_to_rename, name)  # Update ChromaDB memories
                if st.session_state.active_project == project_to_rename:
                    st.session_state.active_project = name
                st.success(f"Renamed '{project_to_rename}' to '{name}'.")
                st.rerun()

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
                
                # Delete chat history file
                delete_chat(project_to_delete)
                
                # Clear ingestion records for this project
                clear_ingested_for_project(project_to_delete)
                
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
            
            # Delete source chat history file
            delete_chat(source_project)
            
            # Switch to target if we were on source
            if st.session_state.active_project == source_project:
                st.session_state.active_project = target_project
            
            st.success(f"Merged {merged_count} memories from '{source_project}' into '{target_project}'.")
            st.rerun()

st.sidebar.caption("Project name is used as the `project` tag in memory metadata.")

col_clear, col_summary = st.sidebar.columns(2)
with col_clear:
    if st.button("🗑️ Clear chat"):
        save_chat(active_project, [])
        st.session_state.chat = []
        st.success("Cleared chat.")
        st.rerun()

with col_summary:
    if st.button("📝 Summarize"):
        if st.session_state.get("chat") and len(st.session_state.chat) >= 2:
            with st.spinner("Summarizing conversation..."):
                summary = summarize_conversation(st.session_state.chat, active_project)
            if summary:
                st.success("Summary saved to memory!")
            else:
                st.warning("Could not generate summary.")
        else:
            st.info("Need at least one exchange to summarize.")

# Chat export options
with st.sidebar.expander("📤 Export chat", expanded=False):
    if st.session_state.get("chat") and len(st.session_state.chat) > 0:
        chat_data = st.session_state.chat
        
        # Markdown export
        chat_md = f"# Chat Export — {active_project}\n"
        chat_md += f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
        for msg in chat_data:
            role = "**User:**" if msg["role"] == "user" else "**Assistant:**"
            chat_md += f"{role}\n\n{msg['content']}\n\n---\n\n"
        
        st.download_button(
            "📥 Markdown",
            data=chat_md,
            file_name=f"chat_{active_project}_{int(time.time())}.md",
            mime="text/markdown",
            key="export_chat_md"
        )
        
        # LaTeX export
        chat_latex = [
            r"\documentclass{article}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{listings}",
            r"\lstset{basicstyle=\ttfamily\small,breaklines=true}",
            r"",
            r"\title{Chat Export — " + active_project + r"}",
            r"\date{" + time.strftime('%Y-%m-%d') + r"}",
            r"\begin{document}",
            r"\maketitle",
            r"",
        ]
        for msg in chat_data:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_latex.append(r"\subsection*{" + role + r"}")
            content = msg["content"]
            # Basic conversions
            content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', content)
            content = re.sub(r'```python\n(.*?)\n```', r'\\begin{lstlisting}[language=Python]\n\1\n\\end{lstlisting}', content, flags=re.DOTALL)
            content = re.sub(r'```\n?(.*?)\n?```', r'\\begin{lstlisting}\n\1\n\\end{lstlisting}', content, flags=re.DOTALL)
            chat_latex.append(content)
            chat_latex.append(r"\hrulefill")
            chat_latex.append("")
        chat_latex.append(r"\end{document}")
        
        st.download_button(
            "📥 LaTeX",
            data="\n".join(chat_latex),
            file_name=f"chat_{active_project}_{int(time.time())}.tex",
            mime="text/x-tex",
            key="export_chat_tex"
        )
    else:
        st.caption("No chat history to export.")

# ---------- Tabs ----------
tab_chat, tab_files, tab_add_memory, tab_memory, tab_map, tab_calendar, tab_review = st.tabs(["💬 Chat", "📁 Files", "➕ Add Memory", "🧠 Memory Dashboard", "🗺️ Memory Map", "📅 Calendar", "🎯 Review"])

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
                # Sources first (where answer came from) - before Save to Memory so it isn't overrun
                citations = msg.get("citations") or []
                valid_citations = [c for c in citations if c.get("name") or c.get("text")]
                if valid_citations:
                    render_citations(valid_citations)

                # Find the preceding user message for this exchange
                user_input = ""
                if i > 0 and st.session_state.chat[i-1]["role"] == "user":
                    user_input = st.session_state.chat[i-1]["content"]
                
                if st.button("💾 Save to Memory", key=f"save_mem_{active_project}_{i}"):
                    with st.spinner("Extracting memories..."):
                        stored, extracted = extract_memories_from_exchange(
                            user_input=user_input,
                            answer=msg["content"],
                            project=active_project,
                            project_goal=current_goal
                        )
                    if stored > 0:
                        st.success(f"Saved {stored} memory(s).")
                    elif extracted > 0:
                        st.info(f"Found {extracted} potential memories, but they were similar to existing ones.")
                    else:
                        st.info("Nothing worth saving was found in this exchange.")

    # Display any pending figures from run:/nlrun: (saved before rerun)
    if st.session_state.get("pending_figs"):
        st.markdown("**Plots:**")
        for fig in st.session_state["pending_figs"]:
            st.pyplot(fig)
        # Clear after displaying
        del st.session_state["pending_figs"]

    user_text = st.chat_input("Use 'run:' for Python, 'nlrun:' for NL→code, 'papers:' to search papers.")

    if user_text:
        # Save user message
        st.session_state.chat.append({"role": "user", "content": user_text})
        save_chat(active_project, st.session_state.chat)

        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_text)

        # Produce assistant response
        valid_citations = []  # filled only for normal chat
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

                # Generate code (no execution yet)
                code = execute_natural_language(task, project=active_project)

                # Check for API errors (returned as comment)
                if code.strip().startswith("# API error:"):
                    st.error(code.strip().replace("# API error: ", ""))
                    response_for_history = f"**Error:** {code.strip()}"
                else:
                    # Run once
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

            # 3) Paper search
            elif user_text.startswith("papers:"):
                query = user_text[len("papers:"):].strip()
                
                with st.spinner(f"Searching papers for: {query}..."):
                    papers = search_papers(query, limit=5)
                
                response_for_history = format_paper_results(papers)
                st.markdown(response_for_history)
                
                # Offer to save paper references to memory
                if papers and "error" not in papers[0]:
                    if st.button("💾 Save these references to memory"):
                        for p in papers:
                            ref_text = f"**{p['title']}**\n\nAuthors: {p['authors']}\nYear: {p['year']}\nCitations: {p['citations']}\nURL: {p['url']}\n\nAbstract: {p['abstract']}"
                            store_memory_if_unique(
                                text=ref_text,
                                metadata={
                                    "name": p['title'][:100],
                                    "type": "reference",
                                    "importance": 3,
                                    "source": "paper_search"
                                },
                                project=active_project
                            )
                        st.success(f"Saved {len(papers)} paper references to memory!")

            # 4) Normal chat (with streaming)
            else:
                history_for_context = st.session_state.chat[:-1]
                project_goal = get_project_goal(projects, active_project)
                result_holder = {}
                stream_gen = ask_agent_stream(
                    user_text, project=active_project,
                    chat_history=history_for_context,
                    project_goal=project_goal,
                    result_holder=result_holder
                )
                # Stream the response; when done, result_holder has answer + citations
                st.write_stream(stream_gen)
                response_for_history = result_holder.get("answer", "")
                citations = result_holder.get("citations", [])
                valid_citations = [c for c in citations if c.get("name") or c.get("text")] if citations else []

                # Display citations if any
                if valid_citations:
                    render_citations(valid_citations)

        # Save assistant message (persist citations so Sources survive rerun)
        save_payload = {"role": "assistant", "content": response_for_history}
        if valid_citations:
            save_payload["citations"] = valid_citations
        st.session_state.chat.append(save_payload)
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
        for file_hash, file_info in list(ingested[active_project].items()):
            file_name = file_info.get("name", "Unknown")
            chunks = file_info.get("chunks", 0)
            ts = file_info.get("ts", 0)
            ts_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts)) if ts else "Unknown"
            
            with st.expander(f"📄 {file_name} ({chunks} chunks)", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"Ingested: {ts_str}")
                    st.caption(f"Hash: {file_hash[:16]}...")
                    st.caption(f"Stored as: {file_info.get('stored_as', 'project')}")
                with col2:
                    if st.button("🗑️ Clear record", key=f"clear_ingested_{file_hash}"):
                        clear_ingested_file(active_project, file_hash)
                        st.success(f"Cleared record for {file_name}")
                        st.rerun()
        
        # Option to clear all
        st.markdown("---")
        if st.button("🗑️ Clear all ingestion records for this project"):
            cleared = clear_ingested_for_project(active_project)
            st.success(f"Cleared {cleared} ingestion records.")
            st.rerun()
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

    # Extract and Clear buttons
    add_memory_goal = get_project_goal(projects, target_project) if target_project else None
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🗑️ Clear", help="Clear the text box", key="add_memory_clear"):
            st.session_state.clear_add_memory_text = True
            st.session_state.add_memory_text_value = ""
            st.rerun()
    with btn_col2:
        if st.button("🧠 Extract & Store Memories", type="primary", disabled=not input_text.strip()):
            with st.spinner("Analyzing text and extracting memories..."):
                count = extract_memories_from_text(
                    input_text.strip(),
                    project=target_project,
                    project_goal=add_memory_goal
                )
            if count > 0:
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
    head_col, storage_col = st.columns([4, 1])
    with head_col:
        st.subheader(f"Memory Dashboard — {active_project}")
    with storage_col:
        with st.popover("Storage", help="Total memories, disk usage, embedding model"):
            stats = get_memory_stats()
            st.metric("Total memories", stats["total_memories"])
            st.metric("Disk usage", stats["disk_display"])
            st.caption("Embedding model")
            st.code(stats["embedding_model"], language=None)

    # Pruning
    with st.expander("✂️ Prune memories", expanded=False):
        prune_col1, prune_col2 = st.columns(2)
        with prune_col1:
            st.markdown("**By importance** — Delete memories with importance ≤ N")
            prune_imp = st.selectbox("Importance ≤", [1, 2], index=1, key="prune_imp")
            prune_scope = st.radio("Scope", ["This project", "Global"], horizontal=True, key="prune_scope")
            prune_limit = st.number_input("Max to delete", min_value=1, max_value=500, value=50, key="prune_limit")
            if st.button("Prune by importance", key="prune_btn"):
                proj_prune = None if prune_scope == "Global" else active_project
                deleted = prune_low_importance(project=proj_prune, importance_below=prune_imp, limit=prune_limit, global_only=(prune_scope == "Global"))
                st.success(f"Deleted {deleted} memory(ies).") if deleted else st.info("No matching memories found.")
                st.rerun()
        with prune_col2:
            st.markdown("**Neglected only** — Low importance + old + never cited")
            prune_neg_days = st.number_input("Older than (days)", min_value=30, max_value=3650, value=365, key="prune_neg_days")
            prune_neg_imp = st.selectbox("Importance ≤", [1, 2], index=1, key="prune_neg_imp")
            prune_neg_limit = st.number_input("Max to delete", min_value=1, max_value=500, value=50, key="prune_neg_limit")
            if st.button("Prune neglected", key="prune_neg_btn"):
                proj_prune = None if prune_scope == "Global" else active_project
                deleted = prune_neglected(project=proj_prune, importance_below=prune_neg_imp, older_than_days=prune_neg_days, limit=prune_neg_limit, global_only=(prune_scope == "Global"))
                st.success(f"Deleted {deleted} memory(ies).") if deleted else st.info("No matching memories found.")
                st.rerun()

    colA, colB, colC, colD, colE = st.columns([1, 1, 1, 1, 1])
    with colA:
        memory_scope = st.radio("Scope", ["This project", "Global"], horizontal=True, key="mem_scope")
    with colB:
        mem_type = st.selectbox(
            "Type filter",
            ["(all)", "definition", "theorem", "formula", "function", "example", "insight", "summary", "assumption", "decision", "result", "reference", "methodology", "user_preference", "agent_trait", "user_goal", "user_knowledge", "pdf_chunk", "txt_chunk", "csv_chunk", "csv_summary", "csv_column"]
        )
    with colC:
        date_filter = st.selectbox(
            "Time range",
            ["All time", "Last 7 days", "Last 30 days", "Last 90 days", "Last year"]
        )
    with colD:
        importance_filter = st.selectbox(
            "Importance",
            ["(all)", "1", "2", "3", "4", "5"],
            help="Filter by importance score (1=low, 5=high)"
        )
    with colE:
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

    scope_global = memory_scope == "Global"
    proj = None if scope_global else active_project
    total = count_memories(project=proj, where_extra=where_extra if where_extra else None, global_only=scope_global)
    st.caption(f"Total memories (before date filter): {total}")

    query = st.text_input(
        "Search memories",
        "",
        placeholder="e.g., theorem, CLT, regression, pdf_chunk..."
    )
    st.caption("Searches text content, type, name, and source. Try: theorem, reference, insight, user_goal, etc.")

    items = list_memories(project=proj, where_extra=where_extra if where_extra else None, limit=limit, global_only=scope_global)

    # Apply date filter
    if date_cutoff > 0:
        items = [it for it in items if (it["metadata"] or {}).get("created_at", 0) >= date_cutoff]

    # Apply importance filter
    if importance_filter != "(all)":
        imp_val = int(importance_filter)
        items = [it for it in items if (it["metadata"] or {}).get("importance") == imp_val]

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

    # Selection and bulk actions
    if items:
        selected_count = sum(1 for it in items if st.session_state.get(f"sel_{it['id']}", False))
        sel_row1, sel_row2, sel_row3 = st.columns([1, 1, 4])
        with sel_row1:
            if st.button("Select all", key="mem_sel_all"):
                for it in items:
                    st.session_state[f"sel_{it['id']}"] = True
                st.rerun()
        with sel_row2:
            if st.button("Clear", key="mem_sel_clear"):
                keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith("sel_")]
                for k in keys_to_clear:
                    st.session_state[k] = False  # Set to False so checkboxes render unchecked
                st.rerun()
        with sel_row3:
            if selected_count > 0:
                if st.button(f"🗑️ Delete ({selected_count})", type="primary", key="mem_bulk_del"):
                    ids_to_delete = [it["id"] for it in items if st.session_state.get(f"sel_{it['id']}", False)]
                    deleted = delete_memories(ids_to_delete)
                    for mid in ids_to_delete:
                        st.session_state.pop(f"sel_{mid}", None)
                    st.success(f"Deleted {deleted} memory(ies).")
                    st.rerun()

    # Export dropdown (exports only selected memories)
    if items:
        def _escape_latex(s):
            chars = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
            for old, new in chars.items():
                s = s.replace(old, new)
            return s

        items_to_export = [it for it in items if st.session_state.get(f"sel_{it['id']}", False)]

        with st.popover("📥 Export", use_container_width=True):
            if not items_to_export:
                st.caption("Select memories to export")
            else:
                export_data = [{"id": it["id"], "text": it["text"], "metadata": it["metadata"]} for it in items_to_export]
                st.download_button("JSON", data=json.dumps(export_data, indent=2, default=str), file_name=f"memories_{active_project}_{int(time.time())}.json", mime="application/json", key="exp_json")
                md_lines = [f"# Memories Export — {active_project}\n", f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", f"Total: {len(items_to_export)} memories\n\n---\n"]
                for it in items_to_export:
                    md = it["metadata"] or {}
                    md_lines.append(f"\n## {md.get('name', 'Untitled')}\n**Type:** {md.get('type', 'unknown')} | **Created:** {md.get('created_at', '')}\n\n{it['text'] or ''}\n\n---\n")
                st.download_button("Markdown", data="".join(md_lines), file_name=f"memories_{active_project}_{int(time.time())}.md", mime="text/markdown", key="exp_md")
                latex_lines = [r"\documentclass{article}", r"\usepackage{amsmath,amssymb,amsthm}", r"\usepackage[margin=1in]{geometry}", r"\usepackage{hyperref}", r"\usepackage{listings}", r"\lstset{basicstyle=\ttfamily\small,breaklines=true}", r"", r"\title{" + f"Memory Export — {active_project}" + r"}", r"\date{" + time.strftime('%Y-%m-%d') + r"}", r"\begin{document}", r"\maketitle", r""]
                for it in items_to_export:
                    md = it["metadata"] or {}
                    text = it["text"] or ""
                    content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
                    content = re.sub(r'```python\n(.*?)\n```', r'\\begin{lstlisting}[language=Python]\n\1\n\\end{lstlisting}', content, flags=re.DOTALL)
                    content = re.sub(r'```\n?(.*?)\n?```', r'\\begin{lstlisting}\n\1\n\\end{lstlisting}', content, flags=re.DOTALL)
                    latex_lines.append(r"\section{" + _escape_latex(md.get("name", "Untitled")) + r"}")
                    latex_lines.append(r"\textbf{Type:} " + _escape_latex(md.get("type", "unknown")) + r"\\[0.5em]")
                    latex_lines.append(content)
                    latex_lines.append(r"\hrulefill")
                latex_lines.append(r"\end{document}")
                st.download_button("LaTeX", data="\n".join(latex_lines), file_name=f"memories_{active_project}_{int(time.time())}.tex", mime="text/x-tex", key="exp_latex")

    if not items:
        st.info("No memories found with current filters.")
    else:
        for it in items:
            md = it["metadata"] or {}
            created = md.get("created_at", "")
            mtype = md.get("type", "(no type)")
            name = md.get("name", "")
            text = it["text"] or ""
            mem_id = it["id"]

            # Generate a short preview title
            first_line = text.split("\n")[0].strip()
            preview = first_line[:80] + ("..." if len(first_line) > 80 else "")
            
            if name:
                title = f"📄 **{mtype}** — {name}"
            elif preview:
                title = f"📄 **{mtype}** — {preview}"
            else:
                title = f"📄 **{mtype}**"

            col_cb, col_exp = st.columns([1, 30])
            with col_cb:
                st.checkbox("Select memory", key=f"sel_{mem_id}", label_visibility="collapsed")
            with col_exp:
                with st.expander(title, expanded=False):
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

# =========================
# TAB: CALENDAR
# =========================
with tab_calendar:
    from datetime import datetime, timedelta

    migrated = migrate_from_tasks()
    if migrated > 0:
        st.info(f"Migrated {migrated} task(s) from previous format.")
    cleaned = cleanup_past_due()
    if cleaned > 0:
        st.caption(f"Cleaned up {cleaned} past-due task(s).")

    st.subheader("📅 Calendar")
    st.caption("Add tasks and events with natural language. Calendar shows due dates and events. Chunk board below shows daily work.")

    # Add task / event
    add_input = st.text_input(
        "Add task or event",
        placeholder='Task: stats hw due feb 17, 2 hours, priority 3/5 | Event: meeting feb 14 2-3pm',
        key="cal_add_input",
    )
    col_add_task, col_add_ev = st.columns(2)
    with col_add_task:
        if st.button("Add task", key="add_task") and add_input.strip():
            with st.spinner("Parsing..."):
                p = parse_task_from_text(add_input.strip())
            if p and not p.get("error"):
                th = p.get("duration_hours")
                th = float(th) if th is not None else 1.0
                try:
                    pr = max(1, min(5, int(p.get("priority") or 3)))
                except (TypeError, ValueError):
                    pr = 3
                add_task(p["name"], p["due_date"], th, pr)
                get_chunks_for_date(datetime.now().strftime("%Y-%m-%d"))  # ensure chunks for new task
                st.success(f"Task: **{p['name']}** due {p['due_date']}")
                st.rerun()
            else:
                st.error("Could not parse. Use: 'name due feb 17, 2 hours, priority 3/5'")
    with col_add_ev:
        if st.button("Add event", key="add_event") and add_input.strip():
            with st.spinner("Parsing..."):
                p = parse_event_from_text(add_input.strip())
            if p and not p.get("error"):
                add_event(p["name"], p["date"], p["start_time"], p["end_time"])
                st.success(f"Event: **{p['name']}** on {p['date']} {p['start_time']}-{p['end_time']}")
                st.rerun()
            else:
                st.error("Could not parse. Use: 'meeting feb 14 2-3pm'")

    # Auto-cleanup past-due tasks and past events when calendar loads
    cleanup_past_due()
    cleanup_past_events()

    # Manage tasks (delete)
    today_str_cal = datetime.now().strftime("%Y-%m-%d")
    active_tasks_list = get_active_tasks(today_str_cal)
    with st.expander("Manage tasks", expanded=False):
        if not active_tasks_list:
            st.caption("No active tasks.")
        else:
            for t in active_tasks_list:
                pct = get_task_completion_pct(t.get("id", ""))
                total_h = t.get("total_hours", 0)
                label = f"{t.get('name', '')} — due {t.get('due_date', '')} ({total_h}h"
                if pct is not None and pct < 100:
                    label += f", {pct}% done"
                label += ")"
                if pct is not None and pct == 100:
                    label = f"✓ {label} — completed"
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption(label)
                with c2:
                    if st.button("🗑️ Delete", key=f"deltask_{t['id']}"):
                        delete_task(t["id"])
                        st.success("Task deleted.")
                        st.rerun()
    all_events = load_events()
    if all_events:
        with st.expander("Manage events", expanded=False):
            for e in all_events:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption(f"{e.get('name', '')} — {e.get('date', '')} {e.get('start_time', '')}-{e.get('end_time', '')}")
                with c2:
                    if st.button("🗑️", key=f"delev_{e['id']}"):
                        delete_event(e["id"])
                        st.success("Event deleted.")
                        st.rerun()

    # Rolling 14 days: 2 rows × 7 columns, weekdays on top, dates in corners
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    week_start = today - timedelta(days=today.weekday())  # Monday of this week
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Refresh today's chunks when app opens (so they reflect current tasks)
    if "cal_last_refresh" not in st.session_state or st.session_state.cal_last_refresh != today_str:
        refresh_chunks_for_date(today_str)
        st.session_state.cal_last_refresh = today_str

    # Column headers: weekdays (calendar locked to today—no day switching)
    header_cols = st.columns(7)
    for i, lbl in enumerate(day_labels):
        with header_cols[i]:
            st.markdown(f"**{lbl}**")

    # Row 1: days 0-6, Row 2: days 7-13 (rolls to next 2 weeks when date advances)
    for row in range(2):
        row_cols = st.columns(7)
        for col in range(7):
            day_idx = row * 7 + col
            d = week_start + timedelta(days=day_idx)
            ds = d.strftime("%Y-%m-%d")
            with row_cols[col]:
                is_today = ds == today_str
                date_label = f"{d.day}" + (" •" if is_today else "")
                st.markdown(f"**{date_label}**" if is_today else date_label)
                # Only due tasks and events
                due_tasks = get_tasks_due_on_date(ds)
                events = get_events_for_date(ds)
                for t in due_tasks:
                    pct = get_task_completion_pct(t.get("id", ""))
                    label = t.get("name", "")
                    if pct is not None and 0 < pct < 100:
                        label = f"{label} ({pct}% done)"
                    elif pct == 100:
                        label = f"{label} ✓"
                    st.caption(f"📋 {label} due")
                for e in events:
                    st.caption(f"📌 {e.get('name', '')} {e.get('start_time', '')}-{e.get('end_time', '')}")

    # Daily chunk board — always today, no day switching
    st.divider()
    sel_date = datetime.strptime(today_str, "%Y-%m-%d").date()
    st.subheader(f"Daily chunk board — {today_str} ({day_labels[sel_date.weekday()]})")
    st.caption("Work chunks for today. Mark done to complete. Chunks regenerate on app open based on current tasks.")

    # Use read-only load so _ensure_chunks_for_date doesn't overwrite completed chunks on every display.
    # Chunks are ensured by refresh_chunks_for_date on first load (above).
    chunks = get_chunks_for_date_read_only(today_str)
    incomplete = [ch for ch in chunks if not ch.get("completed")]

    # Optional form for "expected time to complete rest" (appears when user clicks ⏱ on a chunk)
    if "chunk_set_remaining_pending" in st.session_state:
        pending = st.session_state.chunk_set_remaining_pending
        with st.form("chunk_remaining_form", clear_on_submit=True):
            st.caption(f"Task: {pending.get('name', '')}")
            hours = st.number_input(
                "Expected time to complete the rest of assignment (hours)",
                min_value=0.0, max_value=40.0, value=0.5, step=0.25, format="%.2f",
                key="chunk_remaining_hours"
            )
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Save & complete chunk")
            with col2:
                cancelled = st.form_submit_button("Cancel")
            if submitted:
                mark_chunk_complete(pending["id"], actual_minutes=pending.get("mins", 30))
                update_task_remaining_estimate(pending["task_id"], hours)
                if "chunk_set_remaining_pending" in st.session_state:
                    del st.session_state.chunk_set_remaining_pending
                st.rerun()
            elif cancelled:
                if "chunk_set_remaining_pending" in st.session_state:
                    del st.session_state.chunk_set_remaining_pending
                st.rerun()
        st.divider()

    if not incomplete:
        st.info("No work scheduled for this day." if not chunks else "All chunks done for this day. ✓")
    else:
        for ch in incomplete:
            name = ch.get("_task_name", "Task")
            mins = ch.get("duration_minutes", 30)
            task_id = ch.get("task_id", "")
            col_chunk, col_done, col_remaining, col_task = st.columns([2, 1, 1, 1])
            with col_chunk:
                st.markdown(f"{name} ({mins}m)")
            with col_done:
                if st.button("✔ Done", key=f"chk_{ch['id']}"):
                    mark_chunk_complete(ch["id"], actual_minutes=mins)
                    st.rerun()
            with col_remaining:
                if st.button("⏱", key=f"time_{ch['id']}", help="Set expected time to complete the rest"):
                    st.session_state.chunk_set_remaining_pending = {
                        "id": ch["id"], "mins": mins, "name": name, "task_id": task_id
                    }
                    st.rerun()
            with col_task:
                if st.button("✓ All", key=f"task_{task_id}_{ch['id']}", help="Mark entire task as completed"):
                    mark_task_complete(task_id)
                    st.rerun()

# =========================
# TAB: REVIEW (Spaced Repetition)
# =========================
with tab_review:
    st.subheader("🎯 Review Mode")
    st.caption("Test your knowledge with flashcard-style review of your memories.")
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        review_scope = st.radio("Scope", ["This project only", "All projects"], horizontal=True, key="review_scope")
    with col2:
        review_types = st.multiselect(
            "Memory types to review",
            ["definition", "theorem", "formula", "function", "example", "insight"],
            default=["definition", "theorem", "formula"],
            key="review_types"
        )
    
    # Determine project filter
    review_project = active_project if review_scope == "This project only" else None
    
    # Initialize review state
    if "review_cards" not in st.session_state:
        st.session_state.review_cards = []
    if "review_index" not in st.session_state:
        st.session_state.review_index = 0
    if "review_revealed" not in st.session_state:
        st.session_state.review_revealed = False
    if "review_stats" not in st.session_state:
        st.session_state.review_stats = {"correct": 0, "review": 0}
    
    # Load/refresh cards button
    if st.button("🔄 Load Review Cards", type="primary"):
        # Get memories of selected types
        all_cards = []
        for mem_type in review_types:
            items = list_memories(project=review_project, where_extra={"type": mem_type}, limit=50)
            all_cards.extend(items)
        
        # Shuffle
        import random
        random.shuffle(all_cards)
        
        st.session_state.review_cards = all_cards
        st.session_state.review_index = 0
        st.session_state.review_revealed = False
        st.session_state.review_stats = {"correct": 0, "review": 0}
        st.rerun()
    
    # Display current card
    cards = st.session_state.review_cards
    idx = st.session_state.review_index
    
    if cards:
        st.markdown(f"**Card {idx + 1} of {len(cards)}**")
        
        # Progress bar
        progress = (idx / len(cards)) if len(cards) > 0 else 0
        st.progress(progress)
        
        current_card = cards[idx]
        md = current_card.get("metadata") or {}
        name = md.get("name", "Unnamed")
        mtype = md.get("type", "unknown")
        text = current_card.get("text", "")
        
        # Show the "question" (name/type as prompt)
        st.markdown(f"### {mtype.title()}: {name}")
        
        # Reveal/hide content
        if st.session_state.review_revealed:
            st.markdown("---")
            st.markdown(convert_latex_for_streamlit(text))
            st.markdown("---")
            
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ I knew it", type="primary"):
                    st.session_state.review_stats["correct"] += 1
                    st.session_state.review_index += 1
                    st.session_state.review_revealed = False
                    if st.session_state.review_index >= len(cards):
                        st.session_state.review_index = 0  # Loop back
                    st.rerun()
            with col2:
                if st.button("🔁 Review again"):
                    st.session_state.review_stats["review"] += 1
                    # Move card to end of deck
                    card = cards.pop(idx)
                    cards.append(card)
                    st.session_state.review_cards = cards
                    st.session_state.review_revealed = False
                    st.rerun()
        else:
            st.info("Try to recall the content before revealing!")
            if st.button("👁️ Reveal Answer", type="secondary"):
                st.session_state.review_revealed = True
                st.rerun()
        
        # Navigation
        st.markdown("---")
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
        with nav_col1:
            if st.button("⬅️ Previous") and idx > 0:
                st.session_state.review_index -= 1
                st.session_state.review_revealed = False
                st.rerun()
        with nav_col2:
            if st.button("➡️ Skip") and idx < len(cards) - 1:
                st.session_state.review_index += 1
                st.session_state.review_revealed = False
                st.rerun()
        
        # Stats
        stats = st.session_state.review_stats
        total_reviewed = stats["correct"] + stats["review"]
        if total_reviewed > 0:
            accuracy = (stats["correct"] / total_reviewed) * 100
            st.sidebar.markdown(f"### Review Stats")
            st.sidebar.markdown(f"✅ Correct: {stats['correct']}")
            st.sidebar.markdown(f"🔁 Need review: {stats['review']}")
            st.sidebar.markdown(f"📊 Accuracy: {accuracy:.0f}%")
    else:
        st.info("Click 'Load Review Cards' to start reviewing your memories.")
