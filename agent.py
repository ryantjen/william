# Run the app: streamlit run app.py (use run_agent.bat / run_agent.sh, or CTRL+C to close)

import json
import re
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME
from memory import retrieve_memory_with_metadata, store_memory, store_memory_if_unique, list_memories, delete_memory, touch_memories

client = OpenAI(api_key=OPENAI_API_KEY)

# Personality cache: {(project, msg_count_key): (user_prefs, agent_traits)}
# Refresh every N messages to keep context fresh
PERSONALITY_CACHE_REFRESH_EVERY = 5

SYSTEM_PROMPT = """
You are a Learning & Research Copilot named William. You help across any subject: math, science, biology, history, programming, statistics, etc.

You help with:
- mathematical proofs and formulas
- statistical modeling and analysis
- simulation design
- identifying assumptions
- learning concepts and definitions
- suggesting research improvements

Use relevant past research memory if helpful.
Think step-by-step before answering.
Respond in Markdown. For math, use LaTeX with $ for inline (e.g., $x^2$) and $$ for blocks (e.g., $$\\int_0^1 f(x) dx$$).
Do NOT use \\( \\) or \\[ \\] for math - only use $ and $$.
Try to answer as succinctly as possible so as not to overwhelm the user, but can elaborate if needed.
"""

def _parse_json_from_llm(raw: str) -> dict | None:
    """
    Extract and parse JSON from LLM response. Handles markdown wrapping and common issues.
    """
    text = (raw or "").strip()
    if not text:
        return None
    # Strip markdown code blocks: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object - look for {...}
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
    return None


def build_personality_prompt(user_prefs: list, agent_traits: list, user_context: list = None) -> str:
    """
    Build a personality and user context block from retrieved preferences, traits, and user context.
    Accepts list of dicts with 'text' and 'confidence'. Sorts by confidence (highest first).
    user_context: user_goal and user_knowledge memories (goals, knowledge level).
    Returns a string to inject into the system prompt.
    """
    def _format_sorted(items: list, limit: int = 5) -> str:
        if not items:
            return ""
        # Sort by confidence desc, then importance, then recency (newer wins on ties)
        sorted_items = sorted(
            items,
            key=lambda x: (x.get("confidence", 0), x.get("importance", 3), x.get("created_at", 0)),
            reverse=True
        )
        texts = [m["text"] if isinstance(m, dict) else m for m in sorted_items[:limit]]
        return "\n".join(f"- {t}" for t in texts)

    sections = []
    if user_context:
        ctx_text = _format_sorted(user_context)
        if ctx_text:
            sections.append(f"User context (goals, knowledge level - adapt explanations accordingly):\n{ctx_text}")
    if user_prefs:
        prefs_text = _format_sorted(user_prefs)
        if prefs_text:
            sections.append(f"User preferences (adapt your style accordingly):\n{prefs_text}")
    if agent_traits:
        traits_text = _format_sorted(agent_traits)
        if traits_text:
            sections.append(f"Your communication traits (what works well):\n{traits_text}")
    if not sections:
        return ""
    return "\n\n".join(sections)

def _retrieve_memories_parallel(user_input: str, project: str, include_personality: bool = True):
    """
    Run memory retrievals in parallel. If include_personality=False (cache hit), only 2 fetches.
    Returns (project_mems, global_mems, user_prefs, agent_traits, user_context).
    user_prefs, agent_traits, user_context are list of dicts with text, confidence (for ordering).
    user_context: user_goal and user_knowledge memories (goals, knowledge level).
    """
    if include_personality:
        with ThreadPoolExecutor(max_workers=5) as ex:
            f1 = ex.submit(retrieve_memory_with_metadata, user_input, project, 8)
            f2 = ex.submit(retrieve_memory_with_metadata, user_input, None, 4)
            f3 = ex.submit(retrieve_memory_with_metadata, "user preferences communication style learning", None, 5)
            f4 = ex.submit(retrieve_memory_with_metadata, "agent personality traits approach style", None, 5)
            f5 = ex.submit(retrieve_memory_with_metadata, "user goals knowledge level current learning facts about the user", None, 5)
            project_mems, global_mems = f1.result(), f2.result()
            raw_prefs, raw_traits, raw_user_ctx = f3.result(), f4.result(), f5.result()
            user_prefs = [m for m in raw_prefs if m.get("type") == "user_preference"]
            agent_traits = [m for m in raw_traits if m.get("type") == "agent_trait"]
            user_context = [m for m in raw_user_ctx if m.get("type") in ("user_goal", "user_knowledge")]
    else:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(retrieve_memory_with_metadata, user_input, project, 8)
            f2 = ex.submit(retrieve_memory_with_metadata, user_input, None, 4)
            project_mems, global_mems = f1.result(), f2.result()
            user_prefs, agent_traits, user_context = [], [], []
    return project_mems, global_mems, user_prefs, agent_traits, user_context


def _get_cached_personality(project: str, chat_history_len: int) -> str | None:
    """Return cached personality block if valid, else None."""
    if not hasattr(_get_cached_personality, "_cache"):
        _get_cached_personality._cache = {}
    key = (project, chat_history_len // PERSONALITY_CACHE_REFRESH_EVERY)
    return _get_cached_personality._cache.get(key)


def _build_messages_for_ask(user_input: str, project: str, project_goal: str | None,
                            project_block: str, global_block: str, personality_block: str,
                            chat_history: list | None) -> list:
    """Build the messages list for chat completion."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if project_goal:
        messages.append({"role": "system", "content": f"Current project: {project}\nProject goal: {project_goal}\n\nTailor your responses to help achieve this goal."})
    if chat_history:
        recent_history = chat_history[-20:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "system", "content": f"Project memories ('{project}'):\n{project_block}"})
    messages.append({"role": "system", "content": f"Global memories:\n{global_block}"})
    if personality_block:
        messages.append({"role": "system", "content": personality_block})
    messages.append({"role": "user", "content": user_input})
    return messages


def ask_agent(user_input: str, project: str, chat_history: list = None, project_goal: str = None):
    """
    Context hierarchy (highest to lowest priority):
    1. System prompt + Project goal
    2. Chat history (recent conversation)
    3. Project memories (specific to current project)
    4. Global memories (cross-project knowledge)
    5. Personality context (communication style)
    
    Returns: tuple (answer: str, citations: list of memory dicts)
    Uses parallel memory retrieval, personality caching, and background personality analysis.
    """
    chat_len = len(chat_history) if chat_history else 0
    cached_personality = _get_cached_personality(project, chat_len)
    include_personality = cached_personality is None
    
    # Parallel memory retrieval (2 or 5 fetches depending on cache)
    project_mems, global_mems, user_prefs, agent_traits, user_context = _retrieve_memories_parallel(user_input, project, include_personality)
    
    # De-duplicate global
    project_texts = set(m["text"] for m in project_mems)
    global_mems = [m for m in global_mems if m["text"] not in project_texts]
    
    project_block = "\n".join(f"- {m['text']}" for m in project_mems) if project_mems else "- (none)"
    global_block = "\n".join(f"- {m['text']}" for m in global_mems) if global_mems else "- (none)"
    all_citations = project_mems + global_mems
    
    # Personality: use cache or freshly fetched, update cache
    if cached_personality is not None:
        personality_block = cached_personality
    else:
        personality_block = build_personality_prompt(user_prefs, agent_traits, user_context)
        _get_cached_personality._cache[(project, chat_len // PERSONALITY_CACHE_REFRESH_EVERY)] = personality_block

    messages = _build_messages_for_ask(user_input, project, project_goal, project_block, global_block, personality_block, chat_history)

    try:
        resp = _chat_completion_with_retry(messages)
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Sorry, I encountered an error: {e}. Please try again."
        all_citations = []

    # Background: learn from conversation (non-blocking)
    if answer and not answer.startswith("Sorry, I encountered"):
        threading.Thread(target=analyze_conversation_style, args=(user_input, answer), daemon=True).start()

    # Background: touch cited memories for recency (non-blocking)
    if all_citations:
        threading.Thread(target=touch_memories, args=([m["id"] for m in all_citations],), daemon=True).start()

    return answer, all_citations


def ask_agent_stream(user_input: str, project: str, chat_history: list = None, project_goal: str = None, result_holder: dict = None):
    """
    Same as ask_agent but streams the response. Yields chunks for display.
    Caller must pass result_holder={} - it will be filled with {"answer": str, "citations": list} when done.
    """
    if result_holder is None:
        result_holder = {}
    
    chat_len = len(chat_history) if chat_history else 0
    cached_personality = _get_cached_personality(project, chat_len)
    include_personality = cached_personality is None
    
    project_mems, global_mems, user_prefs, agent_traits, user_context = _retrieve_memories_parallel(user_input, project, include_personality)

    project_texts = set(m["text"] for m in project_mems)
    global_mems = [m for m in global_mems if m["text"] not in project_texts]

    project_block = "\n".join(f"- {m['text']}" for m in project_mems) if project_mems else "- (none)"
    global_block = "\n".join(f"- {m['text']}" for m in global_mems) if global_mems else "- (none)"
    all_citations = project_mems + global_mems

    if cached_personality is not None:
        personality_block = cached_personality
    else:
        personality_block = build_personality_prompt(user_prefs, agent_traits, user_context)
        _get_cached_personality._cache[(project, chat_len // PERSONALITY_CACHE_REFRESH_EVERY)] = personality_block
    
    messages = _build_messages_for_ask(user_input, project, project_goal, project_block, global_block, personality_block, chat_history)

    try:
        stream = client.chat.completions.create(model=MODEL_NAME, messages=messages, stream=True)
        full_answer = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_answer.append(delta)
                yield delta
        
        answer = "".join(full_answer)
        
        # Background personality analysis
        if answer and not answer.startswith("Sorry, I encountered"):
            threading.Thread(target=analyze_conversation_style, args=(user_input, answer), daemon=True).start()
        
        result_holder["answer"] = answer
        result_holder["citations"] = all_citations
        if all_citations:
            threading.Thread(target=touch_memories, args=([m["id"] for m in all_citations],), daemon=True).start()

    except Exception as e:
        err_msg = f"Sorry, I encountered an error: {e}. Please try again."
        yield err_msg
        result_holder["answer"] = err_msg
        result_holder["citations"] = []

def _chat_completion_with_retry(messages: list, max_retries: int = 2):
    """Call OpenAI API with retry on transient failures (rate limit, timeout, connection)."""
    last_error = None
    err_str_lower = ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(model=MODEL_NAME, messages=messages)
            return resp
        except Exception as e:
            last_error = e
            err_str_lower = str(e).lower()
            if attempt < max_retries and any(kw in err_str_lower for kw in ("rate", "timeout", "connection", "503", "502")):
                time.sleep(2 ** attempt)
                continue
            raise last_error


def execute_natural_language(task: str, project: str) -> str:
    """
    Natural language -> Python code. Returns code only (no execution).
    Passes project memories (especially csv_summary, csv_column) for context.
    """
    # Retrieve relevant project memories for context (dataset summaries, column info, etc.)
    project_mems = retrieve_memory_with_metadata(task, project=project, n_results=5)
    context_block = ""
    if project_mems:
        # Prefer csv_summary and csv_column for data tasks
        def fmt(m):
            t = m["text"]
            return f"- {m['name']}: {t[:600]}..." if len(t) > 600 else f"- {m['name']}: {t}"
        context_lines = [fmt(m) for m in project_mems[:4]]
        context_block = "\n\nRelevant context from project memory (column names, dataset structure - use if applicable):\n" + "\n".join(context_lines)
    
    prompt = f"""
Write Python code to accomplish the task below.
Rules:
- Only output code (no explanations and no markdown formatting).
- Use matplotlib for plots if needed. Do NOT call plt.show().
- If plotting, create the figure normally (Streamlit will render it).
- numpy is available as np, pandas as pd.
Task: {task}
{context_block}
"""
    try:
        code_resp = _chat_completion_with_retry([
            {"role": "system", "content": "You write clean Python code for statistics and data analysis. Output only code, no markdown."},
            {"role": "user", "content": prompt},
        ])
    except Exception as e:
        return f"# API error: {e}\n# Task: {task}"
    
    return code_resp.choices[0].message.content

def should_store_memory(user_input: str, answer: str, project: str, project_goal: str | None = None):
    """
    Analyze conversation and extract multiple memories worth storing.
    Returns a list of memory objects to store.
    Uses project_goal (if set) to prioritize content that supports the goal.
    """
    goal_block = f"\n    Project goal: {project_goal}\n    Prioritize content that supports this goal. Content that directly supports the goal should generally receive higher importance (4-5).\n" if project_goal else ""
    gate_prompt = f"""
    You are deciding what should be saved into long-term memory for a learning and research assistant (any subject: math, science, biology, history, programming, etc.).

    Project: {project}{goal_block}

    Conversation:
    User: {user_input}
    Assistant: {answer}

    Extract ALL distinct pieces of knowledge worth remembering:
    - Definitions (formal definitions of terms/concepts)
    - Theorems and formulas
    - Functions (reusable code or algorithms worth saving)
    - Examples (worked examples, illustrations of concepts)
    - Insights and key observations
    - Assumptions made
    - Decisions or conclusions
    - Results and findings
    - References mentioned
    - user_goal: When the user states a learning goal, objective, or what they want to achieve (e.g. "User's goal: master hypothesis testing")
    - user_knowledge: When the user indicates their knowledge level, background, or what they already know (e.g. "User knows calculus; learning measure theory")

    Do NOT store:
    - Trivial Q&A or small talk
    - Vague statements without specific content
    - Duplicates of the same concept

    Return JSON with a "memories" array. Each memory should have:
    - name (a short descriptive title, e.g. "Central Limit Theorem", "bootstrap_ci", "CLT Example")
    - type (one of: definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology, user_goal, user_knowledge)
    - importance (1-5) - Use the FULL range relative to the memories you're extracting:
      * 5 = Critical/fundamental (core theorems, essential definitions, key methodologies)
      * 4 = Very important (important functions, significant insights, major results)
      * 3 = Moderately important (useful examples, helpful functions, standard concepts)
      * 2 = Somewhat useful (minor insights, edge cases, less common examples)
      * 1 = Nice to have (trivial examples, obvious facts, minor details)
      IMPORTANT: Distribute importance across the full 1-5 range. Don't default everything to 3+.
    - text (the full content - see formatting rules below)

    **FORMATTING RULES for text field:**
    - For formulas/equations: Put the formula name first, then use $$...$$ for the main equation on its own line
    - Use $...$ only for inline math within sentences
    - For examples: Clearly label as "Example:" and show the worked solution
    - For definitions: Start with the term being defined in bold
    - For functions: Use function name as title, include docstring description, wrap code in ```python blocks

    If nothing is worth storing, return {{"memories": []}}

    Example response:
    {{"memories": [
        {{"name": "Central Limit Theorem", "type": "theorem", "importance": 5, "text": "**Central Limit Theorem**\\n\\n..."}},
        {{"name": "bootstrap_ci", "type": "function", "importance": 4, "text": "**bootstrap_ci**\\n\\nComputes bootstrap confidence interval for a dataset.\\n\\n```python\\ndef bootstrap_ci(data, n_boot=1000, alpha=0.05):\\n    import numpy as np\\n    samples = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boot)]\\n    return np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])\\n```"}},
        {{"name": "CLT Example - Dice", "type": "example", "importance": 2, "text": "**Example: CLT with Dice**\\n\\nRolling a fair die 100 times: $\\\\mu = 3.5$, $\\\\sigma^2 = 35/12$.\\n\\nBy CLT, $\\\\bar{{X}} \\\\sim N(3.5, 35/1200) = N(3.5, 0.029)$"}}
    ]}}
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON with a 'memories' array. No markdown. Escape backslashes in text: use \\\\frac not \\frac. Use $...$ and $$...$$ for math."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    raw = resp.choices[0].message.content.strip()
    data = _parse_json_from_llm(raw)
    if data is None:
        return []
    return data.get("memories", [])

def _get_contradicted_ids(new_text: str, mem_type: str) -> list:
    """
    Check if new_text contradicts existing memories of mem_type.
    Returns list of memory ids to delete (contradicted ones). New memory should supersede them.
    """
    existing = list_memories(project=None, where_extra={"type": mem_type}, limit=20, global_only=True)
    if not existing:
        return []
    existing_texts = [m["text"] for m in existing if m.get("text")]
    if not existing_texts:
        return []
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(existing_texts[:10]))
    prompt = f"""Existing {mem_type.replace('_', ' ')}s:
{numbered}

New observation to store: "{new_text}"

Which existing one(s) does the new observation CONTRADICT? (e.g. "prefers concise" vs "prefers detailed" - the new replaces the old)
Reply with the number(s) of contradicted ones (e.g. "1" or "1, 3"), or "none" if no contradiction."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Identify which existing items the new observation contradicts. Reply with numbers (e.g. 1 or 1, 3) or none."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        if "none" in answer or not answer:
            return []
        ids_to_delete = []
        for part in answer.replace(",", " ").split():
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1  # 1-based to 0-based
                if 0 <= idx < len(existing):
                    ids_to_delete.append(existing[idx]["id"])
        return ids_to_delete
    except Exception:
        return []  # On error, allow store (fail open)

def analyze_conversation_style(user_input: str, response: str):
    """
    Analyze conversation and extract:
    1. User preferences (how they like things explained)
    2. Agent traits (what approaches worked well)
    
    Stores findings as global memories (not project-specific).
    Uses rate limiting to avoid flooding memory.
    """
    analysis_prompt = f"""
    Analyze this conversation exchange to learn about communication style.

    User said: {user_input}
    Assistant responded: {response}

    Look for signals like:
    - User asking for simpler/more detailed explanations
    - User's technical vocabulary level
    - User's preferred response length (concise vs detailed)
    - User expressing satisfaction or confusion
    - What explanation approach seemed effective

    Return JSON with these keys:
    - store_preference (true/false) - should we store a user preference?
    - preference_text (string) - the preference to store, e.g. "User prefers examples before formal definitions"
    - store_trait (true/false) - should we store an agent personality trait?
    - trait_text (string) - the trait to store, e.g. "Step-by-step breakdowns work well in our conversations"
    - confidence (1-5) - how confident are you in these observations?

    IMPORTANT:
    - Only store if there's a CLEAR signal (confidence >= 4)
    - Preferences are about the USER's style (3rd person: "User prefers...")
    - Traits are about the AGENT's effective approaches (1st person: "I find that...")
    - Most conversations won't have anything worth storing - that's fine
    - Don't store trivial observations
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Be conservative - only flag things worth remembering long-term."},
            {"role": "user", "content": analysis_prompt},
        ]
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return  # Fail silently - style learning is optional

    confidence = int(data.get("confidence", 0))
    
    # Only store if confidence is high enough
    if confidence < 4:
        return

    # Store user preference if flagged (supersede contradicted, then store new)
    if data.get("store_preference") and data.get("preference_text"):
        pref_text = data["preference_text"]
        contradicted_ids = _get_contradicted_ids(pref_text, "user_preference")
        for mem_id in contradicted_ids:
            delete_memory(mem_id)
        store_memory_if_unique(
            text=pref_text,
            metadata={
                "type": "user_preference",
                "confidence": confidence,
                "source": "style_analysis"
            },
            project=None  # Global - applies across all projects
        )

    # Store agent trait if flagged (supersede contradicted, then store new)
    if data.get("store_trait") and data.get("trait_text"):
        trait_text = data["trait_text"]
        contradicted_ids = _get_contradicted_ids(trait_text, "agent_trait")
        for mem_id in contradicted_ids:
            delete_memory(mem_id)
        store_memory_if_unique(
            text=trait_text,
            metadata={
                "type": "agent_trait",
                "confidence": confidence,
                "source": "style_analysis"
            },
            project=None  # Global - applies across all projects
        )

# =============================================================================
# ON-DEMAND MEMORY EXTRACTION (called by user action, not automatic)
# =============================================================================

def extract_memories_from_exchange(user_input: str, answer: str, project: str, project_goal: str | None = None) -> tuple[int, int]:
    """
    On-demand: Extract and store memories from a chat exchange.
    Called when user clicks "Save to Memory" button.
    project_goal (if set) influences what gets stored and how importance is assigned.
    Returns (stored_count, extracted_count).
    """
    memories = should_store_memory(user_input, answer, project, project_goal)
    stored_count = 0
    extracted_count = sum(1 for m in memories if m.get("text"))
    
    for mem in memories:
        if mem.get("text"):
            result = store_memory_if_unique(
                text=mem["text"],
                metadata={
                    "name": mem.get("name", ""),
                    "type": mem.get("type", "insight"),
                    "importance": int(mem.get("importance", 3)),
                    "source": "chat"
                },
                project=project
            )
            if result:
                stored_count += 1
    
    return stored_count, extracted_count

def extract_memories_from_text(text: str, project: str | None, project_goal: str | None = None) -> int:
    """
    On-demand: Extract and store memories from arbitrary text input.
    Called from the "Add Memory" tab.
    project_goal (if set) influences what gets stored and how importance is assigned.
    Returns count of memories stored.
    """
    goal_block = f"\n    Project goal: {project_goal}\n    Prioritize content that supports this goal. Content that directly supports the goal should generally receive higher importance (4-5).\n" if project_goal else ""
    gate_prompt = f"""
    You are extracting knowledge to save into long-term memory for a learning and research assistant (any subject: math, science, biology, history, programming, etc.).

    Project: {project or "Global"}{goal_block}

    Text to analyze:
    {text}

    Extract ALL distinct pieces of knowledge worth remembering:
    - Definitions (formal definitions of terms/concepts)
    - Theorems and formulas
    - Functions (reusable code functions)
    - Examples (worked examples, illustrations of concepts)
    - Insights and key observations
    - Assumptions or conditions
    - Important results or findings
    - References or citations
    - Methodology notes
    - user_goal: When the user states a learning goal, objective, or what they want to achieve (e.g. "User's goal: master hypothesis testing")
    - user_knowledge: When the user indicates their knowledge level, background, or what they already know (e.g. "User knows calculus; learning measure theory")

    Do NOT store:
    - Vague statements without specific content
    - Duplicates of the same concept
    - Trivial or obvious information

    Return JSON with a "memories" array. Each memory should have:
    - name (a short descriptive title, e.g. "Central Limit Theorem", "bootstrap_ci", "CLT Example")
    - type (one of: definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology, user_goal, user_knowledge)
    - importance (1-5) - Use the FULL range relative to the memories you're extracting:
      * 5 = Critical/fundamental (core theorems, essential definitions, foundational concepts)
      * 4 = Very important (important functions, significant insights, key methodologies)
      * 3 = Moderately important (useful examples, helpful functions, standard concepts)
      * 2 = Somewhat useful (minor insights, edge cases, less common examples)
      * 1 = Nice to have (trivial examples, obvious facts, minor details)
      IMPORTANT: Distribute importance across the full 1-5 range. Don't default everything to 3+.
    - text (the full content - see formatting rules below)

    **FORMATTING RULES for text field:**
    - For formulas/equations: Put the formula name first, then use $$...$$ for the main equation on its own line
    - Use $...$ only for inline math within sentences
    - For examples: Clearly label as "Example:" and show the worked solution
    - For definitions: Start with the term being defined in bold
    - For functions: Use function name as title, include brief description, wrap code in ```python blocks

    If nothing is worth storing, return {{"memories": []}}
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON with a 'memories' array. No markdown. Escape backslashes: use \\\\frac not \\frac."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    data = _parse_json_from_llm(resp.choices[0].message.content)
    memories = data.get("memories", []) if data else []

    stored_count = 0
    for mem in memories:
        if mem.get("text"):
            result = store_memory_if_unique(
                text=mem["text"],
                metadata={
                    "name": mem.get("name", ""),
                    "type": mem.get("type", "insight"),
                    "importance": int(mem.get("importance", 3)),
                    "source": "manual_input"
                },
                project=project
            )
            if result:  # Not a duplicate
                stored_count += 1
    
    return stored_count


def summarize_conversation(chat_history: list, project: str) -> str:
    """
    Summarize a chat session into key points and store as a memory.
    Returns the summary text, or empty string if failed.
    """
    if not chat_history:
        return ""
    
    # Build conversation text
    conv_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conv_text += f"{role}: {msg['content']}\n\n"
    
    prompt = f"""
    Summarize this research conversation into a concise summary that captures:
    - Main topics discussed
    - Key questions asked and answers given
    - Important insights, theorems, or formulas mentioned
    - Decisions made or conclusions reached
    - Any action items or next steps
    
    Keep the summary focused and useful for future reference. Use bullet points.
    If formulas were discussed, include them using LaTeX ($...$ for inline, $$...$$ for block).
    
    Conversation:
    {conv_text}
    """
    
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are summarizing a research conversation. Be concise but capture all important details."},
            {"role": "user", "content": prompt}
        ]
    )
    
    summary = resp.choices[0].message.content.strip()
    
    if summary:
        # Store the summary as a memory
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        store_memory_if_unique(
            text=f"**Session Summary ({timestamp})**\n\n{summary}",
            metadata={
                "name": f"Session Summary - {timestamp}",
                "type": "summary",
                "importance": 4,
                "source": "conversation_summary"
            },
            project=project
        )
    
    return summary


# =============================================================================
# CALENDAR TASKS
# =============================================================================


def suggest_chunk_sizes(total_mins: int, task_name: str = "") -> list[int] | None:
    """
    Use LLM to suggest optimal chunk sizes from [15, 25, 30, 45, 60] minutes.
    Prefer fewer longer chunks for deep work. Returns list of mins, or None to use fallback.
    """
    allowed = [15, 25, 30, 45, 60]
    prompt = f"""Split {total_mins} minutes of work into chunks. Allowed sizes only: {allowed}.

Prefer fewer, longer chunks for deep work (e.g. 60 or 45 min). Use 25-30 for shorter sessions.
Task context: {task_name or "general"}.

Return ONLY a JSON array of minutes, e.g. [30, 30, 30] or [45, 45] or [60, 15].
The numbers must sum to {total_mins}. Use only values from {allowed}. You may combine (e.g. 15+15=30) by using multiple entries."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only a valid JSON array. No markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Parse [30, 30, 30] or similar
        match = re.search(r"\[[\d\s,]+\]", raw)
        if match:
            arr = json.loads(match.group())
            if isinstance(arr, list) and all(isinstance(x, (int, float)) for x in arr):
                result = [max(1, min(60, int(x))) for x in arr if x > 0]
                if not result:
                    return None
                diff = total_mins - sum(result)
                if diff != 0:
                    result[-1] = max(15, result[-1] + diff)
                if sum(result) == total_mins and all(15 <= x <= 60 for x in result):
                    return result
    except Exception:
        pass
    return None


def _weekday_dates_from_today() -> str:
    """Build explicit weekday->date mapping for the next 7 days from today."""
    from datetime import datetime, timedelta
    now = datetime.now().date()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    parts = []
    for i in range(7):
        d = now + timedelta(days=i)
        parts.append(f"{days[d.weekday()]}={d.strftime('%Y-%m-%d')}")
    return ", ".join(parts)


def _parse_weekday_in_text(text: str) -> str | None:
    """If text contains a weekday name, return YYYY-MM-DD for next occurrence. Deterministic."""
    from datetime import datetime, timedelta
    text_lower = text.lower()
    # Order by length desc so "wednesday" matches before "wed"
    weekdays = [
        ("wednesday", 2), ("thursday", 3), ("saturday", 5), ("tuesday", 1),
        ("monday", 0), ("friday", 4), ("sunday", 6),
        ("wed", 2), ("thu", 3), ("thurs", 3), ("sat", 5), ("tue", 1), ("tues", 1),
        ("mon", 0), ("fri", 4), ("sun", 6),
    ]
    now = datetime.now().date()
    for name, target_wd in weekdays:
        if name in text_lower:
            days_ahead = (target_wd - now.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7  # "today" -> next week
            d = now + timedelta(days=days_ahead)
            return d.strftime("%Y-%m-%d")
    return None


def parse_task_from_text(text: str) -> dict | None:
    """
    Parse natural language task input into structured fields.
    Example: "stats hw due feb 17 (optional: should take 2 hours, priority 3/5)"
    Returns dict with: name, due_date (YYYY-MM-DD), duration_hours, priority. Or None if parse failed.
    """
    from datetime import datetime
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    today_long = now.strftime("%A, %B %d, %Y")
    year = now.year
    weekday_map = _weekday_dates_from_today()

    prompt = f"""Parse this task into structured fields. Today is {today_long} ({today}).

Weekday to date (use EXACTLY these): {weekday_map}
Example: user says "Wednesday" -> use the Wednesday date from above. "Thursday" -> use the Thursday date. Do NOT confuse weekdays.

User input: "{text}"

Extract:
- name: short task name (e.g. "stats hw", "project proposal")
- due_date: YYYY-MM-DD. For weekdays use the mapping above. For "feb 17" use {year} (or {year+1} if before today).
- duration_hours: number if mentioned (e.g. "2 hours" -> 2, "30 min" -> 0.5). null if not mentioned
- priority: 1-5 if mentioned (e.g. "priority 3/5" -> 3). null if not mentioned

Return ONLY valid JSON: {{"name": "...", "due_date": "YYYY-MM-DD", "duration_hours": number|null, "priority": number|null}}
If the input cannot be parsed as a task with a due date, return {{"error": "brief reason"}}
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        data = _parse_json_from_llm(raw)
        if not data or data.get("error"):
            return None
        if not data.get("name") or not data.get("due_date"):
            return None
        # Override with deterministic weekday parse if present (LLM often gets weekdays wrong)
        weekday_date = _parse_weekday_in_text(text)
        if weekday_date:
            data["due_date"] = weekday_date
        return data
    except Exception:
        return None


def parse_event_from_text(text: str) -> dict | None:
    """
    Parse natural language event input.
    Example: "team meeting feb 14 2-3pm", "dentist feb 20 10am (1 hour)"
    Returns: name, date (YYYY-MM-DD), start_time (HH:MM), end_time (HH:MM). Or None.
    """
    from datetime import datetime
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    today_long = now.strftime("%A, %B %d, %Y")
    weekday_map = _weekday_dates_from_today()

    prompt = f"""Parse this as a calendar event. Today is {today_long} ({today}).

Weekday to date (use EXACTLY these): {weekday_map}
Example: "Wednesday" -> use the Wednesday date above. "Thursday" -> use the Thursday date above.

User input: "{text}"

Extract:
- name: short event name
- date: YYYY-MM-DD (use weekday mapping above for weekdays)
- start_time: HH:MM 24h (e.g. 14:00 for 2pm)
- end_time: HH:MM 24h

If user says "2-3pm" use 14:00-15:00. If only start given (e.g. "10am"), assume 1 hour duration.
Return ONLY valid JSON: {{"name": "...", "date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM"}}
If not parseable, return {{"error": "reason"}}"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        data = _parse_json_from_llm(resp.choices[0].message.content)
        if not data or data.get("error"):
            return None
        if not all(data.get(k) for k in ("name", "date", "start_time", "end_time")):
            return None
        # Override with deterministic weekday parse if present
        weekday_date = _parse_weekday_in_text(text)
        if weekday_date:
            data["date"] = weekday_date
        return data
    except Exception:
        return None


# =============================================================================
# PAPER SEARCH (Semantic Scholar API)
# =============================================================================

def search_papers(query: str, limit: int = 5) -> list:
    """
    Search for academic papers using Semantic Scholar API.
    Returns list of paper dicts with title, authors, year, abstract, url.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,url,citationCount"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        papers = []
        for paper in data.get("data", []):
            authors = paper.get("authors", [])
            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_names += " et al."
            
            papers.append({
                "title": paper.get("title", "Untitled"),
                "authors": author_names,
                "year": paper.get("year", "N/A"),
                "abstract": paper.get("abstract", "No abstract available."),
                "url": paper.get("url", ""),
                "citations": paper.get("citationCount", 0)
            })
        
        return papers
    except Exception as e:
        return [{"error": str(e)}]


def format_paper_results(papers: list) -> str:
    """Format paper search results as markdown."""
    if not papers:
        return "No papers found."
    
    if papers and "error" in papers[0]:
        return f"Error searching papers: {papers[0]['error']}"
    
    lines = ["## 📚 Paper Search Results\n"]
    
    for i, p in enumerate(papers, 1):
        lines.append(f"### {i}. {p['title']}")
        lines.append(f"**Authors:** {p['authors']}")
        lines.append(f"**Year:** {p['year']} | **Citations:** {p['citations']}")
        if p['url']:
            lines.append(f"**Link:** [{p['url']}]({p['url']})")
        lines.append("")
        
        # Truncate abstract if too long
        abstract = p['abstract'] or "No abstract available."
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(f"*{abstract}*")
        lines.append("\n---\n")
    
    return "\n".join(lines)
