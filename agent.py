#to run agent in terminal: python main.py
#to run agent in streamlit: streamlit run app.py and CTRL + C to close
# .venv\Scripts\activate to activate virtual environment in terminal (Windows)

import json
import requests
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME
from memory import retrieve_memory, retrieve_memory_with_metadata, store_memory, store_memory_if_unique, retrieve_hybrid_memory
from tools import run_python

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a Statistical Research Copilot named William.

You help with:
- mathematical proofs
- statistical modeling
- simulation design
- identifying assumptions
- suggesting research improvements (overarching goal is to help environment)

Use relevant past research memory if helpful.
Think step-by-step before answering.
Respond in Markdown. For math, use LaTeX with $ for inline (e.g., $x^2$) and $$ for blocks (e.g., $$\\int_0^1 f(x) dx$$).
Do NOT use \\( \\) or \\[ \\] for math - only use $ and $$.
Try to answer as succintly as possible as to not overwhelm the user, but can elaborate if needed.
"""

def build_personality_prompt(user_prefs: list, agent_traits: list) -> str:
    """
    Build a personality context block from retrieved preferences and traits.
    Returns a string to inject into the system prompt.
    """
    sections = []
    
    if user_prefs:
        prefs_text = "\n".join(f"- {p}" for p in user_prefs)
        sections.append(f"User preferences (adapt your style accordingly):\n{prefs_text}")
    
    if agent_traits:
        traits_text = "\n".join(f"- {t}" for t in agent_traits)
        sections.append(f"Your communication traits (what works well):\n{traits_text}")
    
    if not sections:
        return ""
    
    return "\n\n".join(sections)

def ask_agent(user_input: str, project: str, chat_history: list = None, project_goal: str = None):
    """
    Context hierarchy (highest to lowest priority):
    1. System prompt + Project goal
    2. Chat history (recent conversation)
    3. Project memories (specific to current project)
    4. Global memories (cross-project knowledge)
    5. Personality context (communication style)
    
    Returns: tuple (answer: str, citations: list of memory dicts)
    """
    # Retrieve memories with metadata for citation tracking
    project_mems = retrieve_memory_with_metadata(user_input, project=project, n_results=8)
    global_mems = retrieve_memory_with_metadata(user_input, project=None, n_results=4)
    
    # De-duplicate global (remove any that appear in project by text)
    project_texts = set(m["text"] for m in project_mems)
    global_mems = [m for m in global_mems if m["text"] not in project_texts]
    
    # Build text blocks for prompt
    project_block = "\n".join(f"- {m['text']}" for m in project_mems) if project_mems else "- (none)"
    global_block = "\n".join(f"- {m['text']}" for m in global_mems) if global_mems else "- (none)"
    
    # Combine all cited memories (excluding personality)
    all_citations = project_mems + global_mems

    # Retrieve personality context (global preferences and traits)
    user_prefs = retrieve_memory("user preferences communication style learning", project=None, n_results=3)
    agent_traits = retrieve_memory("agent personality traits approach style", project=None, n_results=3)
    personality_block = build_personality_prompt(user_prefs, agent_traits)

    # Build messages in priority order
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    # 1. Project goal (highest context priority)
    if project_goal:
        messages.append({"role": "system", "content": f"Current project: {project}\nProject goal: {project_goal}\n\nTailor your responses to help achieve this goal."})
    
    # 2. Chat history (recent conversation context)
    if chat_history:
        # Limit to last 10 messages (5 exchanges) to avoid token overflow
        recent_history = chat_history[-10:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 3. Project memories (specific to this project)
    messages.append({"role": "system", "content": f"Project memories ('{project}'):\n{project_block}"})
    
    # 4. Global memories (cross-project knowledge)
    messages.append({"role": "system", "content": f"Global memories:\n{global_block}"})
    
    # 5. Personality context (communication style - lowest priority)
    if personality_block:
        messages.append({"role": "system", "content": personality_block})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    answer = resp.choices[0].message.content

    # Learn from this conversation (adaptive personality - stays automatic)
    analyze_conversation_style(user_input, answer)

    return answer, all_citations

def execute_natural_language(task: str, project: str):
    """
    Natural language -> Python code -> execute.
    """
    prompt = f"""
    Write Python code to accomplish the task below.
    Rules:
    - Only output code (no explanations and no Python markdown formatting).
    - Use matplotlib for plots if needed.
    - Do NOT call plt.show().
    - If plotting, create the figure normally (Streamlit will render it).
    Task: {task}
    Notes:
    - Use standard libraries when possible.
    - If you generate plots, save them to 'plot.png' in the current directory.
    """
    code_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You write clean Python code for statistics and data analysis."},
            {"role": "user", "content": prompt},
        ]
    )
    code = code_resp.choices[0].message.content
    output = run_python(code)

    # 🧠 Memory gate - store multiple memories with duplicate detection
    memories_to_store = should_store_simulation(task, code, output, project)
    for mem in memories_to_store:
        if mem.get("text"):
            store_memory_if_unique(
                text=mem["text"],
                metadata={
                    "name": mem.get("name", ""),
                    "type": mem.get("type", "result"),
                    "importance": int(mem.get("importance", 3)),
                    "source": "simulation"
                },
                project=project
            )

    return output, code

def should_store_memory(user_input: str, answer: str, project: str):
    """
    Analyze conversation and extract multiple memories worth storing.
    Returns a list of memory objects to store.
    """
    gate_prompt = f"""
    You are deciding what should be saved into long-term memory for a Statistical Research Copilot.

    Project: {project}

    Conversation:
    User: {user_input}
    Assistant: {answer}

    Extract ALL distinct pieces of knowledge worth remembering:
    - Definitions (formal definitions of terms/concepts)
    - Theorems and formulas
    - Functions (reusable code functions worth saving)
    - Examples (worked examples, illustrations of concepts)
    - Insights and key observations
    - Assumptions made
    - Decisions or conclusions
    - Results and findings
    - References mentioned

    Do NOT store:
    - Trivial Q&A or small talk
    - Vague statements without specific content
    - Duplicates of the same concept

    Return JSON with a "memories" array. Each memory should have:
    - name (a short descriptive title, e.g. "Central Limit Theorem", "bootstrap_ci", "CLT Example")
    - type (one of: definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology)
    - importance (1-5)
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
        {{"name": "Random Variable Definition", "type": "definition", "importance": 4, "text": "**Random Variable**\\n\\nA random variable is a function that maps outcomes from a sample space to real numbers. Denoted as $X: \\\\Omega \\\\to \\\\mathbb{{R}}$."}},
        {{"name": "bootstrap_ci", "type": "function", "importance": 4, "text": "**bootstrap_ci**\\n\\nComputes bootstrap confidence interval for a dataset.\\n\\n```python\\ndef bootstrap_ci(data, n_boot=1000, alpha=0.05):\\n    import numpy as np\\n    samples = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boot)]\\n    return np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])\\n```"}},
        {{"name": "CLT Example - Dice", "type": "example", "importance": 3, "text": "**Example: CLT with Dice**\\n\\nRolling a fair die 100 times: $\\\\mu = 3.5$, $\\\\sigma^2 = 35/12$.\\n\\nBy CLT, $\\\\bar{{X}} \\\\sim N(3.5, 35/1200) = N(3.5, 0.029)$"}}
    ]}}
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON with a 'memories' array. Format formulas with $$...$$ on their own line for block display. Use $...$ for inline math only."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return data.get("memories", [])
    except Exception:
        # fail safe: return empty list if parsing fails
        return []

def should_store_simulation(task: str, code: str, output: str, project: str):
    """
    Analyze simulation results and extract multiple memories worth storing.
    Returns a list of memory objects to store.
    """
    gate_prompt = f"""
    You are deciding what should be saved into long-term research memory from a computational result.

    Project: {project}

    Task: {task}
    Generated Code:
    {code}

    Output:
    {output}

    Extract ALL distinct pieces of knowledge worth remembering:
    - Meaningful statistical results or findings
    - Useful methodology or code patterns
    - Reusable functions from the code
    - Key insights from the output
    - Decisions based on the results
    - Any formulas or equations discovered/used
    - Examples that could be useful later

    Do NOT store:
    - Trivial calculations or simple arithmetic
    - Temporary debugging information
    - Redundant or duplicate information

    Return JSON with a "memories" array. Each memory should have:
    - name (a short descriptive title, e.g. "Monte Carlo Pi Estimate", "bootstrap_ci", "Bootstrap CI Example")
    - type (one of: definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology)
    - importance (1-5)
    - text (the full content - see formatting rules below)

    **FORMATTING RULES for text field:**
    - For formulas/equations: Put the formula name first, then use $$...$$ for the main equation on its own line
    - Use $...$ only for inline math within sentences
    - For examples: Clearly label as "Example:" and show the worked solution
    - For functions: Use function name as title, include brief description, wrap code in ```python blocks

    If nothing is worth storing, return {{"memories": []}}
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON with a 'memories' array. Format formulas with $$...$$ on their own line for block display. Use $...$ for inline math only."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("memories", [])
    except Exception:
        return []

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

    # Store user preference if flagged
    if data.get("store_preference") and data.get("preference_text"):
        store_memory(
            text=data["preference_text"],
            metadata={
                "type": "user_preference",
                "confidence": confidence,
                "source": "style_analysis"
            },
            project=None  # Global - applies across all projects
        )

    # Store agent trait if flagged
    if data.get("store_trait") and data.get("trait_text"):
        store_memory(
            text=data["trait_text"],
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

def extract_memories_from_exchange(user_input: str, answer: str, project: str) -> int:
    """
    On-demand: Extract and store memories from a chat exchange.
    Called when user clicks "Save to Memory" button.
    Returns count of memories stored.
    """
    memories = should_store_memory(user_input, answer, project)
    stored_count = 0
    
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
            if result:  # Not a duplicate
                stored_count += 1
    
    return stored_count

def extract_memories_from_text(text: str, project: str | None) -> int:
    """
    On-demand: Extract and store memories from arbitrary text input.
    Called from the "Add Memory" tab.
    Returns count of memories stored.
    """
    gate_prompt = f"""
    You are extracting knowledge to save into long-term memory for a Statistical Research Copilot.

    Project: {project or "Global"}

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

    Do NOT store:
    - Vague statements without specific content
    - Duplicates of the same concept
    - Trivial or obvious information

    Return JSON with a "memories" array. Each memory should have:
    - name (a short descriptive title, e.g. "Central Limit Theorem", "bootstrap_ci", "CLT Example")
    - type (one of: definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology)
    - importance (1-5)
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
            {"role": "system", "content": "Return only valid JSON with a 'memories' array. Format formulas with $$...$$ on their own line for block display. Use $...$ for inline math only."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    try:
        data = json.loads(resp.choices[0].message.content.strip())
        memories = data.get("memories", [])
    except Exception:
        return 0

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
                "type": "insight",
                "importance": 4,
                "source": "conversation_summary"
            },
            project=project
        )
    
    return summary


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
