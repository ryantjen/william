#to run agent in terminal: python main.py
#to run agent in streamlit: streamlit run app.py and CTRL + C to close
# .venv\Scripts\activate to activate virtual environment in terminal (Windows)

from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME
from memory import retrieve_memory, store_memory
from tools import run_python
from memory import retrieve_hybrid_memory
import json

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a Statistical Research Copilot.

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

def ask_agent(user_input: str, project: str):
    memories = retrieve_hybrid_memory(user_input, project=project, n_project=8, n_global=2)
    memory_block = "\n\n".join(f"- {m}" for m in memories) if memories else "- (none)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Relevant memory (project='{project}' prioritized):\n{memory_block}"},
        {"role": "user", "content": user_input},
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    answer = resp.choices[0].message.content

    gate = should_store_memory(user_input, answer, project)

    if gate.get("store") is True:
        store_memory(
            text=gate.get("memory_text", ""),
            metadata={
                "type": gate.get("type", "insight"),
                "importance": int(gate.get("importance", 3)),
                "source": "chat"
            },
            project=project
        )
    return answer

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

    # 🧠 Memory gate
    gate = should_store_simulation(task, code, output, project)

    if gate.get("store") is True:
        store_memory(
            text=gate.get("memory_text", ""),
            metadata={
                "type": gate.get("type", "result"),
                "importance": int(gate.get("importance", 3)),
                "source": "simulation"
            },
            project=project
        )

    return output, code

def should_store_memory(user_input: str, answer: str, project: str):
    gate_prompt = f"""
    You are deciding what should be saved into long-term memory for a Statistical Research Copilot.

    Project: {project}

    Conversation:
    User: {user_input}
    Assistant: {answer}

    Only store if it will be useful later (theorem/insight/assumption/decision/result/reference).
    Do NOT store trivial Q&A, small clarifications, or duplicates.

    Return JSON with keys:
    - store (true/false)
    - type (one of: theorem, insight, assumption, decision, result, reference)
    - importance (1-5)
    - memory_text (a concise standalone note to store; for math use $...$ for inline and $$...$$ for block LaTeX)
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON. For any math in memory_text, use $...$ for inline LaTeX and $$...$$ for block LaTeX."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return data
    except Exception:
        # fail safe: don't store if parsing fails
        return {"store": False}

def should_store_simulation(task: str, code: str, output: str, project: str):
    gate_prompt = f"""
    You are deciding whether a computational result should be saved into long-term research memory.

    Project: {project}

    Task: {task}
    Generated Code:
    {code}

    Output:
    {output}

    Only store if:
    - The result reveals a meaningful statistical insight
    - It changes modeling decisions
    - It provides reusable methodology
    - It demonstrates non-trivial simulation behavior

    Do NOT store:
    - Trivial calculations
    - Simple arithmetic
    - Temporary debugging runs
    - Redundant experiments

    Return JSON:
    - store (true/false)
    - type (result | methodology | insight | decision)
    - importance (1-5)
    - memory_text (concise standalone note; for math use $...$ for inline and $$...$$ for block LaTeX)
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON. For any math in memory_text, use $...$ for inline LaTeX and $$...$$ for block LaTeX."},
            {"role": "user", "content": gate_prompt},
        ]
    )

    import json
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"store": False}
