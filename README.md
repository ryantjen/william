# William - Learning & Research Copilot

An AI-powered learning and research assistant with persistent memory, file ingestion, and Python code execution. Works across any subject (math, science, biology, history, programming, etc.).

## Features

### Chat & Memory
- **Chat with Memory** - Conversations use the last 20 messages plus relevant memories from your research history
- **Citation Tracking** - Each response shows Sources (memories referenced), displayed before Save to Memory
- **Memory Dashboard** - Search, filter by importance (1–5), type, date; select memories; bulk delete; prune low-importance or neglected
- **Memory Storage Health** - Popover shows total memories, disk usage, and embedding model
- **Memory Export** - Export selected memories as JSON, Markdown, or LaTeX via an Export popover
- **Memory Map** - Visualize your memories in 2D semantic space (UMAP); color by type, source, or importance
- **Multiple Memory Types** - Store definitions, theorems, formulas, functions, examples, insights, user goals, knowledge level, and more
- **Scope Toggle** - View memories for "This project" or "Global"

### Code & Papers
- **Code Execution** - Run Python code directly with `run:` or generate code from natural language with `nlrun:`
- **Paper Search** - Search academic papers with `papers:` using Semantic Scholar *(faulty for now; will update later)*

### Files & Ingestion
- **Smart File Ingestion** - Upload PDFs, CSVs, and text files with intelligent chunking and table understanding

### Calendar
- **Tasks & Events** - Add tasks (with hours) and events via natural language
- **Daily Chunk Board** - One chunk per task per day; mark done, set expected time remaining, or mark entire task complete
- **Task Completion %** - Calendar shows completion percentage per task
- **Manage Tasks** - View tasks with completion status; auto-delete past-due tasks and past events

### Review & Summarization
- **Spaced Repetition** - Review tab with flashcard-style quiz for memorization
- **Conversation Summaries** - Summarize chat sessions and save to memory

### Projects & Personality
- **Project Organization** - Keep research organized by project with optional goals
- **Project Management** - Create, rename, delete, or merge projects
- **Adaptive Personality** - Learns your communication preferences; supersedes conflicting preferences when you change them
- **User Context** - Store goals and knowledge level (`user_goal`, `user_knowledge`); the agent adapts explanations accordingly

### Documentation
- **📖 Docs Button** - Full documentation in top-right corner; comprehensive Memory System doc (storage, retrieval, decay, priority)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/william.git
cd william
```

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys).

### 5. Run the app

**Quick Start (Recommended):**

*Windows:*
```bash
run_agent.bat
```
Double-click the file or run it from command prompt.

*Mac/Linux:*
```bash
chmod +x run_agent.sh   # First time only - make it executable
./run_agent.sh
```

**Manual Start:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

### Chat
Just type normally to chat with the AI assistant. It will use relevant memories from your project to enhance responses.

### Run Python Code
```
run: print("Hello, World!")
```

### Natural Language to Code
```
nlrun: plot a sine wave from 0 to 2pi
```

### Search Academic Papers
```
papers: bootstrap confidence intervals
```
*Faulty for now — will update later.* Intended to search Semantic Scholar for relevant papers and save results to memory.

### Save to Memory
Click the 💾 **Save to Memory** button on any assistant message to extract and store important knowledge (theorems, insights, formulas, etc.). If your project has a goal set (e.g. "learn about biology"), that goal influences what gets stored and how important it is.

### Add Memory
Use the **Add Memory** tab to paste text (notes, formulas, excerpts, code) and have the agent extract structured memories from it. The agent works for any subject (math, biology, history, etc.). Project goals influence what gets stored and importance. Content is automatically categorized as definitions, theorems, formulas, functions, examples, user goals, knowledge level, etc. Use the **🗑️ Clear** button to clear the text box.

### File Ingestion
Go to the **Files** tab to upload PDFs, CSVs, or text files. Files are processed with smart chunking that respects natural boundaries (sections, paragraphs, sentences).

**CSV files** get special treatment:
- Automatic dataset summary (rows, columns, stats)
- Column-by-column analysis with statistics
- Correlation detection for numeric columns
- Sample data for semantic search

**Manage ingested files:**
- View all ingested files with metadata
- Clear individual file records
- Clear all records when memories are deleted

### Memory Dashboard
Use the **Memory Dashboard** tab to search, filter, and manage your stored memories.

**Storage** — Popover shows total memories, disk usage, embedding model

**Filters:**
- **Scope** — View "This project" or "Global" memories
- **Importance** — Filter by score (1–5)
- **Type, date, keyword** — Standard search and filter

**Prune** — Remove low-importance memories (≤2) or "neglected" only (low importance + old + never cited)

**Selection & actions:**
- Select individual memories or use **Select all** / **Clear**
- **Delete selected** to bulk remove memories

**Export** (only selected memories):
- 📥 JSON - Full data export for backup or processing
- 📥 Markdown - Human-readable format
- 📥 LaTeX - Ready to compile for papers/documents

**Memory types include:** definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology, user_goal, user_knowledge, user_preference, agent_trait

### Memory Map
Use the **Memory Map** tab to visualize all your memories in a 2D semantic space. Similar memories cluster together, making it easy to see relationships. Click on any point to view the full memory content.

- Adjust the **spread factor** to emphasize distances between dissimilar memories
- Connection lines show clusters of related memories
- Color by type, source, or importance

### Calendar
Use the **Calendar** tab to manage tasks and events.

**Add tasks/events** — Natural language: "stats hw due wednesday, 2 hours, priority 3/5" or "meeting feb 14 2-3pm"

**Daily chunk board** — Shows one chunk per task for today. Locked to today (no day switching).

- **✔ Done** — Mark chunk complete (chunk disappears)
- **⏱** — Set expected time to complete the rest of the assignment; updates task hours and completion %
- **✓ All** — Mark entire task as completed (100%)

**Manage tasks** — View tasks with completion status and total hours. Past-due tasks and past events auto-delete when the calendar loads.

### Project Management
- **Create projects** - Organize memories by research topic
- **Set project goals** - Add optional goals that provide context to the agent and influence memory extraction (what gets stored and importance)
- **Rename projects** - Change a project's name; memories, chat, and ingestion records are migrated
- **Delete projects** - Remove a project and all its associated memories
- **Merge projects** - Combine two projects, moving all memories from source to target

### Review Mode (Spaced Repetition)
Use the **Review** tab to test your knowledge with flashcard-style quizzes:
- Select which memory types to review (definitions, theorems, formulas, etc.)
- Cards show the name/type first - try to recall before revealing
- Mark cards as "I knew it" or "Review again"
- Track your accuracy and progress

### Conversation Tools
- **📝 Summarize** - Condense the current chat session into key points (saved to memory)
- **📤 Export chat** - Download chat history as Markdown or LaTeX
- **🗑️ Clear chat** - Start fresh while keeping your memories

### Citation Tracking
Each response shows **Sources** — which memories were referenced — so you can see what influenced the agent's answer. Sources appear above the Save to Memory button and persist across reruns. Click to expand and view the cited memories.

### Documentation
Click the **📖 Docs** button in the top-right corner to open full documentation. Includes a comprehensive Memory System guide (storage, retrieval, decay, priority) plus feature summaries for Chat, Files, Add Memory, Memory Dashboard, Memory Map, Calendar, and Review.

## Project Structure

```
william/
├── app.py              # Main Streamlit application
├── agent.py            # LLM agent and memory gating logic
├── memory.py           # ChromaDB memory storage/retrieval
├── storage.py          # Local JSON storage for projects/chat
├── calendar_data.py    # Tasks, events, chunks (calendar.json)
├── docs.py             # Documentation content for Docs button
├── tools.py            # Python code execution
├── ingest_files.py     # File-specific ingestion (PDF, CSV, TXT)
├── config.py           # Configuration and API settings
├── requirements.txt    # Python dependencies
├── run_agent.bat       # Quick start script (Windows)
├── run_agent.sh        # Quick start script (Mac/Linux)
└── .env                # Your API key (not committed)
```

## Requirements

- Python 3.9+
- OpenAI API key

## License

MIT
