# William - Statistical Research Copilot

An AI-powered research assistant with persistent memory, file ingestion, and Python code execution.

## Features

- **Chat with Memory** - Conversations are enhanced with relevant context from your research history
- **File Ingestion** - Upload PDFs, CSVs, and text files to build your knowledge base
- **Code Execution** - Run Python code directly with `run:` or generate code from natural language with `nlrun:`
- **Memory Dashboard** - Search, filter, edit, and manage your stored memories
- **Memory Map** - Visualize your memories in 2D semantic space to see clusters and connections
- **Project Organization** - Keep research organized by project with optional goals
- **Project Management** - Delete projects or merge them together
- **Adaptive Personality** - The agent learns your communication preferences over time
- **Multiple Memory Types** - Store definitions, theorems, formulas, functions, examples, insights, and more

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

### Save to Memory
Click the 💾 **Save to Memory** button on any assistant message to extract and store important knowledge (theorems, insights, formulas, etc.).

### Add Memory
Use the **Add Memory** tab to paste text (notes, formulas, excerpts, code) and have the agent extract structured memories from it. The agent will automatically categorize content as definitions, theorems, formulas, functions, examples, etc.

### File Ingestion
Go to the **Files** tab to upload PDFs, CSVs, or text files. They'll be chunked and stored in your memory for retrieval.

### Memory Dashboard
Use the **Memory Dashboard** tab to search, filter, and manage your stored memories by type, date, or keyword. You can also edit existing memories.

**Memory types include:** definition, theorem, formula, function, example, insight, assumption, decision, result, reference, methodology

### Memory Map
Use the **Memory Map** tab to visualize all your memories in a 2D semantic space. Similar memories cluster together, making it easy to see relationships. Click on any point to view the full memory content.

- Adjust the **spread factor** to emphasize distances between dissimilar memories
- Connection lines show clusters of related memories
- Color by type, source, or importance

### Project Management
- **Create projects** - Organize memories by research topic
- **Set project goals** - Add optional goals that provide context to the agent
- **Delete projects** - Remove a project and all its associated memories
- **Merge projects** - Combine two projects, moving all memories from source to target

## Project Structure

```
william/
├── app.py              # Main Streamlit application
├── agent.py            # LLM agent and memory gating logic
├── memory.py           # ChromaDB memory storage/retrieval
├── storage.py          # Local JSON storage for projects/chat
├── tools.py            # Python code execution
├── ingest.py           # Text chunking utilities
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
