# William - Statistical Research Copilot

An AI-powered research assistant with persistent memory, file ingestion, and Python code execution.

## Features

- **Chat with Memory** - Conversations are enhanced with relevant context from your research history
- **File Ingestion** - Upload PDFs, CSVs, and text files to build your knowledge base
- **Code Execution** - Run Python code directly with `run:` or generate code from natural language with `nlrun:`
- **Memory Dashboard** - Search, filter, and manage your stored memories
- **Project Organization** - Keep research organized by project with tagged memories

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

### Pin Important Info
Click the 📌 button on any assistant message to save it as a core memory.

### File Ingestion
Go to the **Files** tab to upload PDFs, CSVs, or text files. They'll be chunked and stored in your memory for retrieval.

### Memory Dashboard
Use the **Memory** tab to search, filter, and manage your stored memories by type, date, or keyword.

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
└── .env                # Your API key (not committed)
```

## Requirements

- Python 3.9+
- OpenAI API key

## License

MIT
