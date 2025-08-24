# FastAPI RAG Demo

This project provides a FastAPI backend and a simple frontend to demonstrate uploading code files,
generating ASTs, indexing via Qdrant, and answering questions via a Retrieval-Augmented Generation (RAG) workflow.

## Prerequisites

- Docker (for Qdrant)
- [Ollama](https://ollama.com/) (for local LLM inference)
- Python 3.8+

## Setup

1. **Install Python dependencies**
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. **Run Qdrant via Docker**
   ```bash
   docker volume create qdrant-storage
   docker volume create qdrant-snapshots

   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant-storage:/qdrant/storage -v qdrant-snapshots:/qdrant/snapshots qdrant/qdrant:latest
   ```

3. **建立範例設定檔**
   ```bash
   cp env.txt.example env.txt
   # 編輯 env.txt，填入您的 Qdrant、Embedding、OLLAMA 或 LLM 參數
   ```
4. **Pull Ollama models**
   ```bash
   ollama pull BGE-M3
   ollama pull deepseek-coder:6.7b
   ```

## Frontend Development (Vite)

Before starting the backend server, you can run the frontend locally with Vite:

```bash
cd frontend
npm install
npm run dev
npm run build
```

## Running the Server

```bash
#uvicorn backend.main:app --reload --port 8000
python3 scripts/run_server.py
```

## Frontend

Open `frontend/index.html` in your browser. Use the interface to upload a code directory and ask questions.

You can check the **Force rebuild** checkbox to force re-indexing even if an existing RAG index already exists.

## API Endpoints

- `POST /upload`: Upload multiple files (form field `files`) to generate AST and build the index.
- `POST /ask`: Send JSON `{ "question": "..." }` to retrieve an answer. Supports markdown and mermaid in the response.

## Testing

Run pytest to verify the basic endpoints:
```bash
pytest -q
```
