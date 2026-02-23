# RAG Chat API

A simple Retrieval-Augmented Generation (RAG) chat service built with FastAPI and ChromaDB.

## Features

- 🔍 **Semantic Search**: Find relevant documents using embeddings
- 💬 **Chat Interface**: Natural language Q&A with context
- 📥 **Document Ingestion**: Add documents via API
- 🐳 **Docker Ready**: Containerized for easy deployment

## Quick Start

### Local Development

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install fastapi uvicorn openai chromadb python-multipart

# Set environment variables
export OPENAI_API_KEY="your-api-key"

# Run the server
uvicorn service:app --reload
```

### Docker

```bash
# Build
docker build -t rag-chat .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key rag-chat
```

## API Endpoints

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "collection_count": 10,
  "timestamp": "2024-01-15T10:30:00"
}
```

### `POST /chat`

Send a message and get a RAG-enhanced response.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "top_k": 3
  }'
```

Response:
```json
{
  "response": "Based on the documentation...",
  "sources": ["ml-basics.pdf", "ai-guide.pdf"],
  "tokens_in": 250,
  "tokens_out": 150,
  "latency_ms": 850.5
}
```

### `POST /ingest`

Add documents to the knowledge base.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Machine learning is a subset of AI...",
        "metadata": {"source": "ml-intro.txt"}
      }
    ]
  }'
```

### `DELETE /clear`

Clear all documents from the knowledge base.

```bash
curl -X DELETE http://localhost:8000/clear
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHAT_MODEL` | `gpt-3.5-turbo` | Chat completion model |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |

## Project Structure

```
rag-chat/
├── service.py      # Main FastAPI application
├── README.md       # This file
├── Dockerfile      # Docker configuration
├── requirements.txt
└── tests/
    └── test_service.py
```

## Deployment

### Render

```yaml
# render.yaml
services:
  - type: web
    name: rag-chat
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn service:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
```

### Fly.io

```toml
# fly.toml
app = "rag-chat"
primary_region = "iad"

[build]
  builder = "heroku/buildpacks:20"

[env]
  PORT = "8080"

[[services]]
  internal_port = 8080
  protocol = "tcp"
```

## Monitoring

The service logs all requests. For production:

1. Add structured logging (JSON format)
2. Integrate with a monitoring service (Datadog, Grafana, etc.)
3. Track metrics: latency, token usage, error rates
4. Set up alerts for failures

## Checklist Before Deploying

- [ ] Set `OPENAI_API_KEY` securely
- [ ] Configure rate limiting
- [ ] Add authentication (API keys)
- [ ] Set up persistent storage for ChromaDB
- [ ] Add request logging with PII redaction
- [ ] Write tests for critical paths
- [ ] Configure CORS for your frontend

## Learning Goals

After completing this project, you should understand:

1. How RAG systems combine retrieval with generation
2. Vector embeddings and similarity search
3. Building REST APIs with FastAPI
4. Containerization and deployment basics
