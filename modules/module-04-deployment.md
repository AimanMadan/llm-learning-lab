# Module 04 • Evaluation & Deployment

**Goal:** Ship a small LLM-backed API with monitoring.

## 1. Topics
- Retrieval-Augmented Generation (RAG) basics
- Latency + cost tradeoffs
- Observability: tracing, logging prompts, redaction
- Human-in-the-loop review workflows

## 2. Build steps
1. Stand up a FastAPI or Flask service exposing `/chat` and `/health`.
2. Add a vector store (e.g., `chromadb`) for a simple RAG loop.
3. Instrument with `langsmith`, `evidently`, or simple JSON logging.
4. Containerize with Docker + include a `render.yaml` or `fly.toml` sample.

## 3. Checklist before sharing
- [ ] Prompt + response logging with PII redaction
- [ ] Rate limiting or API key enforcement
- [ ] Unit tests for utility functions (e.g., text chunking)
- [ ] README instructions for running locally and via Docker

> ✅ **Deliverable:** `projects/rag-chat/service.py` plus deployment notes in `projects/rag-chat/README.md`.
