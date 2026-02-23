"""
RAG Chat Service

A simple RAG-powered chat API using FastAPI and ChromaDB.

Usage:
    uvicorn service:app --reload

Endpoints:
    POST /chat - Send a message and get a RAG-enhanced response
    GET /health - Health check
    POST /ingest - Add documents to the knowledge base
"""

import os
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "knowledge_base"

# Initialize FastAPI
app = FastAPI(
    title="RAG Chat API",
    description="Simple RAG-powered chat service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    max_tokens: int = 500


class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    tokens_in: int
    tokens_out: int
    latency_ms: float


class IngestRequest(BaseModel):
    documents: List[dict]  # [{"content": "...", "metadata": {...}}]


class HealthResponse(BaseModel):
    status: str
    collection_count: int
    timestamp: str


# Helper functions
def get_embedding(text: str) -> List[float]:
    """Get embedding for text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def query_knowledge_base(query: str, top_k: int = 3) -> List[dict]:
    """Query the knowledge base for relevant documents."""
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    sources = []
    for i, doc in enumerate(results["documents"][0]):
        sources.append({
            "content": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    
    return sources


def build_context(sources: List[dict]) -> str:
    """Build context string from retrieved documents."""
    context_parts = []
    for i, source in enumerate(sources, 1):
        context_parts.append(f"[{i}] {source['content']}")
    return "\n\n".join(context_parts)


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        collection_count=collection.count(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message with RAG enhancement."""
    import time
    start = time.time()
    
    # Query knowledge base
    sources = query_knowledge_base(request.message, top_k=request.top_k)
    
    if not sources:
        # No relevant documents found
        context = "No relevant information found in the knowledge base."
        source_texts = []
    else:
        context = build_context(sources)
        source_texts = [s["metadata"].get("source", "unknown") for s in sources]
    
    # Build prompt
    system_prompt = """You are a helpful assistant. Use the provided context to answer questions accurately.
If the context doesn't contain relevant information, say so clearly.
Always cite your sources when using information from the context."""

    user_prompt = f"""Context:
{context}

Question: {request.message}

Please answer the question using the context above. If you use information from the context, mention which source it came from."""

    # Call LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=request.max_tokens,
        temperature=0.7,
    )
    
    latency_ms = (time.time() - start) * 1000
    
    return ChatResponse(
        response=response.choices[0].message.content,
        sources=source_texts,
        tokens_in=response.usage.prompt_tokens,
        tokens_out=response.usage.completion_tokens,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Add documents to the knowledge base."""
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for i, doc in enumerate(request.documents):
        doc_id = f"doc_{datetime.utcnow().timestamp()}_{i}"
        content = doc["content"]
        metadata = doc.get("metadata", {})
        
        ids.append(doc_id)
        documents.append(content)
        metadatas.append(metadata)
        embeddings.append(get_embedding(content))
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    
    return {
        "status": "success",
        "documents_added": len(ids),
        "total_documents": collection.count(),
    }


@app.delete("/clear")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base."""
    global collection
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return {"status": "cleared", "remaining": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
