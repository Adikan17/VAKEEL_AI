import os
import json
import pickle
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -------------------------
# Config
# -------------------------
CHROMA_PATH = "chroma_db"  # folder created by build_db.py
CHROMA_COLLECTION = "indian_legal_docs_final"  # use your actual collection name
BM25_PATH = "bm25_index.pkl"
CHUNKS_PATH = "processed_legal_data.json"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # downloaded once, then cached locally
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # local Ollama endpoint
LLM_MODEL = "llama3"  # you pulled with: ollama pull llama3

# -------------------------
# Init Flask
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# Warmup: load models and indexes
# -------------------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading ChromaDB...")
client = PersistentClient(path=CHROMA_PATH)
chroma_collection = client.get_collection(CHROMA_COLLECTION)

print("Loading BM25 index...")
with open(BM25_PATH, "rb") as f:
    bm25_index = pickle.load(f)

print("Loading chunks JSON...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
# Expect each item like:
# {"chunk_id": "...", "source_document": "...", "document_type": "...", "section_id": "...", "chunk_text": "..."}

print("Backend ready.")

# -------------------------
# Helpers
# -------------------------
def embed(text: str):
    return embedder.encode([text], convert_to_tensor=False)[0].tolist()

def vector_search(query: str, k: int = 10):
    qv = embed(query)
    res = chroma_collection.query(query_embeddings=[qv], n_results=k)
    # documents are strings (chunk_texts)
    docs = res.get("documents", [[]])[0] if res and "documents" in res else []
    return docs

def bm25_topk(query: str, k: int = 10):
    # bm25_index was trained over a tokenized corpus matching all_chunks order
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    idx = np.argsort(scores)[-k:]
    # map indices to chunk_texts; your JSON uses "chunk_text"
    safe = []
    for i in idx:
        try:
            safe.append(all_chunks[int(i)]["chunk_text"])
        except Exception:
            pass
    return safe

def fuse_and_trim(vec_chunks, bm25_chunks, k=12, max_chars=8000):
    # de-dup while preserving order
    seen = set()
    merged = []
    for c in (vec_chunks + bm25_chunks):
        if not c or c in seen:
            continue
        seen.add(c)
        merged.append(c)
        if len(merged) >= k:
            break
    # trim by char budget
    out = []
    total = 0
    for c in merged:
        if total + len(c) > max_chars:
            break
        out.append(c)
        total += len(c)
    return out

def build_prompt(context_chunks, query):
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are a legal assistant for Indian law. Answer ONLY using the provided context. "
        "Cite acts/sections if present. If the answer is not in context, say so briefly.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    )

def call_ollama(prompt: str, model: str = LLM_MODEL, temperature: float = 0.2, max_tokens: int = 768):
    # Use non-streaming to get a single JSON response
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"response": "...", ...}
    return data.get("response", "").strip()

# -------------------------
# Routes
# -------------------------
@app.get("/")
def health():
    return jsonify({"message": "Local Legal AI Backend (Ollama) running"})

@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400

        # Retrieval
        vec_chunks = vector_search(query, k=10)
        bm25_chunks = bm25_topk(query, k=10)

        if not vec_chunks and not bm25_chunks:
            return jsonify({
                "answer": "The local knowledge base does not contain relevant material for this query.",
                "confidence": 0.0
            })

        fused = fuse_and_trim(vec_chunks, bm25_chunks, k=12, max_chars=8000)
        prompt = build_prompt(fused, query)

        # Local LLM via Ollama
        answer = call_ollama(prompt)

        return jsonify({
            "answer": answer,
            "used_chunks": min(len(fused), 12),
            "confidence": "offline_local_model"
        })

    except requests.HTTPError as e:
        return jsonify({"error": f"Ollama HTTP error: {e}", "detail": getattr(e, 'response', None).text if hasattr(e, 'response') else ""}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Port 7860 as requested
    app.run(host="0.0.0.0", port=7860)
