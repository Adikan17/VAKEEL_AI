# embeddings/chunker.py
def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks
