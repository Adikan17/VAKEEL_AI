# embeddings/embedder.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # fast small model

def embed_texts(list_of_texts):
    vectors = model.encode(list_of_texts, show_progress_bar=True, convert_to_numpy=True)
    return vectors
