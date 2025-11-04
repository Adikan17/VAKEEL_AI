# indexer/faiss_indexer.py
import faiss
import numpy as np
import os
import pickle

INDEX_PATH = "./faiss_index.bin"
META_PATH = "./faiss_meta.pkl"

class FaissIndexer:
    def __init__(self, dim):
        self.dim = dim
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.meta = pickle.load(f)
        else:
            # use Inner Product on normalized vectors as cosine:
            self.index = faiss.IndexFlatIP(dim)
            self.meta = []  # list of metadata dicts parallel to vectors

    def add(self, vectors, metadatas):
        # normalize vectors for cosine similarity
        faiss.normalize_L2(np.asarray(vectors))
        self.index.add(np.asarray(vectors).astype('float32'))
        self.meta.extend(metadatas)
        self.save()

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.meta, f)

    def search(self, q_vector, topk=5):
        v = q_vector.astype('float32')
        faiss.normalize_L2(v)
        D, I = self.index.search(np.expand_dims(v, 0), topk)
        results = []
        for idx in I[0]:
            if idx == -1: continue
            results.append(self.meta[idx])
        return results
