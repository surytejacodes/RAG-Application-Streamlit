import faiss, pickle, numpy as np
from pathlib import Path

class VectorStore:
    def __init__(self, dim=1024):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metadata):
        self.index.add(np.array(embeddings, dtype='float32'))
        self.metadata.extend(metadata)

    def search(self, embedding, k=5):
        D, I = self.index.search(np.array([embedding], dtype='float32'), k)
        return [self.metadata[i] for i in I[0] if i < len(self.metadata)]

    def save(self, folder):
        folder = Path(folder)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        with open(folder / "meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, folder):
        folder = Path(folder)
        self.index = faiss.read_index(str(folder / "index.faiss"))
        with open(folder / "meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)
