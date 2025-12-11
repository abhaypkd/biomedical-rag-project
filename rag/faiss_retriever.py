import json
import faiss
import numpy as np


class FaissRetriever:

    def __init__(self, embeddings_file="data/pdf_embeddings.json"):
        self.embeddings_file = embeddings_file
        self.embeddings = []
        self.chunk_metadata = []
        self.index = None

        self._load_embeddings()
        self._build_faiss_index()

    def _load_embeddings(self):
        with open(self.embeddings_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_embeddings = []
        chunk_info = []

        for entry in data:
            pdf_file = entry["file"]

            for ch in entry["chunks"]:
                all_embeddings.append(ch["embedding"])
                chunk_info.append({
                    "pdf": pdf_file,
                    "chunk_id": ch["chunk_id"],
                    "text": ch["text"]
                })

        self.embeddings = np.array(all_embeddings, dtype="float32")
        self.chunk_metadata = chunk_info

    def _build_faiss_index(self):
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            info = self.chunk_metadata[idx]
            info["distance"] = float(score)    # cosine similarity score
            results.append(info)

        return results
