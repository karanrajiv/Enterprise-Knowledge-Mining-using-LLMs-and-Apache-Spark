import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def query_faiss_index(index, query_embedding):
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    return I
