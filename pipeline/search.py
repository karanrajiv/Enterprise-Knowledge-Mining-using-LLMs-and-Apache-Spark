import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def load_index(index_path):
    return faiss.read_index(index_path)

def semantic_search(query, index, model_name='all-MiniLM-L6-v2', top_k=5):
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return distances, indices

def load_chunks(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
