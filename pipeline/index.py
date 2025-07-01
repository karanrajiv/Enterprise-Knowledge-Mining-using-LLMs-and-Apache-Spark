import faiss
import numpy as np
import pickle

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, file_path):
    faiss.write_index(index, file_path)

def save_metadata(chunks, embeddings, doc_file, emb_file):
    with open(doc_file, 'wb') as f:
        pickle.dump(chunks, f)
    with open(emb_file, 'wb') as f:
        pickle.dump(embeddings, f)
