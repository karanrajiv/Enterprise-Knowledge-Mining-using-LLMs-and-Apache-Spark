import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index and metadata
@st.cache_resource
def load_index(index_path="embeddings/faiss_index.index"):
    return faiss.read_index(index_path)

@st.cache_data
def load_chunks(chunk_path="embeddings/doc_chunks.pkl"):
    with open(chunk_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Semantic Search Function
def semantic_search(query, index, chunks, model, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype("float32"), top_k)
    results = [(chunks[i], distances[0][rank]) for rank, i in enumerate(indices[0])]
    return results

# Load resources
index = load_index()
chunks = load_chunks()
model = load_model()

# Streamlit UI
st.title("🔍 Enterprise Knowledge Semantic Search")
st.markdown("Ask a question and get the most relevant knowledge from internal documents.")

query = st.text_input("Enter your query:", "")

if query:
    results = semantic_search(query, index, chunks, model)
    st.markdown("### Top Results")
    for idx, (chunk, score) in enumerate(results):
        st.markdown(f"**Result {idx+1}** (Score: {score:.2f})")
        st.code(chunk, language='markdown')
