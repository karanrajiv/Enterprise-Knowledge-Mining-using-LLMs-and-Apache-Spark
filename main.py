from spark_preprocessing import preprocess_documents
from embedding_generation import generate_embeddings
from faiss_indexing import build_faiss_index, query_faiss_index

def run_pipeline(doc_path):
    preprocessed = preprocess_documents(doc_path)
    embeddings, metadata = generate_embeddings(preprocessed)
    index = build_faiss_index(embeddings)
    return index, metadata

if __name__ == "__main__":
    index, meta = run_pipeline("data/sample_docs")
    results = query_faiss_index(index, "What is our latest financial performance?")
    print(results)
