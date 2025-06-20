from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(doc_df):
    embeddings = []
    metadata = []
    for row in doc_df.collect():
        inputs = tokenizer(row['value'], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embed = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embed)
            metadata.append(row['filename'])
    return embeddings, metadata
