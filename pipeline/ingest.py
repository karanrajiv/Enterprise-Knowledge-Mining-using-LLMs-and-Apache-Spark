import os
from PyPDF2 import PdfReader

def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                docs.append(f.read())
        elif filename.endswith(".pdf"):
            with open(os.path.join(directory, filename), 'rb') as f:
                reader = PdfReader(f)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
                docs.append(text)
    return docs

def chunk_document(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
