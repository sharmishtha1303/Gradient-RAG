import os
import pickle
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
input_txt = "data_clean/ai_ml/ai_ml_clean.txt"  # ✅ Cleaned text file
output_folder = "embeddings"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# LOAD & CLEAN TEXT
# -----------------------------
if not os.path.exists(input_txt):
    raise FileNotFoundError(f"❌ File not found: {input_txt}")

with open(input_txt, "r", encoding="utf-8") as f:
    text = f.read()

# ✅ NEW: Remove unwanted numbers/junk before embedding
def clean_text(text):
    text = re.sub(r'\b\d{2,6}\b', '', text)     # remove long numbers (IDs, indexes)
    text = re.sub(r'Page \d+', '', text)        # remove page numbers
    text = re.sub(r'[\n\s]+', ' ', text)        # extra spaces/newlines
    text = re.sub(r'[^a-zA-Z0-9.,;:?!()\[\] ]', '', text)  # junk characters
    return text.strip()

text = clean_text(text)

# -----------------------------
# STEP 1: Chunking
# -----------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits text into smaller chunks for better retrieval performance.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 50:  # ignore very small chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(text)
print(f"✅ Total chunks created: {len(chunks)}")

# -----------------------------
# STEP 2: Generate Embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = model.encode(chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=True)

# Normalize embeddings for better FAISS performance
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# STEP 3: Create FAISS Index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (faster + normalized vectors)
index.add(embeddings)

# -----------------------------
# STEP 4: Save Index & Chunks
# -----------------------------
faiss.write_index(index, os.path.join(output_folder, "ai_ml_faiss.index"))
with open(os.path.join(output_folder, "ai_ml_chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("✅ Data pipeline completed: Clean chunks and FAISS index saved successfully!")