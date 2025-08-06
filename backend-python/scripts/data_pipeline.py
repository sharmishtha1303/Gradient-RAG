import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
input_files = [
    "data_clean/AI_DL_Data.txt",
    "data_clean/ML_Data.txt",
    "data_clean/ai_ml_clean.txt"
]  # ✅ Multiple input files
output_folder = "embeddings"
os.makedirs(output_folder, exist_ok=True)

index_path = os.path.join(output_folder, "combined_faiss.index")
metadata_path = os.path.join(output_folder, "combined_metadata.pkl")

# -----------------------------
# 1️⃣ Chunking Function
# -----------------------------
def chunk_text(text, chunk_size=250, overlap=50):
    """
    Splits text into smaller overlapping chunks for better RAG retrieval.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 30:  # Ignore very small chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# 2️⃣ Collect chunks + metadata
# -----------------------------
all_chunks = []
metadata = []

for file_path in input_files:
    file_name = os.path.basename(file_path)
    
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}, skipping...")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print(f"⚠️ File is empty: {file_path}, skipping...")
        continue

    chunks = chunk_text(text)
    print(f"✅ Chunks created from {file_name}: {len(chunks)}")

    for chunk in chunks:
        all_chunks.append(chunk)
        metadata.append({"text": chunk, "source": file_name})

if not all_chunks:
    raise ValueError("❌ No valid text found in input files!")

print(f"📦 Total chunks from all files: {len(all_chunks)}")

# -----------------------------
# 3️⃣ Generate Embeddings
# -----------------------------
print("🔹 Generating embeddings, please wait...")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = model.encode(all_chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=True)

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# 4️⃣ Create & Save FAISS Index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, index_path)

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print("\n✅ Embedding pipeline completed successfully!")
print(f"📂 Saved combined files in: {output_folder}")
print("   - combined_faiss.index")
print("   - combined_metadata.pkl")
