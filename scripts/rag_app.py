import streamlit as st
import faiss
import pickle
import numpy as np
import requests
import time
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
API_KEY = "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"  # Replace with your OpenRouter key
MODEL = "deepseek/deepseek-r1:free"
MAX_CONTEXT_LEN = 900   # Reduce token size for faster responses
TOP_K_RESULTS = 2       # Retrieve only top 2 chunks for speed

# -----------------------------
# CACHED LOADING FUNCTIONS
# -----------------------------
@st.cache_resource
def load_model():
    """Load SentenceTransformer model only once to save time."""
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

@st.cache_resource
def load_faiss_index():
    """Load FAISS index and chunks only once."""
    index = faiss.read_index("embeddings/ai_ml_faiss.index")
    with open("embeddings/ai_ml_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Load resources
model = load_model()
index, chunks = load_faiss_index()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Gradient RAG - AI/ML Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Gradient RAG – AI/ML Knowledge Assistant")

query = st.text_input("🔎 Ask your AI/ML question:")

if query:
    start_time = time.time()

    # -----------------------------
    # STEP 1 – Semantic Search
    # -----------------------------
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, results = index.search(np.array(query_vec), k=TOP_K_RESULTS)
    retrieved_chunks = [chunks[i] for i in results[0]]

    # Limit context length safely
    retrieved_text = "\n".join(retrieved_chunks)
    if len(retrieved_text) > MAX_CONTEXT_LEN:
        retrieved_text = retrieved_text[:MAX_CONTEXT_LEN]

    # -----------------------------
    # STEP 2 – Call DeepSeek R1 via OpenRouter
    # -----------------------------
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert assistant. Provide a clear, short, and direct answer based ONLY on the provided context. Avoid repeating the question."},
            {"role": "user", "content": f"Context:\n{retrieved_text}\n\nQuestion: {query}"}
        ],
        "max_tokens": 250,
        "temperature": 0.3
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=payload, timeout=30)

        # Handle different response scenarios
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                answer = data["choices"][0]["message"]["content"].strip()
            else:
                answer = "⚠️ The AI model did not return a valid response. Try rephrasing your question."
        else:
            answer = f"⚠️ API Error: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        answer = "⏳ API request timed out. Please try again."
    except Exception as e:
        answer = f"⚠️ Unexpected error: {e}"

    # Post-process: remove repeated question if present
    if answer and answer.lower().startswith(query.lower()):
        answer = answer[len(query):].strip(" :.-")

    # -----------------------------
    # STEP 3 – Display Results
    # -----------------------------
    st.subheader("💡 AI Answer:")
    st.write(answer if answer.strip() != "" else "⚠️ No answer returned. Try again.")

    with st.expander("📄 View Retrieved Context"):
        st.write(retrieved_text)

    st.write(f"⏱️ Response Time: **{time.time() - start_time:.2f} seconds**")
