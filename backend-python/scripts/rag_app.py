import streamlit as st
import faiss
import pickle
import numpy as np
import time
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"  # Groq model
MAX_CONTEXT_LEN = 1000     # Limit context to avoid overload
TOP_K_RESULTS = 2          # Retrieve only the most relevant chunk

# Initialize Groq client
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

# -----------------------------
# CACHED LOADING FUNCTIONS
# -----------------------------
@st.cache_resource
def load_model():
    """Load SentenceTransformer model only once."""
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

@st.cache_resource
def load_faiss_index():
    """Load FAISS index and metadata only once."""
    index = faiss.read_index("embeddings/combined_faiss.index")
    with open("embeddings/combined_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Load resources
model = load_model()
index, metadata = load_faiss_index()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Gradient RAG - AI/ML Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Gradient RAG – AI/ML Knowledge Assistant")

query = st.text_input("🔎 Ask your AI/ML question:")

if query:
    start_time = time.time()

    # STEP 1 – Semantic Search
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, results = index.search(np.array(query_vec), k=TOP_K_RESULTS)

    # Collect retrieved chunks with sources
    retrieved_chunks = []
    retrieved_sources = []
    for idx in results[0]:
        retrieved_chunks.append(metadata[idx]['text'])
        retrieved_sources.append(metadata[idx]['source'])

    # Limit context length safely
    retrieved_text = "\n".join(retrieved_chunks)
    if len(retrieved_text) > MAX_CONTEXT_LEN:
        retrieved_text = retrieved_text[:MAX_CONTEXT_LEN]

    # STEP 2 – Call Groq API with improved summarization prompt
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI/ML assistant. Answer the question based ONLY on the provided context. "
                        "If the answer is not directly available, summarize the most relevant information from the context "
                        "instead of saying 'I don't know'. Avoid hallucinating or adding external information. "
                        "Keep the response concise and clear (2-3 sentences maximum)."
                    )
                },
                {"role": "user", "content": f"Context:\n{retrieved_text}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"⚠️ Error fetching response from Groq: {e}"

    # Post-process: remove repeated question if present
    if answer and answer.lower().startswith(query.lower()):
        answer = answer[len(query):].strip(" :.-")

    # STEP 3 – Display Results
    st.subheader("💡 AI Answer:")
    st.write(answer if answer.strip() != "" else "⚠️ No answer returned. Try again.")

    with st.expander("📄 View Retrieved Context and Sources"):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Source:** {retrieved_sources[i]}")
            st.write(chunk)
            st.markdown("---")

    st.write(f"⏱️ Response Time: **{time.time() - start_time:.2f} seconds**")