import os
import pickle
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# Load Nomic embedding model
# ---------------------------
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_resume_index.index")
METADATA_PATH = os.path.join(CACHE_DIR, "faiss_resume_metadata.pkl")

# ---------------------------
# FAISS Indexing
# ---------------------------
def build_or_load_faiss_index(resume_texts, filepaths=None):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("✅ Loading FAISS index and metadata from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        return index, metadata_store

    print("🛠️ Building new FAISS index...")
    metadata_store = []
    embeddings = []

    for idx, resume_text in enumerate(resume_texts):
        emb = model.encode([resume_text], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        emb = normalize(emb)
        embeddings.append(emb)

        metadata_store.append({
            "id": idx,
            "filepath": filepaths[idx] if filepaths else f"resume_{idx}.txt",
            "content": resume_text
        })

    embeddings_matrix = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    # Save index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print("✅ FAISS index and metadata saved locally.")
    return index, metadata_store

# ---------------------------
# Gemini Resume Summarization
# ---------------------------
def summarize_resumes_with_gemini(resume_matches):
    model = genai.GenerativeModel("gemini-1.5-flash")

    print("\n📝 Resume Summaries using Gemini:\n")
    for rank, meta in enumerate(resume_matches, start=1):
        prompt = f"""
You are a helpful assistant. Summarize the following resume for a recruiter. Keep the summary clear, concise, and job-focused.

Resume Text:
\"\"\"
{meta['content']}
\"\"\"

Summary:
"""
        try:
            response = model.generate_content(prompt)
            summary = response.text.strip()
            print(f"🔹 Summary for Resume Rank {rank} (File: {meta['filepath']}):\n{summary}\n")
        except Exception as e:
            print(f"⚠️ Gemini API error for resume {rank}: {e}\n")

# ---------------------------
# Top-K Resume Matching
# ---------------------------
def find_top_k_matches(index, metadata_store, job_description, top_k=5):
    jd_embedding = model.encode([job_description], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    jd_embedding = normalize(jd_embedding).astype("float32")

    similarities, indices = index.search(jd_embedding, top_k)

    print(f"\n🔍 Top {top_k} Matching Resumes:\n")
    top_k_resumes = []

    for rank, (idx, score) in enumerate(zip(indices[0], similarities[0])):
        meta = metadata_store[idx]
        print(f"Rank {rank+1}")
        print(f"ID: {meta['id']}")
        print(f"Filepath: {meta['filepath']}")
        print(f"Match: {round(score * 100, 2)}%")
        print(f"Content Preview:\n{meta['content'][:1000]}\n...\n")
        top_k_resumes.append(meta)

    # ✨ Summarize with Gemini
    summarize_resumes_with_gemini(top_k_resumes)
