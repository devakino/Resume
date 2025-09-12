import os
import pickle
import fitz  # PyMuPDF

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "resume_texts.pkl")

def extract_resume_texts(force_reextract=False):
    if os.path.exists(CACHE_PATH) and not force_reextract:
        print("✅ Loaded extracted resume text from cache.")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("🛠️ Extracting text from PDFs...")
    resume_texts = []

    all_files = os.listdir(DATA_DIR)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        try:
            with fitz.open(pdf_path) as doc:
                full_text = ""
                for page in doc:
                    page_text = page.get_text().strip()
                    if page_text:
                        full_text += page_text + "\n"

            if full_text.strip():
                resume_texts.append(full_text.strip())
                print(f"✅ Extracted: {pdf_file}")
            else:
                print(f"❌ Skipped (no text): {pdf_file}")
        except Exception as e:
            print(f"⚠️ Error reading {pdf_file}: {e}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(resume_texts, f)

    print(f"\n📦 Saved extracted resume texts to cache ({CACHE_PATH})")
    return resume_texts
