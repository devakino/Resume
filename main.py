from extractors.extractor import extract_resume_texts
from embeddings.faiss_indexer import build_or_load_faiss_index, find_top_k_matches

# Get resumes
resume_texts = extract_resume_texts()
filepaths = [f"resume_{i}.txt" for i in range(len(resume_texts))]

# Example Job Description
job_description = """
At [Company X], we’re building human-focused technology solutions that improve efficiency and performance. We’re seeking a front-end developer to translate user-friendly designs into crisp code and thoughtful experiences for our customers. This person will work alongside our web team to determine customer needs, brainstorm solutions, generate website mockups and prototypes, present work to customers, and develop and optimize live platforms. We’re looking for someone who is quick, technically strong, and not afraid to ask questions. An understanding of not just how to code but why certain code works is essential.
"""

# Build FAISS index and search
index, metadata_store = build_or_load_faiss_index(resume_texts, filepaths)
find_top_k_matches(index, metadata_store, job_description, top_k=5)
