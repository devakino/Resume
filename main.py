from extractors.extractor import extract_resume_texts
from embeddings.faiss_indexer import build_or_load_faiss_index, find_top_k_matches

# Get resumes
resume_texts = extract_resume_texts()
filepaths = [f"resume_{i}.txt" for i in range(len(resume_texts))]

# Example Job Description
job_description = """
Robust and responsive web applications instill a sense of trust and reliability toward a brand in today’s dynamic and ever-evolving cyberspace. We are on the lookout for a top-class AngularJS developer to build streamlined applications that seamlessly meet the users’ needs. Our ideal AngularJS developer must first and foremost be a team player who has a compelling and deep-rooted fascination with coding. With the ability to see the big picture, the AngularJS developer should be a natural at taking complex technical decisions for both frontend and backend JavaScript applications. If you are a problem-solver with proficient knowledge in HTML, JavaScipt, and CSS and have the thirst for creating powerful and prominent web applications, then you might be the right fit for us.
"""


# Build FAISS index and search
index, metadata_store = build_or_load_faiss_index(resume_texts, filepaths)
find_top_k_matches(index, metadata_store, job_description, top_k=5)

