# For pydantic and TypeDict and LLM 

from extractors.extractor import extract_resume_texts
from embeddings.faiss_indexer import build_or_load_faiss_index, find_top_k_matches
import time

if __name__ == "__main__":
    # Extract resumes (text + file paths)
    resume_texts, filepaths = extract_resume_texts()

    # Example job description (can be replaced with any input)
    job_description = """
    We are seeking a highly skilled and experienced Web Developer with over 4 years of hands-on experience specializing in React.js to join our dynamic team. The ideal candidate should have a strong command of modern front-end technologies including HTML5, CSS3, JavaScript (ES6+), Redux, and TypeScript, with a proven track record of building scalable, high-performance web applications. Proficiency in integrating RESTful APIs, version control systems like Git, and familiarity with Agile/Scrum development methodologies are essential. Experience with testing frameworks such as Jest or React Testing Library, as well as knowledge of CI/CD pipelines, performance optimization, and cross-browser compatibility, will be a significant advantage. Strong problem-solving abilities, attention to detail, and a collaborative mindset are crucial for success in this role.
    """

    # Build/load FAISS index
    index, metadata_store = build_or_load_faiss_index(resume_texts, filepaths)

    # Caluculating time of execution
    start_time = time.time()    
    
    # Find best matching resume filepath using updated search (with LLM-based reranking)
    best_resume_json_or_none = find_top_k_matches(index, metadata_store, job_description, top_k=5)

    if best_resume_json_or_none:
        best_filepath = best_resume_json_or_none['metadata']['filepath']
        print(f"\n🔥 Best matched resume file path: {best_filepath}")
    else:
        print("No suitable matching resume found.")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n⏱️ Execution Time: {execution_time:.2f} seconds")