# For pydantic and TypeDict and LLM 

from extractors.extractor import extract_resume_texts
from embeddings.faiss_indexer import build_or_load_faiss_index, find_top_k_matches
import time

if __name__ == "__main__":
    # Extract resumes (text + file paths)
    resume_texts, filepaths = extract_resume_texts()

    # Example job description (can be replaced with any input)
    job_description = """
    We are looking for an experienced AI Engineer with 3+ years in designing, developing, and deploying machine learning solutions. The ideal candidate has strong proficiency in Python and ML frameworks like TensorFlow, PyTorch, or Scikit-learn, along with hands-on experience in data preprocessing, model training, and deploying models to production using tools such as Docker, Kubernetes, and cloud platforms (AWS, GCP, or Azure). You should have a solid understanding of machine learning algorithms, deep learning architectures (e.g., CNNs, RNNs, transformers), and experience working with large datasets using tools like Pandas, NumPy, and SQL. Familiarity with MLOps practices, version control systems, and collaborative development tools is essential. Experience with NLP, computer vision, or generative AI is a plus. You will work closely with cross-functional teams to deliver scalable, data-driven solutions while continuously improving model performance and system reliability.
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