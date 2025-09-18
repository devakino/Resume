# # Using pydantic

# import os
# import pickle
# import faiss
# import numpy as np
# import json
# import re
# from sklearn.preprocessing import normalize
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import google.generativeai as genai
# from pydantic import BaseModel, Field
# from typing import Optional, Union

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Load sentence transformer model
# model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# CACHE_DIR = os.path.join(BASE_DIR, "cache")
# os.makedirs(CACHE_DIR, exist_ok=True)
# FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_resume_index.index")
# METADATA_PATH = os.path.join(CACHE_DIR, "faiss_resume_metadata.pkl")


# # Pydantic models

# class Metadata(BaseModel):
#     name: Optional[str] = Field(default="N/A", description="Name of the person")
#     filepath: str = Field(description="Link of the Resume")
#     id: int = Field(description="ID of the person")

# class Content(BaseModel):
#     expertise: str = Field(description="Expertise of the person")
#     experience: float = Field(description="Years of experience of the person")
#     education: str = Field(description="Education of the person")
#     skills: str = Field(description="Skills of the person")

# class Summary(BaseModel):
#     metadata: Metadata
#     content: Content

# class JD(BaseModel):
#     """Summary of a Job Description"""
#     expertise: str = Field(description="")
#     experience: int = Field(description="")
#     skills: str = Field(description="")


# def ensure_string_field(field: Optional[Union[str, list]]) -> str:
#     """
#     Helper to convert a field that can be string or list to a clean string.
#     """
#     if isinstance(field, list):
#         # Join list items with commas, stripping whitespace
#         return ", ".join(str(i).strip() for i in field)
#     if field is None:
#         return ""
#     return str(field).strip()


# def build_or_load_faiss_index(resume_texts, filepaths=None):
#     """
#     Build or load FAISS index for the resumes.
#     Args:
#         resume_texts: list of resume text strings
#         filepaths: list of original file paths corresponding to resumes
#     Returns:
#         index: FAISS index
#         metadata_store: list of metadata dicts for each resume (id, filepath, content)
#     """
#     if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
#         print("✅ Loading FAISS index and metadata from disk...")
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         with open(METADATA_PATH, "rb") as f:
#             metadata_store = pickle.load(f)
#         return index, metadata_store

#     print("🛠️ Building new FAISS index...")
#     metadata_store = []
#     embeddings = []

#     for idx, resume_text in enumerate(resume_texts):
#         emb = model.encode([resume_text], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
#         emb = normalize(emb)
#         embeddings.append(emb)

#         metadata_store.append({
#             "id": idx,
#             "filepath": filepaths[idx] if filepaths else f"resume_{idx}.txt",
#             "content": resume_text
#         })

#     embeddings_matrix = np.vstack(embeddings).astype("float32")

#     index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
#     index.add(embeddings_matrix)

#     # Save index and metadata
#     faiss.write_index(index, FAISS_INDEX_PATH)
#     with open(METADATA_PATH, "wb") as f:
#         pickle.dump(metadata_store, f)

#     print("✅ FAISS index and metadata saved locally.")
#     return index, metadata_store


# def parse_resume_to_json(resume_text, meta, model_gemini):
#     """
#     Use Gemini LLM to parse resume text into structured JSON.
#     """
#     prompt = f"""
# You are an expert resume parser. Convert the following resume into a structured JSON object with these fields:
# - metadata: {{ name (if available), id, filepath }}
# - content: {{ expertise, experience (in years as float), education, skills }}

# Return only the JSON. Do not include any explanation or markdown formatting.

# Resume:
# \"\"\"
# {resume_text[:3000]}
# \"\"\"
# """
#     try:
#         response = model_gemini.generate_content(prompt)
#         raw_text = response.text.strip()

#         json_match = re.search(r'{.*}', raw_text, re.DOTALL)
#         if not json_match:
#             raise ValueError("No valid JSON found in model response.")

#         json_str = json_match.group(0)
#         parsed = json.loads(json_str)
#         parsed['metadata']['filepath'] = meta['filepath']
#         parsed['metadata']['id'] = meta['id']
#         return parsed

#     except Exception as e:
#         print(f"⚠️ Failed to parse resume ID {meta['id']}: {e}")
#         return None


# def parse_job_description_to_json(job_description, model_gemini):
#     """
#     Use Gemini LLM to parse job description into structured JSON.
#     """
#     prompt = f"""
# You are a job description parser. Convert the following job description into a structured JSON object with these fields:
# - expertise
# - experience (in years as a number)
# - skills

# Return only the JSON. Do not include any explanation or markdown formatting.

# Job Description:
# \"\"\"
# {job_description}
# \"\"\"
# """
#     try:
#         response = model_gemini.generate_content(prompt)
#         raw_text = response.text.strip()

#         json_match = re.search(r'{.*}', raw_text, re.DOTALL)
#         if not json_match:
#             raise ValueError("No valid JSON found in job description response.")

#         json_str = json_match.group(0)
#         return json.loads(json_str)
#     except Exception as e:
#         print(f"⚠️ Failed to parse job description: {e}")
#         return None


# def find_top_k_matches(index, metadata_store, job_description, top_k=5):
#     """
#     Given the FAISS index and job description, find the top-k matching resumes,
#     parse them with Gemini, and display results with their original file paths.
#     """
#     model_gemini = genai.GenerativeModel("gemini-2.5-flash")

#     # Embed job description and search
#     jd_embedding = model.encode([job_description], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
#     jd_embedding = normalize(jd_embedding).astype("float32")
#     similarities, indices = index.search(jd_embedding, top_k)

#     top_k_resumes = []
#     for rank, (idx, score) in enumerate(zip(indices[0], similarities[0])):
#         meta = metadata_store[idx]
#         top_k_resumes.append(meta)

#     # Parse resumes with Gemini LLM
#     resume_json_list = []
#     for meta in top_k_resumes:
#         parsed = parse_resume_to_json(meta['content'], meta, model_gemini)
#         if parsed:
#             # Clean list fields to strings for Pydantic validation
#             parsed['content']['expertise'] = ensure_string_field(parsed['content'].get('expertise'))
#             parsed['content']['education'] = ensure_string_field(parsed['content'].get('education'))
#             parsed['content']['skills'] = ensure_string_field(parsed['content'].get('skills'))
#             resume_json_list.append(parsed)

#     if not resume_json_list:
#         print("❌ No resumes were successfully parsed. Exiting.")
#         return None

#     # Parse job description with Gemini
#     jd_json = parse_job_description_to_json(job_description, model_gemini)
#     if not jd_json:
#         print("❌ Job description parsing failed.")
#         return None

#     # Clean job description fields to strings
#     jd_json['expertise'] = ensure_string_field(jd_json.get('expertise'))
#     jd_json['skills'] = ensure_string_field(jd_json.get('skills'))

#     # Use Pydantic to validate and serialize
#     try:
#         resume_models = [Summary.parse_obj(r) for r in resume_json_list]
#         jd_model = JD.parse_obj(jd_json)
#     except Exception as e:
#         print(f"⚠️ Pydantic validation failed: {e}")
#         return None

#     # Prepare JSON strings for prompt
#     resume_json = json.dumps([r.model_dump() for r in resume_models], indent=2)
#     jd_json_str = json.dumps(jd_model.model_dump(), indent=2)

#     system_prompt = """You are an expert resume evaluator.
# You are given a job description and a list of candidate resumes in JSON format.
# Your task is to compare each resume against the job description and return ONLY the full JSON of the one best matching resume.

# You must consider:
# - Expertise match (job role vs resume role)
# - Minimum experience (must meet or exceed the required years)
# - Skill overlap (more shared skills = better)

# Return only the best matching resume JSON. Do not include any explanation or extra output.
# """

#     human_prompt = f"""Job Description JSON:
# {jd_json_str}

# Resume List JSON:
# {resume_json}
# """

#     try:
#         full_prompt = system_prompt + "\n\n" + human_prompt
#         response = model_gemini.generate_content(full_prompt)
#         raw_text = response.text.strip()

#         json_match = re.search(r'{.*}', raw_text, re.DOTALL)
#         if not json_match:
#             raise ValueError("No valid JSON found in evaluation response.")

#         json_str = json_match.group(0)
#         best_resume_json = json.loads(json_str)

#         # print(f"\n🔥 Best matched resume file: {best_resume_json['metadata']['filepath']}")
#         return best_resume_json

#     except Exception as e:
#         print(f"⚠️ Failed to parse LLM response JSON: {e}")
#         return None
















# # Using TypedDict

import os
import pickle
import faiss
import numpy as np
import json
import re
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Union, List, TypedDict

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load sentence transformer model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_resume_index.index")
METADATA_PATH = os.path.join(CACHE_DIR, "faiss_resume_metadata.pkl")

# TypedDict models

class Metadata(TypedDict, total=False):
    name: Optional[str]
    filepath: str
    id: int

class Content(TypedDict):
    expertise: str
    experience: float
    education: str
    skills: str

class Summary(TypedDict):
    metadata: Metadata
    content: Content

class JD(TypedDict):
    expertise: str
    experience: int
    skills: str


def ensure_string_field(field: Optional[Union[str, list]]) -> str:
    """
    Helper to convert a field that can be string or list to a clean string.
    """
    if isinstance(field, list):
        # Join list items with commas, stripping whitespace
        return ", ".join(str(i).strip() for i in field)
    if field is None:
        return ""
    return str(field).strip()


def build_or_load_faiss_index(resume_texts, filepaths=None):
    """
    Build or load FAISS index for the resumes.
    Args:
        resume_texts: list of resume text strings
        filepaths: list of original file paths corresponding to resumes
    Returns:
        index: FAISS index
        metadata_store: list of metadata dicts for each resume (id, filepath, content)
    """
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


def parse_resume_to_json(resume_text, meta, model_gemini):
    """
    Use Gemini LLM to parse resume text into structured JSON.
    """
    prompt = f"""
You are an expert resume parser. Convert the following resume into a structured JSON object with these fields:
- metadata: {{ name (if available), id, filepath }}
- content: {{ expertise, experience (in years as float), education, skills }}

Return only the JSON. Do not include any explanation or markdown formatting.

Resume:
\"\"\"
{resume_text[:3000]}
\"\"\"
"""
    try:
        response = model_gemini.generate_content(prompt)
        raw_text = response.text.strip()

        json_match = re.search(r'{.*}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in model response.")

        json_str = json_match.group(0)
        parsed: Summary = json.loads(json_str)
        parsed['metadata']['filepath'] = meta['filepath']
        parsed['metadata']['id'] = meta['id']
        return parsed

    except Exception as e:
        print(f"⚠️ Failed to parse resume ID {meta['id']}: {e}")
        return None


def parse_job_description_to_json(job_description, model_gemini):
    """
    Use Gemini LLM to parse job description into structured JSON.
    """
    prompt = f"""
You are a job description parser. Convert the following job description into a structured JSON object with these fields:
- expertise
- experience (in years as a number)
- skills

Return only the JSON. Do not include any explanation or markdown formatting.

Job Description:
\"\"\"
{job_description}
\"\"\"
"""
    try:
        response = model_gemini.generate_content(prompt)
        raw_text = response.text.strip()

        json_match = re.search(r'{.*}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in job description response.")

        json_str = json_match.group(0)
        jd_parsed: JD = json.loads(json_str)
        return jd_parsed
    except Exception as e:
        print(f"⚠️ Failed to parse job description: {e}")
        return None


def find_top_k_matches(index, metadata_store, job_description, top_k=5):
    """
    Given the FAISS index and job description, find the top-k matching resumes,
    parse them with Gemini, and display results with their original file paths.
    """
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")

    # Embed job description and search
    jd_embedding = model.encode([job_description], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    jd_embedding = normalize(jd_embedding).astype("float32")
    similarities, indices = index.search(jd_embedding, top_k)

    top_k_resumes = []
    for rank, (idx, score) in enumerate(zip(indices[0], similarities[0])):
        meta = metadata_store[idx]
        top_k_resumes.append(meta)

    # Parse resumes with Gemini LLM
    resume_json_list: List[Summary] = []
    for meta in top_k_resumes:
        parsed = parse_resume_to_json(meta['content'], meta, model_gemini)
        if parsed:
            # Clean list fields to strings for consistency
            parsed['content']['expertise'] = ensure_string_field(parsed['content'].get('expertise'))
            parsed['content']['education'] = ensure_string_field(parsed['content'].get('education'))
            parsed['content']['skills'] = ensure_string_field(parsed['content'].get('skills'))
            resume_json_list.append(parsed)

    if not resume_json_list:
        print("❌ No resumes were successfully parsed. Exiting.")
        return None

    # Parse job description with Gemini
    jd_json = parse_job_description_to_json(job_description, model_gemini)
    if not jd_json:
        print("❌ Job description parsing failed.")
        return None

    # Clean job description fields to strings
    jd_json['expertise'] = ensure_string_field(jd_json.get('expertise'))
    jd_json['skills'] = ensure_string_field(jd_json.get('skills'))

    # Serialize TypedDicts as JSON strings
    resume_json_str = json.dumps(resume_json_list, indent=2)
    jd_json_str = json.dumps(jd_json, indent=2)

    system_prompt = """You are an expert resume evaluator.
You are given a job description and a list of candidate resumes in JSON format.
Your task is to compare each resume against the job description and return ONLY the full JSON of the one best matching resume.

You must consider:
- Expertise match (job role vs resume role)
- Minimum experience (must meet or exceed the required years)
- Skill overlap (more shared skills = better)

Return only the best matching resume JSON. Do not include any explanation or extra output.
"""

    human_prompt = f"""Job Description JSON:
{jd_json_str}

Resume List JSON:
{resume_json_str}
"""

    try:
        full_prompt = system_prompt + "\n\n" + human_prompt
        response = model_gemini.generate_content(full_prompt)
        raw_text = response.text.strip()

        json_match = re.search(r'{.*}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in evaluation response.")

        json_str = json_match.group(0)
        best_resume_json = json.loads(json_str)

        return best_resume_json

    except Exception as e:
        print(f"⚠️ Failed to parse LLM response JSON: {e}")
        return None











# # Using LLM

# import os
# import pickle
# import faiss
# import numpy as np
# import json
# import re
# from sklearn.preprocessing import normalize
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load env and configure Gemini
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# # Load embedding model
# embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# CACHE_DIR = os.path.join(BASE_DIR, "cache")
# os.makedirs(CACHE_DIR, exist_ok=True)
# FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_resume_index.index")
# METADATA_PATH = os.path.join(CACHE_DIR, "faiss_resume_metadata.pkl")


# def ensure_string(field):
#     if isinstance(field, list):
#         return ", ".join(str(i).strip() for i in field)
#     return str(field).strip() if field else ""


# def build_or_load_faiss_index(resume_texts, filepaths=None):
#     if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
#         print("✅ Loaded FAISS index and metadata.")
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         with open(METADATA_PATH, "rb") as f:
#             metadata_store = pickle.load(f)
#         return index, metadata_store

#     print("🛠️ Building FAISS index...")
#     metadata_store, embeddings = [], []

#     for idx, text in enumerate(resume_texts):
#         emb = embed_model.encode([text], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
#         emb = normalize(emb)
#         embeddings.append(emb)
#         metadata_store.append({
#             "id": idx,
#             "filepath": filepaths[idx] if filepaths else f"resume_{idx}.txt",
#             "content": text
#         })

#     matrix = np.vstack(embeddings).astype("float32")
#     index = faiss.IndexFlatIP(matrix.shape[1])
#     index.add(matrix)

#     faiss.write_index(index, FAISS_INDEX_PATH)
#     with open(METADATA_PATH, "wb") as f:
#         pickle.dump(metadata_store, f)

#     return index, metadata_store


# def parse_with_gemini(prompt):
#     try:
#         response = model_gemini.generate_content(prompt)
#         raw = response.text.strip()
#         match = re.search(r'{.*}', raw, re.DOTALL)
#         return json.loads(match.group(0)) if match else None
#     except Exception as e:
#         print(f"⚠️ LLM parsing failed: {e}")
#         return None


# def parse_resume(resume_text, meta):
#     prompt = f"""
# You are an expert resume parser. Convert the following resume into structured JSON:
# - metadata: {{ name (if available), id, filepath }}
# - content: {{ expertise, experience (in years), education, skills }}

# Return only JSON.

# Resume:
# \"\"\"
# {resume_text[:3000]}
# \"\"\"
# """
#     parsed = parse_with_gemini(prompt)
#     if parsed:
#         parsed.setdefault("metadata", {})
#         parsed["metadata"]["id"] = meta["id"]
#         parsed["metadata"]["filepath"] = meta["filepath"]
#     return parsed


# def parse_job_description(jd_text):
#     prompt = f"""
# You are a job description parser. Convert the following into JSON:
# - expertise
# - experience (years)
# - skills

# Return only JSON.

# JD:
# \"\"\"
# {jd_text}
# \"\"\"
# """
#     return parse_with_gemini(prompt)


# def find_top_k_matches(index, metadata_store, job_description, top_k=5):
#     jd_emb = embed_model.encode([job_description], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
#     jd_emb = normalize(jd_emb).astype("float32")
#     similarities, indices = index.search(jd_emb, top_k)

#     top_k_resumes = [metadata_store[i] for i in indices[0]]
#     parsed_resumes = []

#     for meta in top_k_resumes:
#         parsed = parse_resume(meta["content"], meta)
#         if parsed:
#             parsed["content"]["expertise"] = ensure_string(parsed["content"].get("expertise"))
#             parsed["content"]["education"] = ensure_string(parsed["content"].get("education"))
#             parsed["content"]["skills"] = ensure_string(parsed["content"].get("skills"))
#             parsed_resumes.append(parsed)

#     if not parsed_resumes:
#         print("❌ No resumes parsed.")
#         return None

#     jd_json = parse_job_description(job_description)
#     if not jd_json:
#         print("❌ JD parsing failed.")
#         return None

#     jd_json["expertise"] = ensure_string(jd_json.get("expertise"))
#     jd_json["skills"] = ensure_string(jd_json.get("skills"))

#     resume_json_str = json.dumps(parsed_resumes, indent=2)
#     jd_json_str = json.dumps(jd_json, indent=2)

#     eval_prompt = f"""
# You are an expert resume evaluator.
# Compare the job description and candidate resumes below.
# Return ONLY the best matching resume JSON.

# Criteria:
# - Expertise match
# - Experience (≥ required)
# - Skill overlap

# Job Description JSON:
# {jd_json_str}

# Resume List JSON:
# {resume_json_str}
# """
#     return parse_with_gemini(eval_prompt)
