import os
import pickle
import time
from typing import List
import requests
import numpy as np
import faiss
import json
from app.pdf.extract import parse_pdf
import logging

logger = logging.getLogger(__name__)

# Local Ollama endpoints and models
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_HOST}/api/embeddings"
OLLAMA_GEN_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
EMBED_MODEL_NAME = "nomic-embed-text"   # ensure you pulled this: ollama pull nomic-embed-text
LLM_MODEL_NAME = "llama3.2"              # ensure you pulled this: ollama pull llama3.2

EMBEDDINGS_DIR = os.path.join(os.getcwd(), "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# --- Helpers: chunking ----------------------------------------------------
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """
    Chunk text by characters (naive) but with overlap.
    We use characters because tokenizers differ per model; this is simple and safe for MVP.
    """
    print("Chunking text...")
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        # advance: keep overlap
        start = max(end - overlap, end)
    return chunks

# --- Ollama embedding helper ----------------------------------------------
def embed_with_ollama(texts: List[str]) -> np.ndarray:
    """
    Query Ollama embeddings endpoint with a list of texts and return array shape (n, dim) float32.
    Uses the /api/embeddings endpoint with payload: {"model": EMBED_MODEL_NAME, "input": texts}
    """
    print("Embedding text with Ollama...")
    if not isinstance(texts, list):
        texts = [texts]

    payload = {"model": EMBED_MODEL_NAME, "input": texts}
    try:
        r = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama embeddings request failed: {e}")

    data = r.json()
    # expected: data contains "embeddings" or for single item maybe "embedding"
    # We handle common shapes
    if isinstance(data, dict) and "embeddings" in data:
        embs = data["embeddings"]
    elif isinstance(data, list):
        # sometimes the API returns list of dicts
        # e.g. [{"embedding": [...]}, ...]
        embs = []
        for item in data:
            if isinstance(item, dict) and "embedding" in item:
                embs.append(item["embedding"])
            else:
                # attempt to treat item as raw vector
                embs.append(item)
    elif isinstance(data, dict) and "embedding" in data:
        embs = [data["embedding"]]
    else:
        raise RuntimeError(f"Unexpected embeddings response from Ollama: {data}")

    arr = np.array(embs, dtype="float32")
    return arr

# --- Ollama text generation helper ---------------------------------------
def generate_with_ollama(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """
    Use Ollama generation endpoint to produce text.
    Uses /api/generate with model and prompt, returns generated text string.
    """
    print("Generating text with Ollama...")
    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_GEN_ENDPOINT, json=payload, timeout=120)
        r.raise_for_status()  # Raises HTTPError for bad status codes (e.g., 4xx, 5xx)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Failed to connect to Ollama server at {OLLAMA_GEN_ENDPOINT}: {e}")
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Ollama request timed out after 120 seconds: {e}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e} - Response: {r.text}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError(f"Failed to parse Ollama JSON response: {e} - Response: {r.text}")

    # Log the raw response for debugging
    print(f"Ollama response: {data}")

    # Handle common response formats
    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"].strip()
        if "text" in data and isinstance(data["text"], str):
            return data["text"].strip()
        if "outputs" in data and isinstance(data["outputs"], list) and len(data["outputs"]) > 0:
            out0 = data["outputs"][0]
            if isinstance(out0, dict):
                if "content" in out0 and isinstance(out0["content"], str):
                    return out0["content"].strip()
                if "message" in out0 and isinstance(out0["message"], dict):
                    msg = out0["message"]
                    if "content" in msg and isinstance(msg["content"], str):
                        return msg["content"].strip()
    raise RuntimeError(f"Unexpected Ollama response format: {data}")

# --- Indexing & retrieval (FAISS) ----------------------------------------
def ensure_index_for_pdf(pdf_path: str, pages: List[dict] = None):
    """
    Build and cache FAISS index + metadata for the given PDF path.
    - pages: optional pre-parsed page dicts (from parse_pdf)
    - stores <filename>.faiss and <filename>.pkl under EMBEDDINGS_DIR
    """
    print("Ensuring FAISS index for PDF...")
    key = os.path.basename(pdf_path)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{key}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        return idx_path

    # parse pdf if pages not provided
    from app.pdf.extract import parse_pdf
    if pages is None:
        pages = parse_pdf(pdf_path)

    texts = []
    metadatas = []
    for p in pages:
        page_num = p["page"]
        page_text = p.get("text", "").strip()
        # chunk page text into segments
        chunks = chunk_text(page_text, max_chars=1200, overlap=200)
        for i, c in enumerate(chunks):
            # skip empty chunks
            if not c.strip():
                continue
            texts.append(c)
            metadatas.append({"page": page_num, "chunk_id": i})

    if len(texts) == 0:
        raise RuntimeError("No text extracted from PDF to index.")

    # compute embeddings via Ollama
    embeddings = embed_with_ollama(texts)  # shape (n, dim)
    dim = embeddings.shape[1]

    # build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, idx_path)

    # save metadata (texts + metadatas)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    return idx_path

def _load_index(pdf_path: str):
    print("Loading FAISS index for PDF...")
    key = os.path.basename(pdf_path)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{key}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        ensure_index_for_pdf(pdf_path)
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def retrieve_top_k(question: str, pdf_path: str, k: int = 3):
    """
    Return top-k retrieved chunks (text + metadata) for the question.
    """
    print("Retrieving top-k chunks...")
    index, meta = _load_index(pdf_path)
    q_emb = embed_with_ollama([question])  # shape (1, dim)
    # faiss expects float32
    if q_emb.dtype != np.float32:
        q_emb = q_emb.astype("float32")
    D, I = index.search(q_emb, k)
    seen_pages = set()
    results = []
    for idx in I[0]:
        text = meta["texts"][idx]
        md = meta["metadatas"][idx]
        page = md["page"]
        if page not in seen_pages:  # only take the first chunk from each page
            seen_pages.add(page)
            results.append({"text": text, "metadata": md})
    return results

# --- System prompt builder ------------------------------------------------
def _build_system_prompt(mode: str):
    print("Building system prompt...")
    if mode.lower() == "novice":
        return (
            "You are an assistant that explains scientific papers to a novice. "
            "Give a clear, concise answer with high-level intuition and point to the source pages. "
            "Keep technical jargon minimal and define any introduced terms."
            "Summarize the content with structure (Problem, Method, Key Equation, Result and Limits)"
        )
    elif mode.lower() == "reviewer":
        return (
            "You are an assistant that helps peer reviewers. Provide critical analysis, list assumptions, "
            "possible threats to validity, and point to specific pages/paragraphs supporting claims."
        )
    else:
        return (
            "You are an assistant that explains scientific papers to an informed researcher. Provide precise, "
            "technical answers, include equations when relevant, and cite page numbers."
        )

# --- High-level pipeline --------------------------------------------------
def process_question(question: str, mode: str, pdf_path: str, k: int = 3):
    """
    Main entrypoint used by the backend.
    Returns: (answer_text, sources_list)
      - sources_list: list of {"page": n, "snippet": "..."}
    """
    print("Processing question...")
    # retrieve
    contexts = retrieve_top_k(question, pdf_path, k=k)

    # prepare system & user prompts
    system_prompt = _build_system_prompt(mode)
    context_blocks = []
    sources = []
    for c in contexts:
        page = c["metadata"]["page"]
        snippet = c["text"][:1000].strip()
        context_blocks.append(f"[page {page}] {snippet}")
        sources.append({"page": page, "snippet": snippet})

    user_prompt = (
        "You are given the following snippets from a target paper (each labeled by page). "
        "Answer the question using ONLY the provided snippets. Quote page numbers inline where relevant. "
        f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
    )

    # generate answer with Ollama
    try:
        answer_text = generate_with_ollama(system_prompt, user_prompt, max_tokens=800)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")
        print(f"Error during generation: {e}")

    return answer_text, sources

# Optional utility: get full page content for UI 'Show context'
def get_page_text(pdf_path: str, page_number: int):
    print("Getting full page text...")
    from app.pdf.extract import parse_pdf
    pages = parse_pdf(pdf_path)
    for p in pages:
        if p["page"] == page_number:
            return p
    return None


def generate_eval_dataset_from_pdf(pdf_path, num_samples=10, num_iterations=3, output_dir="eval_outputs"):
    """
    Use the LLM to generate evaluation data multiple times and append results.
    Ensures results are appended to existing file if it exists.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path)}.eval.json")

    # Parse PDF once
    pages = parse_pdf(pdf_path)
    content = "\n".join([f"[page {p['page']}] {p['text'][:1000]}" for p in pages if p['text'].strip()])
    
    # Track existing questions to avoid duplicates
    existing_questions = set()
    all_results = []
    
    # Load existing results if file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if 'eval_data' in existing_data:
                    start_idx = existing_data['eval_data'].find('[')
                    end_idx = existing_data['eval_data'].rfind(']') + 1
                    if start_idx != -1 and end_idx != -1:
                        try:
                            existing_qa = json.loads(existing_data['eval_data'][start_idx:end_idx])
                            # Add existing QA pairs to results
                            all_results.extend(existing_qa)
                            # Track existing questions
                            existing_questions.update(qa['question'] for qa in existing_qa)
                            logger.info(f"Loaded {len(existing_qa)} existing QA pairs")
                        except json.JSONDecodeError:
                            logger.warning("Could not parse existing QA pairs")
        except Exception as e:
            logger.warning(f"Error loading existing file: {e}")
    
    # Generate new QA pairs with updated prompt
    for i in range(num_iterations):
        try:
            logger.info(f"Generating batch {i+1}/{num_iterations}")
            
            # Update prompt with existing questions to avoid
            system_prompt = (
                "You are a scientific assistant. Generate different evaluation questions.\n"
                "IMPORTANT: Return ONLY valid JSON array with question-answer pairs.\n"
                "Do not include any other text before or after the JSON array."
            )
            
            user_prompt = f"""
Based on the following content, generate {num_samples} DIFFERENT pairs of question and concise answer.
Each answer must cite the source (page number). Focus on key concepts.

Requirements:
- Questions must be different from existing ones
- Answers must be concise and cite specific sources
- Focus on important concepts from the paper
- Return ONLY the JSON array in this exact format:
[
  {{
    "question": "What is...",
    "answer": "According to...",
    "source": "[page X]"
  }},
  ...
]

Content:
{content}
"""
            
            result = generate_with_ollama(system_prompt, user_prompt)
            logger.debug(f"Raw LLM response: {result[:200]}...")
            
            # Clean and validate JSON response
            result = result.strip()
            if not result.startswith('['):
                # Try to find JSON array
                start_idx = result.find('[')
                if start_idx == -1:
                    logger.error(f"No JSON array found in response: {result[:100]}...")
                    continue
                result = result[start_idx:]
            
            if not result.endswith(']'):
                # Try to find end of JSON array
                end_idx = result.rfind(']')
                if end_idx == -1:
                    logger.error(f"No closing bracket found in response: {result[-100:]}")
                    continue
                result = result[:end_idx + 1]
            
            try:
                # Parse new QA pairs
                new_pairs = json.loads(result)
                if not isinstance(new_pairs, list):
                    logger.error(f"Expected JSON array, got: {type(new_pairs)}")
                    continue
                
                # Validate each pair
                valid_pairs = []
                for pair in new_pairs:
                    if not isinstance(pair, dict):
                        continue
                    if not all(k in pair for k in ('question', 'answer', 'source')):
                        continue
                    if not all(isinstance(pair[k], str) for k in ('question', 'answer', 'source')):
                        continue
                    valid_pairs.append(pair)
                
                # Filter duplicates
                unique_pairs = [
                    pair for pair in valid_pairs 
                    if pair['question'] not in existing_questions
                ]
                
                # Update tracking
                existing_questions.update(pair['question'] for pair in unique_pairs)
                all_results.extend(unique_pairs)
                
                logger.info(f"Added {len(unique_pairs)} new unique QA pairs from {len(valid_pairs)} valid pairs")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in iteration {i+1}: {str(e)}")
                logger.error(f"Invalid JSON: {result[:100]}...")
                continue
            
            # Add delay between calls
            if i < num_iterations - 1:
                time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {str(e)}")
            continue

    # Format final result
    final_result = (
        "Here is an evaluation dataset for a document QA system based on the provided content:\n\n"
        "```json\n" + 
        json.dumps(all_results, indent=2, ensure_ascii=False) +
        "\n```"
    )

    # Save combined results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "pdf": os.path.basename(pdf_path),
            "eval_data": final_result
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved total of {len(all_results)} QA pairs to {output_file}")
    return final_result

# Utility to convert saved eval JSON to list of (question, answer, context)
def convert_json_to_qa_list(json_file):
    """Convert evaluation JSON file to list of (question, answer, context) tuples."""
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.debug(f"Raw file content: {content[:200]}...")
            data = json.loads(content)
        
        # Extract the eval_data string
        eval_data = data.get('eval_data')
        if not eval_data:
            raise ValueError("No eval_data field found in JSON")
        
        # Find the JSON array part
        start_marker = "```json\n["
        end_marker = "]\n```"
        
        start_idx = eval_data.find(start_marker)
        if start_idx == -1:
            start_idx = eval_data.find("```\n[")
            if start_idx == -1:
                raise ValueError("Could not find start of JSON array")
        start_idx = eval_data.find("[", start_idx)
        
        end_idx = eval_data.find(end_marker)
        if end_idx == -1:
            end_idx = eval_data.find("]```")
        if end_idx == -1:
            raise ValueError("Could not find end of JSON array")
        end_idx = eval_data.rfind("]", 0, end_idx + 1)
        
        # Extract and clean the JSON array
        json_str = eval_data[start_idx:end_idx + 1]
        
        # Process the JSON string line by line
        lines = json_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or '...' in line:
                continue
                
            # Quote unquoted source values
            if '"source":' in line:
                source_start = line.find('"source":') + len('"source":')
                source_value = line[source_start:].strip()
                if source_value.startswith('"'):
                    cleaned_lines.append(line)
                else:
                    # Remove trailing comma if present
                    if source_value.endswith(','):
                        source_value = source_value[:-1]
                    # Quote the source value
                    quoted_line = line[:source_start] + f' "{source_value}"'
                    if line.endswith(','):
                        quoted_line += ','
                    cleaned_lines.append(quoted_line)
            else:
                cleaned_lines.append(line)
        
        # Reconstruct valid JSON
        cleaned_json = '\n'.join(cleaned_lines)
        # Remove trailing commas
        cleaned_json = cleaned_json.replace(',\n}', '\n}')
        cleaned_json = cleaned_json.replace(',\n]', '\n]')
        
        logger.debug(f"Cleaned JSON string: {cleaned_json[:200]}...")
        
        # Parse JSON and convert to tuples
        qa_json = json.loads(cleaned_json)
        qa_list = [
            (item['question'], item['answer'], item['source'].strip('[]')) 
            for item in qa_json
            if all(k in item for k in ('question', 'answer', 'source'))
        ]
        
        return qa_list
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error at {e.pos}: {e.msg}")
        logger.error(f"Error context: {e.doc[max(0, e.pos-50):e.pos+50]}")
        raise
    except Exception as e:
        logger.error(f"Error converting JSON to QA list: {str(e)}")
        raise