import os
import pickle
import time
from typing import List
import requests
import numpy as np
import faiss
from app.pdf.extract import parse_pdf

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
    Chunk text into overlapping windows by character length.
    Ensures forward progress to avoid infinite loops.
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

        # advance start while keeping overlap
        if end == length:  # reached end, stop
            break
        start = end - overlap
        if start < 0:  # safety
            start = 0
        if start >= end:  # safety against infinite loop
            start = end
    return chunks


# --- Ollama embedding helper ----------------------------------------------
def embed_with_ollama(texts: List[str], retries: int = 3, timeout: int = 120) -> np.ndarray:
    print(f"Embedding {len(texts)} texts with Ollama...")
    if not isinstance(texts, list):
        texts = [texts]
    if not texts:
        raise RuntimeError("No texts provided for embedding")

    expected_dim = 768  # Default dimension for nomic-embed-text
    embeddings = []

    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)} (length: {len(text)} chars)")
        embedding = None
        attempt = 0
        while attempt < retries:
            try:
                payload = {"model": EMBED_MODEL_NAME, "prompt": text}
                print(f"Sending payload for text {i+1}: {payload}")
                r = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                print(f"Ollama embeddings response for text {i+1}: {data}")
                
                if isinstance(data, dict) and "embedding" in data and data["embedding"]:
                    embedding = data["embedding"]
                elif isinstance(data, dict) and "embeddings" in data and data["embeddings"]:
                    embedding = data["embeddings"][0]
                else:
                    print(f"Unexpected response format for text {i+1}: {data}")
                    embedding = None
                break
            except requests.exceptions.RequestException as e:
                print(f"Embedding attempt {attempt + 1} failed for text {i+1}: {e}, Response: {r.text if 'r' in locals() else 'No response'}")
                attempt += 1
                if attempt == retries:
                    print(f"Failed to embed text {i+1} after {retries} attempts, using zero vector")
                    embedding = [0.0] * expected_dim
                time.sleep(1)

        if not embedding or len(embedding) != expected_dim:
            print(f"Invalid embedding for text {i+1}, using zero vector")
            embeddings.append([0.0] * expected_dim)
        else:
            embeddings.append(embedding)

    try:
        arr = np.array(embeddings, dtype="float32")
    except ValueError as e:
        print(f"NumPy array creation failed: {e}, ensuring consistent dimensions")
        embeddings = [emb if len(emb) == expected_dim else [0.0] * expected_dim for emb in embeddings]
        arr = np.array(embeddings, dtype="float32")

    print(f"Generated embeddings shape: {arr.shape}")
    if arr.shape[0] != len(texts):
        raise RuntimeError(f"Invalid number of embeddings: expected {len(texts)}, got {arr.shape[0]}")
    if arr.shape[1] != expected_dim:
        raise RuntimeError(f"Invalid embedding dimension: expected {expected_dim}, got {arr.shape[1]}")
    if np.any(np.isnan(arr)) or np.any(np.all(arr == 0, axis=1)):
        print("Warning: Invalid embeddings detected (NaN or zero vectors)")
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
        r = requests.post(OLLAMA_GEN_ENDPOINT, json=payload, timeout=300)
        r.raise_for_status()  # Raises HTTPError for bad status codes (e.g., 4xx, 5xx)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Failed to connect to Ollama server at {OLLAMA_GEN_ENDPOINT}: {e}")
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Ollama request timed out after 300 seconds: {e}")
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
    results = []
    for idx in I[0]:
        text = meta["texts"][idx]
        md = meta["metadatas"][idx]
        page = md["page"]
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
    pages = parse_pdf(pdf_path)
    for p in pages:
        if p["page"] == page_number:
            return p
    return None

# Get all formulas from the PDF
def get_formulas(pdf_path: str):
    print(f"Extracting formulas from {pdf_path}...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        pages = parse_pdf(pdf_path)
        if not pages:
            print("No pages extracted from PDF")
            return []
        formulas = []
        for p in pages:
            page_num = p.get("page")
            page_formulas = p.get("formulas", [])
            print(f"Page {page_num}: Found {len(page_formulas)} formulas")
            for f in page_formulas:
                if f and isinstance(f, str):  # Ensure formula is non-empty string
                    formulas.append({"page": page_num, "formula": f.strip()})
        print(f"Total formulas extracted: {len(formulas)}")
        return formulas
    except Exception as e:
        print(f"Error in get_formulas: {str(e)}")
        raise