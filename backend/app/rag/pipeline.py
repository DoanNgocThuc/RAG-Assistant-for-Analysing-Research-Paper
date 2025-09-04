import os
import pickle
import time
from typing import List
import requests
import numpy as np
import faiss
from app.pdf.extract import parse_pdf
from app.evaluator.evaluate import evaluate_rag_with_gemini
import json

# Local Ollama endpoints and models
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_HOST}/api/embeddings"
OLLAMA_GEN_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
EMBED_MODEL_NAME = "nomic-embed-text"   # ensure you pulled this: ollama pull nomic-embed-text
LLM_MODEL_NAME = "llama3.2"              # ensure you pulled this: ollama pull llama3.2
CHUNK_SIZE = 1200
OVERLAP_SIZE = 200
SNIPPET_SIZE = CHUNK_SIZE - OVERLAP_SIZE

EMBEDDINGS_DIR = os.path.join(os.getcwd(), "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# --- Helpers: chunking ----------------------------------------------------
def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE):
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
                #print(f"Sending payload for text {i+1}: {payload}")
                r = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                #print(f"Ollama embeddings response for text {i+1}: {data}")
                
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
    #print(f"Ollama response: {data}")

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
        chunks = chunk_text(page_text, max_chars=CHUNK_SIZE, overlap=OVERLAP_SIZE)
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

# --- Reranking with LLM ------------------------------------------
def rerank_chunks(question, chunks, k:int =3):
    """
    Dùng LLM để đánh giá lại mức độ liên quan của từng chunk với câu hỏi.
    Trả về danh sách các chunk đã được sắp xếp lại, loại bỏ các chunk có rerank_score bằng 0.
    """
    reranked = []
    for chunk in chunks:
        system_prompt = "Bạn là chuyên gia. Đánh giá mức độ liên quan của đoạn sau với câu hỏi."
        user_prompt = (
            f"Câu hỏi: {question}\n"
            f"Đoạn: {chunk['text']}\n"
            "Trả về kết quả dưới dạng JSON với định dạng: {\"score\": số_thực_từ_0_đến_1}. "
            "Chỉ trả về JSON đó với đúng định dạng đã quy định, Không trả về thêm bất kỳ thông tin nào ngoài JSON này."
        )
        score_str = generate_with_ollama(system_prompt, user_prompt, max_tokens=20)
        print(f"Rerank score string: '{score_str}'")
        try:
            # Tìm JSON trong chuỗi trả về
            start = score_str.find('{')
            end = score_str.rfind('}')
            if start != -1 and end != -1:
                score_json = score_str[start:end+1]
                score = json.loads(score_json).get("score", 0.0)
            else:
                score = 0.0
            score = float(score)
        except Exception as e:
            print(f"Error parsing rerank score: {e}")
            score = 0.0
        reranked.append({**chunk, "rerank_score": score})
    # Loại bỏ các chunk có rerank_score bằng 0
    # reranked = [c for c in reranked if c["rerank_score"] > 0.5]
    reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:k]

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
    elif mode.lower() == "normal":
        return (
            "Keep it simple and avoid unnecessary jargon."
        )
    else:
        return (
            "You are an assistant that explains scientific papers to an informed researcher. Provide precise, "
            "technical answers, include equations when relevant, and cite page numbers."
        )

def explain_context(question:str, snippet:str):
    #print("Explaining context...")
    system_prompt = (
        "You are an assistant that explains the context of a given snippet in relation to a question. "
        "Provide a clear and concise explanation."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Snippet: {snippet}\n\n"
        "Explain why this snippet is relevant to the question and provide a brief reasoning."
    )

    try:
        explanation = generate_with_ollama(system_prompt, user_prompt, max_tokens=500)
    except Exception as e:
        explanation = f"Failed to generate explanation: {e}"

    return explanation

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

    # Sort chunks by page and chunk ID
    # sorted_chunks_contexts = reindex_contexts(contexts)

    for c in contexts:
        page = c["metadata"]["page"]
        snippet = c["text"][:SNIPPET_SIZE].strip()
        context_blocks.append(f"[page {page}] {snippet}")
        explanation = explain_context(question=question, snippet=snippet)
        sources.append({"page": page, "snippet": snippet, "explanation": explanation})

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

def process_question_reranked(question: str, mode: str, pdf_path: str, k: int = 3):
    """
    Main entrypoint used by the backend.
    Returns: (answer_text, sources_list)
      - sources_list: list of {"page": n, "snippet": "..."}
    """
    print("Processing question...")
    # retrieve
    contexts = retrieve_top_k(question, pdf_path, k=k)
    print(f"Retrieved {len(contexts)} chunks.")
    # rerank
    reranks = rerank_chunks(question, contexts, k=3)
    print(f"Reranked chunks: {[ (c['metadata']['page'], c['rerank_score']) for c in reranks ]}")

    # prepare system & user prompts
    system_prompt = _build_system_prompt(mode)
    context_blocks = []
    sources = []

    # Sort chunks by page and chunk ID
    # sorted_chunks_contexts = reindex_contexts(contexts)

    for c in reranks:
        page = c["metadata"]["page"]
        snippet = c["text"][:SNIPPET_SIZE].strip()
        context_blocks.append(f"[page {page}] {snippet}")
        explanation = explain_context(question=question, snippet=snippet)
        sources.append({"page": page, "snippet": snippet, "explanation": explanation})

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

def evaluate_RAG (question_groundtruth:list, pdf_path:str):
    print("Evaluating RAG...")
    answers = []
    questions = []
    reference_contexts_list = []
    contexts = []
    references = []
    for item in question_groundtruth:
        # Extract question
        question = item["question"]
        questions.append(question)

        # Extract references
        reference_contexts = item["groundtruth_chunks"]
        reference_contexts_list.append(reference_contexts)

        # Extract ground truth answer
        groundtruth_answer = item["groundtruth_answer"]
        references.append(groundtruth_answer)

        # Process the question -> answer, sources
        generated_answer,sources = process_question(question, mode="normal", pdf_path=pdf_path, k=3)

        # Append the generated answer to the answers list
        answers.append(generated_answer)

        # Append the sources to the contexts list
        snippets = []
        snippets = [item["snippet"] for item in sources]
        contexts.append(snippets)

        # Compare the generated answer with the ground truth
        # Return a score between 0 and 1

    evaluation_triad ={
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "reference": references,
        "reference_contexts": reference_contexts_list
    }
    score = evaluate_rag_with_gemini(rag_triad = evaluation_triad)
    return score

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

def evaluate_answer_relevancy(evaluation_triad):
    """
    Đánh giá độ liên quan của answer với question cho 1 QA pair.
    Args:
        evaluation_triad: dict với keys "question", "contexts", "answer"
    Returns:
        relevance_score (float between 0 and 1)
    """
    print("Evaluating answer relevancy...")
    question = evaluation_triad["question"][0]
    answer = evaluation_triad["answer"][0]
    contexts = evaluation_triad["contexts"][0]  # list các context

    system_prompt = """You are an expert evaluator for question answering systems.
Your task is to evaluate how relevant and responsive the answers are to their questions.
Focus on whether the answer directly addresses what was asked.

Score relevance using these exact criteria:
0 = Answer is completely off-topic or unrelated to the question
0.1 - 0.3 = Answer barely relates to the question's topic
0.4 - 0.6 = Answer partially addresses the question but misses key aspects
0.7 - 0.9 = Answer is mostly relevant and addresses the question well, but could be more focused
1 = Answer perfectly matches the question's requirements with excellent focus

IMPORTANT: Return ONLY valid JSON with scores and explanations.
Do not include any other text before or after the JSON."""

    user_prompt = f"""
Evaluate the relevance of this answer to its question:

Question: {question}
Answer: {answer}
Contexts: {json.dumps(contexts, ensure_ascii=False, indent=2)}

Analyze:
1. How directly the answer addresses the specific question asked
2. Whether the answer includes unnecessary or off-topic information
3. Whether the answer covers all aspects of the question
4. The focus and precision of the answer

Return your evaluation in this exact JSON format:
{{
    "relevance_score": score_between_0_and_1,
    "explanation": "Detailed explanation of the score",
    "addressed_aspects": ["List aspects of the question that were addressed"],
    "missing_aspects": ["List aspects of the question that were not addressed"],
    "off_topic_content": ["List any irrelevant or unnecessary content"]
}}
"""

    result = generate_with_ollama(system_prompt, user_prompt)
    result = result.strip()
    if not result.startswith('{'):
        start_idx = result.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        result = result[start_idx:]
    if not result.endswith('}'):
        end_idx = result.rfind('}')
        if end_idx == -1:
            raise ValueError("No closing brace found in response")
        result = result[:end_idx + 1]

    evaluation = json.loads(result)
    return evaluation["relevance_score"]

def suggest_related_papers_with_difference(pdf_path: str, top_n: int = 5, similarity_threshold: float = 0.9):
    """
    Gợi ý các PDF liên quan dựa trên semantic similarity giữa các chunks.
    Trả về danh sách tối đa top_n PDF liên quan nhất (file name, điểm similarity trung bình, các chunk liên quan, mô tả khác biệt).
    """
    print(f"Suggesting related papers for: {pdf_path}")
    key = os.path.basename(pdf_path)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{key}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        ensure_index_for_pdf(pdf_path)
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    current_embeddings = index.reconstruct_n(0, index.ntotal)

    candidates = []
    for fname in os.listdir(EMBEDDINGS_DIR):
        print(f"Checking {fname}")
        if fname.endswith(".faiss") and fname != f"{key}.faiss":
            other_pdf = fname[:-6]
            other_idx_path = os.path.join(EMBEDDINGS_DIR, fname)
            other_meta_path = os.path.join(EMBEDDINGS_DIR, f"{other_pdf}.pkl")
            if not os.path.exists(other_meta_path):
                continue
            other_index = faiss.read_index(other_idx_path)
            with open(other_meta_path, "rb") as f:
                other_meta = pickle.load(f)
            other_embeddings = other_index.reconstruct_n(0, other_index.ntotal)

            # Lưu lại index các chunk liên quan của cả hai bên
            main_related_idxs = []
            other_related_idxs = []

            def cosine_sim(a, b):
                a = a / (np.linalg.norm(a) + 1e-8)
                b = b / (np.linalg.norm(b) + 1e-8)
                return np.dot(a, b)

            sim_scores = []
            for main_idx, emb in enumerate(current_embeddings):
                scores = [cosine_sim(emb, oemb) for oemb in other_embeddings]
                max_score = max(scores) if scores else 0
                if max_score >= similarity_threshold:
                    sim_scores.append(max_score)
                    main_related_idxs.append(main_idx)
                    other_idx = scores.index(max_score)
                    other_related_idxs.append(other_idx)

            if sim_scores:
                avg_score = float(np.mean(sim_scores))
                # Sinh mô tả khác biệt bằng LLM dựa trên các chunk liên quan
                difference = describe_difference_with_chunks(
                    main_meta=meta, main_chunks=main_related_idxs, other_meta=other_meta, other_chunks=other_related_idxs
                )
                candidates.append({
                    "pdf": other_pdf,
                    "avg_similarity": avg_score,
                    "num_related_chunks": len(sim_scores),
                    "main_related_idxs": main_related_idxs,
                    "other_related_idxs": other_related_idxs,
                    "difference": difference
                })
    candidates = sorted(candidates, key=lambda x: x["avg_similarity"], reverse=True)[:top_n]
    print(f"Related papers found: {candidates}")
    return candidates

def describe_difference_with_chunks(main_meta, main_chunks, other_meta, other_chunks, max_len=600):
    """
    Sinh mô tả khác biệt chính giữa hai bài báo bằng LLM, dựa trên các chunk liên quan nhất.
    - main_chunks, other_chunks: list các chunk text liên quan (có similarity cao)
    """
    def join_chunks(meta, chunks):
        # Lấy thông tin trang và nội dung
        result = []
        for idx in chunks:
            page = meta["metadatas"][idx]["page"] if "metadatas" in meta else "?"
            text = meta["texts"][idx][:max_len] if "texts" in meta else ""
            result.append(f"[Page {page}] {text}")
        return "\n".join(result)

    main_text = join_chunks(main_meta, main_chunks)
    other_text = join_chunks(other_meta, other_chunks)

    system_prompt = (
        "Bạn là chuyên gia khoa học. Hãy phân tích và nêu rõ sự khác biệt chính giữa hai bài báo sau, "
        "dựa trên các đoạn nội dung liên quan nhất. Chỉ liệt kê các điểm khác biệt chính, không nêu điểm giống nhau, "
        "không trả lời dài dòng. Trả về đúng format: khác biệt chính: ... (khác biệt chính giữa bài báo hiện tại và bài báo được gợi ý, không thêm thông tin khác)."
    )
    user_prompt = (
        f"Bài báo hiện tại (các đoạn liên quan):\n{main_text}\n\n"
        f"Bài báo gợi ý (các đoạn liên quan):\n{other_text}\n\n"
        "Hãy trả về đúng format: khác biệt chính: ... (không thêm thông tin khác)."
    )
    return generate_with_ollama(system_prompt, user_prompt, max_tokens=100)