import os
import pickle
import time
from typing import List, Dict
import requests
import numpy as np
import faiss
import json
from app.pdf.extract import parse_pdf
import logging
from datasets import load_dataset
import datetime
import re



ds = load_dataset("neural-bridge/rag-dataset-12000")

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
    Sends each text string individually to the API, as Ollama only accepts one string per request.
    """
    print("Embedding text with Ollama...")
    if not isinstance(texts, list):
        texts = [texts]

    embeddings = []
    for text in texts:
        payload = {"model": EMBED_MODEL_NAME, "prompt": text}
        try:
            r = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=60)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama embeddings request failed: {e}")

        data = r.json()
        # expected: data contains "embedding"
        if isinstance(data, dict) and "embedding" in data:
            emb = data["embedding"]
        elif isinstance(data, dict) and "embeddings" in data:
            emb = data["embeddings"][0]
        elif isinstance(data, list) and len(data) > 0:
            # sometimes the API returns list of dicts
            # e.g. [{"embedding": [...]}, ...]
            if isinstance(data[0], dict) and "embedding" in data[0]:
                emb = data[0]["embedding"]
            else:
                emb = data[0]
        else:
            raise RuntimeError(f"Unexpected embeddings response from Ollama: {data}")

        embeddings.append(emb)

    arr = np.array(embeddings, dtype="float32")
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
    logger.info(f"Index key: {key}")

    idx_path = os.path.join(EMBEDDINGS_DIR, f"{key}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")

    logger.info(f"Index path: {idx_path}")
    logger.info(f"Metadata path: {meta_path}")

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        print("Index already exists, skipping indexing.")
        return idx_path

    # parse pdf if pages not provided
    from app.pdf.extract import parse_pdf
    if pages is None:
        print(f"Parsing PDF... {pdf_path}")
        pages = parse_pdf(pdf_path)
    if not pages or len(pages) == 0:
        raise RuntimeError("No pages found in PDF.")

    logger.info(f"Parsed {len(pages)} pages from PDF.")

    texts = []
    metadatas = []
    for p in pages:
        page_num = p["page"]
        print(f"Processing page {page_num}...")
        page_text = p.get("text", "").strip()
        # chunk page text into segments
        chunks = chunk_text(page_text, max_chars=1200, overlap=200)
        for i, c in enumerate(chunks):
            print(f"Chunk {i} from page {page_num}: {c}")
            # skip empty chunks
            if not c.strip():
                continue
            texts.append(c)
            metadatas.append({"page": page_num, "chunk_id": i})

    if len(texts) == 0:
        raise RuntimeError("No text extracted from PDF to index.")

    # compute embeddings via Ollama
    embeddings = embed_with_ollama(texts)  # shape (n, dim)
    logger.info(f"Computed embeddings shape: {embeddings.shape}")
    dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dim}")

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

def evaluate_answer_relevance(
    benchmark_file: str,
    output_dir: str = "eval_outputs"
) -> dict:
    """
    Evaluate answer relevance of QA pairs loaded from a benchmark JSON file.
    This measures how well the answer addresses the actual question being asked.

    Args:
        benchmark_file: Path to the benchmark dataset file (in benchmark_datasets)
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary containing evaluation results and scores
    """
    logger.info(f"Loading benchmark dataset from {benchmark_file}")

    # Load QA pairs from benchmark file
    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Try to extract QA pairs from common keys
        if "data" in data:
            # If multiple splits, flatten all
            qa_pairs = []
            if isinstance(data["data"], dict):
                for split in data["data"].values():
                    qa_pairs.extend(split)
            else:
                qa_pairs = data["data"]
        elif "qa_pairs" in data:
            qa_pairs = data["qa_pairs"]
        else:
            qa_pairs = data
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from benchmark")
    except Exception as e:
        logger.error(f"Error loading benchmark file: {str(e)}")
        raise

    # Only keep pairs with question and answer
    filtered_pairs = []
    for pair in qa_pairs:
        if all(k in pair for k in ("question", "answer")):
            filtered_pairs.append({
                "question": pair["question"],
                "answer": pair["answer"]
            })
    qa_pairs = filtered_pairs

    # Answer relevance evaluation logic
    system_prompt = """You are an expert evaluator for question answering systems.
Your task is to evaluate how relevant and responsive the answers are to their questions.
Focus on whether the answer directly addresses what was asked.

Score relevance using these exact criteria:
1 = Answer is completely off-topic or unrelated to the question
2 = Answer barely relates to the question's topic
3 = Answer touches on the topic but doesn't address the specific question
4 = Answer partially addresses the question but misses key aspects
5 = Answer addresses the main point but could be more focused
6 = Answer is mostly relevant with minor divergences
7 = Answer is relevant and addresses the question well
8 = Answer is highly relevant with good focus on what was asked
9 = Answer directly addresses all aspects of the question
10 = Answer perfectly matches the question's requirements with excellent focus

IMPORTANT: Return ONLY valid JSON with scores and explanations.
Do not include any other text before or after the JSON."""

    all_evaluations = []

    for i, pair in enumerate(qa_pairs, 1):
        try:
            user_prompt = f"""
Evaluate the relevance of this answer to its question:

Question: {pair['question']}
Answer: {pair['answer']}

Analyze:
1. How directly the answer addresses the specific question asked
2. Whether the answer includes unnecessary or off-topic information
3. Whether the answer covers all aspects of the question
4. The focus and precision of the answer

Return your evaluation in this exact JSON format:
{{
    "relevance_score": score_between_1_and_10,
    "explanation": "Detailed explanation of the score",
    "addressed_aspects": ["List aspects of the question that were addressed"],
    "missing_aspects": ["List aspects of the question that were not addressed"],
    "off_topic_content": ["List any irrelevant or unnecessary content"]
}}"""

            # Get LLM evaluation
            result = generate_with_ollama(system_prompt, user_prompt)
            logger.debug(f"Raw LLM response for pair {i}: {result[:200]}...")

            # Clean and parse response
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

            # Parse evaluation
            evaluation = json.loads(result)

            # Add question for reference
            evaluation['question'] = pair['question']
            all_evaluations.append(evaluation)

            logger.info(f"Evaluated pair {i}/{len(qa_pairs)}: score = {evaluation['relevance_score']}")

            # Add delay between evaluations
            if i < len(qa_pairs):
                time.sleep(2)

        except Exception as e:
            logger.error(f"Error evaluating pair {i}: {str(e)}")
            continue

    # Calculate statistics
    scores = [e['relevance_score'] for e in all_evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0

    evaluation_results = {
        "evaluations": all_evaluations,
        "statistics": {
            "total_pairs": len(qa_pairs),
            "evaluated_pairs": len(all_evaluations),
            "average_relevance": round(avg_score, 2),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0
        }
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"relevance_evaluation.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation results to {output_file}")
    logger.info(f"Average relevance score: {avg_score:.2f}")

    return evaluation_results

def evaluate_faithfulness(
    benchmark_file: str,
    output_dir: str = "benchmark_datasets"
) -> dict:
    """
    Đánh giá độ trung thực (faithfulness) của các cặp QA bằng cách tính tỷ lệ giữa
    số câu được context hỗ trợ và tổng số câu trong câu trả lời.

    Args:
        benchmark_file (str): Đường dẫn đến file benchmark dataset
        output_dir (str, optional): Thư mục lưu kết quả đánh giá. Mặc định: "benchmark_datasets"

    Returns:
        dict: Dictionary chứa kết quả đánh giá và các thống kê
    """

    # 1. Load dữ liệu từ benchmark file
    logger.info(f"Đang load benchmark dataset từ {benchmark_file}")
    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Xử lý các format JSON khác nhau
        if "data" in data:
            qa_pairs = []
            if isinstance(data["data"], dict):
                for split in data["data"].values():
                    qa_pairs.extend(split)
            else:
                qa_pairs = data["data"]
        elif "qa_pairs" in data:
            qa_pairs = data["qa_pairs"]
        else:
            qa_pairs = data
    except Exception as e:
        logger.error(f"Lỗi khi load benchmark file: {str(e)}")
        raise

    # 2. Lọc và chuẩn hóa cặp QA
    filtered_pairs = [
        {
            "question": pair["question"],
            "answer": pair["answer"],
            "context": pair.get("context", "")
        }
        for pair in qa_pairs
        if all(k in pair for k in ("question", "answer"))
    ]
    qa_pairs = filtered_pairs

    # 3. Định nghĩa prompts
    system_prompt = """You are an expert evaluator for question answering systems.
Your task is to:
1. Extract ALL statements from the answer (put in total_statements)
2. From those statements, identify which ones are supported by context (put in supported_statements)
3. Calculate faithfulness as supported/total ratio

You must return response in this EXACT format:
```json
{
    "total_statements": ["ALL statements from answer"],
    "supported_statements": ["ONLY statements from total_statements that are supported by context"],
    "faithfulness_score": 0.XX,
    "analysis": "your analysis here"
}
```
IMPORTANT:
- supported_statements must be a subset of total_statements
- faithfulness_score must be between 0 and 1
- Do not include any text outside the JSON block"""

    # 4. Đánh giá từng cặp QA
    all_evaluations = []
    for i, pair in enumerate(qa_pairs, 1):
        try:
            # Tạo user prompt
            user_prompt = f"""Analyze this QA pair for faithfulness:

Question: {pair['question']}
Context: {pair['context']}
Answer: {pair['answer']}

Rules:
1. Extract ALL factual statements from the answer
2. List which statements are supported by the context
3. Calculate faithfulness = len(supported) / len(total)
4. Format your response exactly as shown in the system prompt
5. Use only ```json block for response"""

            # Gọi LLM để đánh giá
            result = generate_with_ollama(system_prompt, user_prompt)
            logger.debug(f"Raw LLM response for pair {i}: {result}")
            
            # Xử lý và parse kết quả
            try:
                result = result.strip()
                json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Thử tìm JSON object trực tiếp
                    start_idx = result.find('{')
                    end_idx = result.rfind('}')
                    if start_idx == -1 or end_idx == -1:
                        raise ValueError("Không tìm thấy JSON object trong response")
                    json_str = result[start_idx:end_idx + 1]
                
                # Parse JSON
                evaluation = json.loads(json_str)
                
                # Validate kết quả
                required_fields = ['total_statements', 'supported_statements']
                if not all(field in evaluation for field in required_fields):
                    raise ValueError(f"Thiếu các trường bắt buộc: {required_fields}")
                
                # Tính điểm faithfulness
                num_total = len(evaluation['total_statements'])
                num_supported = len(evaluation['supported_statements'])
                
                if num_total == 0:
                    logger.warning(f"Không tìm thấy statements cho cặp {i}")
                    continue
                    
                calculated_score = num_supported / num_total
                evaluation['faithfulness_score'] = round(calculated_score, 3)
                evaluation['question'] = pair['question']
                
                all_evaluations.append(evaluation)
                logger.info(f"Đã đánh giá cặp {i}/{len(qa_pairs)}: score = {evaluation['faithfulness_score']}")

            except Exception as e:
                logger.error(f"Lỗi xử lý cặp {i}: {str(e)}")
                logger.debug(f"Response gây lỗi: {result}")
                continue

            # Delay giữa các lần gọi LLM
            if i < len(qa_pairs):
                time.sleep(2)

        except Exception as e:
            logger.error(f"Lỗi đánh giá cặp {i}: {str(e)}")
            continue

    # 5. Tính toán thống kê
    scores = [e['faithfulness_score'] for e in all_evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0

    evaluation_results = {
        "evaluations": all_evaluations,
        "statistics": {
            "total_pairs": len(qa_pairs),
            "evaluated_pairs": len(all_evaluations),
            "average_faithfulness": round(avg_score, 3),
            "max_score": round(max(scores), 3) if scores else 0,
            "min_score": round(min(scores), 3) if scores else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 6. Lưu kết quả
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"faithfulness_evaluation_{int(time.time())}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã lưu kết quả đánh giá vào {output_file}")
    logger.info(f"Điểm faithfulness trung bình: {avg_score:.3f}")

    return evaluation_results

def load_rag_benchmark_dataset(subset=None, num_samples=None, save_to_file=False):
    """
    Load the RAG benchmark dataset from Hugging Face and optionally save to JSON file.
    
    Args:
        subset: Optional subset of data to return ('train', 'test', 'validation')
        num_samples: Optional number of samples to return (returns all if None)
        save_to_file: Boolean indicating whether to save dataset to JSON file
    
    Returns:
        Dictionary containing dataset information and samples
    """
    try:
        # Load dataset
        dataset = load_dataset("neural-bridge/rag-dataset-12000")
        logger.info("Successfully loaded RAG benchmark dataset")
        available_splits = list(dataset.keys())

        # Get specified split or all splits
        if subset:
            if subset not in available_splits:
                raise ValueError(f"Invalid subset '{subset}'. Available: {available_splits}")
            data = dataset[subset]
        else:
            # Combine all splits without raising error
            data = {split: dataset[split] for split in available_splits}

        # Convert to list and optionally limit samples
        result = {}
        
        if isinstance(data, dict):
            # Multiple splits
            for split, split_data in data.items():
                samples = split_data.to_pandas().to_dict('records')
                if num_samples:
                    samples = samples[:num_samples]
                result[split] = samples
        else:
            # Single split
            samples = data.to_pandas().to_dict('records')
            if num_samples:
                samples = samples[:num_samples]
            result = samples

        # Add metadata
        metadata = {
            "dataset": "neural-bridge/rag-dataset-12000",
            "available_splits": available_splits,
            "total_samples": len(samples) if not isinstance(data, dict) else {k: len(v) for k, v in result.items()},
            "subset": subset if subset else "all",
            "num_samples": num_samples if num_samples else "all",
            "timestamp": datetime.datetime.now().isoformat()
        }

        final_result = {
            "metadata": metadata,
            "data": result
        }

        # Save to JSON file if requested
        if save_to_file:
            # Create directory if it doesn't exist
            output_dir = "benchmark_datasets"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename based on parameters
            subset_str = subset if subset else "all"
            samples_str = str(num_samples) if num_samples else "all"
            filename = f"rag_benchmark_file.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved benchmark dataset to: {filepath}")

        return final_result

    except Exception as e:
        logger.error(f"Error loading RAG benchmark dataset: {str(e)}")
        raise

def get_candidate_chunks(question: str, pdf_path: str, similarity_threshold: float = 0.5) -> List[Dict]:
    """
    Bước 1: Lấy ra các đoạn văn ứng viên dựa trên vector similarity
    
    Args:
        question: Câu hỏi
        pdf_path: Đường dẫn đến file PDF
        similarity_threshold: Ngưỡng similarity để lọc ứng viên (0-1)
        
    Returns:
        Danh sách các đoạn văn ứng viên, mỗi đoạn chứa text và metadata
    """
    # 1. Lấy embeddings và tìm kiếm
    index, meta = _load_index(pdf_path)

    logger.info(f"Index loaded with {index.ntotal} vectors")
    logger.info(f"Metadata: {meta}")
    q_emb = embed_with_ollama([question])

    if q_emb.dtype != np.float32:
        q_emb = q_emb.astype("float32")
    
    # 2. Lấy tất cả vectors từ index
    total_chunks = index.ntotal
    logger.info(f"Total chunks in index: {total_chunks}")

    D, I = index.search(q_emb, total_chunks)
    logger.info(f"Distances: {D}")
    logger.info(f"Indices: {I}")

    if D is None or I is None or len(D) == 0 or len(I) == 0:
        logger.warning("No valid search results found.")
        return []
    if len(D[0]) != total_chunks or len(I[0]) != total_chunks:
        logger.warning("Unexpected search result lengths.")
        return []
    # 3. Chuyển distances thành similarities
    max_distance = max(D[0])
    similarities = [1 - (d / max_distance) for d in D[0]]
    
    # 4. Lọc chunks có similarity >= threshold
    candidates = []
    seen_pages = set()
    
    for idx, similarity in zip(I[0], similarities):
        if similarity < similarity_threshold:
            continue
            
        text = meta["texts"][idx]
        md = meta["metadatas"][idx]
        page = md["page"]
        
        if page not in seen_pages:  # Tránh trùng lặp trang
            candidates.append({
                "text": text,
                "metadata": md,
                "vector_similarity": similarity
            })
            seen_pages.add(page)
    
    print(f"Bước 1: Tìm thấy {len(candidates)} đoạn văn ứng viên")
    return candidates

def evaluate_context_relevance(question: str, context: str) -> Dict:
    """
    Đánh giá mức độ liên quan giữa câu hỏi và đoạn văn bản.
    
    Args:
        question: Câu hỏi cần đánh giá
        context: Đoạn văn bản cần so sánh với câu hỏi
        
    Returns:
        Dict chứa thông tin đánh giá:
        - relevance_score: Điểm liên quan (1-10)
        - analysis: Dict chứa phân tích chi tiết
        - key_matches: List các từ khóa trùng khớp
    """
    system_prompt = """Bạn là một chuyên gia đánh giá mức độ liên quan giữa câu hỏi và đoạn văn bản.
Nhiệm vụ của bạn là xác định xem đoạn văn bản có chứa thông tin cần thiết để trả lời câu hỏi hay không.

Đánh giá theo thang điểm từ 1-10:
1 = Hoàn toàn không liên quan
3 = Có liên quan nhẹ nhưng không đủ thông tin để trả lời
5 = Có liên quan vừa phải, có thể trả lời một phần câu hỏi
7 = Khá liên quan, có thể trả lời phần lớn câu hỏi 
10 = Rất liên quan, chứa đầy đủ thông tin để trả lời

Trả về kết quả dưới dạng JSON với các trường:
{
    "relevance_score": số_điểm_từ_1_đến_10,
    "analysis": {
        "question_focus": "trọng tâm của câu hỏi",
        "context_coverage": "phần trăm thông tin cần thiết có trong đoạn văn",
        "missing_info": "những thông tin còn thiếu"
    },
    "key_matches": ["từ khóa 1", "từ khóa 2"]
}"""

    user_prompt = f"""Đánh giá mức độ liên quan giữa:

Câu hỏi: {question}

Đoạn văn: {context[:1000]}  # Giới hạn độ dài để tránh token quá lớn

Phân tích:
1. Trọng tâm của câu hỏi là gì?
2. Đoạn văn có chứa bao nhiêu % thông tin cần thiết?
3. Những từ khóa quan trọng nào xuất hiện?
4. Thông tin nào còn thiếu?"""

    try:
        result = generate_with_ollama(system_prompt, user_prompt)
        evaluation = json.loads(result)
        
        # Đảm bảo có đầy đủ các trường
        if "relevance_score" not in evaluation:
            evaluation["relevance_score"] = 1
        if "analysis" not in evaluation:
            evaluation["analysis"] = {}
        if "key_matches" not in evaluation:
            evaluation["key_matches"] = []
            
        return evaluation
        
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá relevance: {str(e)}")
        # Trả về giá trị mặc định nếu có lỗi
        return {
            "relevance_score": 1,
            "analysis": {
                "error": str(e)
            },
            "key_matches": []
        }

def get_relevant_chunks(question: str, candidates: List[Dict], relevance_threshold: float = 7.0) -> List[Dict]:
    """
    Bước 2: Đánh giá và lọc ra các đoạn văn thực sự liên quan
    
    Args:
        question: Câu hỏi
        candidates: Danh sách đoạn văn ứng viên từ bước 1
        relevance_threshold: Ngưỡng điểm relevance để xác định đoạn thực sự liên quan
        
    Returns:
        Danh sách các đoạn văn thực sự liên quan
    """
    relevant_chunks = []
    
    for chunk in candidates:
        try:
            # Đánh giá bằng LLM
            relevance = evaluate_context_relevance(question, chunk["text"])
            relevance_score = relevance.get("relevance_score", 0)
            
            # Chỉ giữ lại các đoạn có điểm >= threshold
            if relevance_score >= relevance_threshold:
                chunk_with_score = {
                    **chunk,
                    "relevance_score": relevance_score,
                    "analysis": relevance.get("analysis", {}),
                    "key_matches": relevance.get("key_matches", [])
                }
                relevant_chunks.append(chunk_with_score)
                
        except Exception as e:
            print(f"Lỗi khi đánh giá đoạn văn: {str(e)}")
            continue
    
    # Sắp xếp theo điểm relevance
    relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    print(f"Bước 2: Tìm thấy {len(relevant_chunks)} đoạn văn thực sự liên quan")
    return relevant_chunks

def calculate_chunk_precision(candidates: List[Dict], relevant_chunks: List[Dict]) -> Dict:
    """
    Bước 3: Tính precision của quá trình retrieval
    
    Args:
        candidates: Danh sách đoạn văn ứng viên từ bước 1
        relevant_chunks: Danh sách đoạn văn thực sự liên quan từ bước 2
        
    Returns:
        Các metrics về precision
    """
    total_candidates = len(candidates)
    total_relevant = len(relevant_chunks)
    
    # Tính precision
    precision = total_relevant / total_candidates if total_candidates > 0 else 0
    
    # Tính điểm trung bình
    avg_similarity = sum(c["vector_similarity"] for c in candidates) / total_candidates if total_candidates > 0 else 0
    avg_relevance = sum(c["relevance_score"] for c in relevant_chunks) / total_relevant if total_relevant > 0 else 0
    
    return {
        "metrics": {
            "precision": round(precision, 3),
            "avg_vector_similarity": round(avg_similarity, 3),
            "avg_relevance_score": round(avg_relevance, 3)
        },
        "counts": {
            "total_candidates": total_candidates,
            "total_relevant": total_relevant
        }
    }