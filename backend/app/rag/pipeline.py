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
from datasets import load_dataset
import datetime

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
    Use the LLM to generate evaluation questions and answers multiple times and append results.
    The 'source' field will be left empty for later context insertion.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path)}.eval.json")

        # Parse PDF once
        pages = parse_pdf(pdf_path)
        content_blocks = [
            f"[page {p['page']}] {p['text'][:1000].strip()}" for p in pages if p['text'].strip()
        ]
        content = "\n".join(content_blocks)

        existing_questions = set()
        all_results = []
        prompt_history = []  # Store prompts and responses

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
                                all_results.extend(existing_qa)
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

                system_prompt = (
                    "You are both scientific assistant and novice. Generate different evaluation questions.\n"
                    "IMPORTANT: Return ONLY valid JSON array with question-answer pairs.\n"
                    "Do not include any other text before or after the JSON array."
                )

                user_prompt = f"""
Based on the following content, generate {num_samples} DIFFERENT pairs of question and concise answer.

Requirements:
- Questions must be different from existing ones
- Answers must be concise and relevant
- For the 'source' field, leave it empty (""), it will be filled later
- Focus on important concepts from the paper
- Return ONLY the JSON array in this exact format:
[
  {{
    "question": "What/How/Why/Who/When ...?",
    "answer": "According to ...",
    "source": ""
  }},
  ...
]

Content:
{content}
"""
                # Generate response
                result = generate_with_ollama(system_prompt, user_prompt)
                logger.debug(f"Raw LLM response: {result[:200]}...")

                # Store prompt and response
                prompt_record = {
                    "iteration": i + 1,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": result
                }
                prompt_history.append(prompt_record)

                result = result.strip()
                if not result.startswith('['):
                    start_idx = result.find('[')
                    if start_idx == -1:
                        logger.error(f"No JSON array found in response: {result[:100]}...")
                        continue
                    result = result[start_idx:]

                if not result.endswith(']'):
                    end_idx = result.rfind(']')
                    if end_idx == -1:
                        logger.error(f"No closing bracket found in response: {result[-100:]}")
                        continue
                    result = result[:end_idx + 1]

                new_pairs = json.loads(result)
                if not isinstance(new_pairs, list):
                    logger.error(f"Expected JSON array, got: {type(new_pairs)}")
                    continue

                valid_pairs = []
                for pair in new_pairs:
                    if not isinstance(pair, dict):
                        continue
                    if not all(k in pair for k in ('question', 'answer', 'source')):
                        continue
                    if not all(isinstance(pair[k], str) for k in ('question', 'answer', 'source')):
                        continue
                    # Only accept pairs with empty source
                    if pair['source'] != "":
                        continue
                    valid_pairs.append(pair)

                unique_pairs = [
                    pair for pair in valid_pairs
                    if pair['question'] not in existing_questions
                ]

                existing_questions.update(pair['question'] for pair in unique_pairs)
                all_results.extend(unique_pairs)

                logger.info(f"Added {len(unique_pairs)} new unique QA pairs from {len(valid_pairs)} valid pairs")

                if i < num_iterations - 1:
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                continue

        final_result = (
            "Here is an evaluation dataset for a document QA system based on the provided content:\n\n"
            "```json\n" +
            json.dumps(all_results, indent=2, ensure_ascii=False) +
            "\n```"
        )

        # Save everything to output file
        output_data = {
            "pdf": os.path.basename(pdf_path),
            "metadata": {
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": num_samples,
                "num_iterations": num_iterations,
                "total_qa_pairs": len(all_results)
            },
            "prompt_history": prompt_history,
            "eval_data": final_result
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved total of {len(all_results)} QA pairs and {len(prompt_history)} prompts to {output_file}")
        return final_result

    except Exception as e:
        logger.error(f"Error generating evaluation dataset: {str(e)}")
        raise


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

def add_contexts_to_qa_pairs(qa_pairs: list, pdf_path: str, k: int = 3) -> list:
    """
    Add retrieved contexts to QA pairs using retrieve_top_k function.
    
    Args:
        qa_pairs: List of dicts with 'question', 'answer', and empty 'context'
        pdf_path: Path to the PDF file
        k: Number of top contexts to retrieve
        
    Returns:
        List of QA pairs with added contexts
    """
    logger.info(f"Adding contexts to {len(qa_pairs)} QA pairs...")
    
    enriched_pairs = []
    
    for pair in qa_pairs:
        try:
            # Get top-k contexts for the question
            contexts = retrieve_top_k(pair['question'], pdf_path, k)
            
            # Format contexts into a single string
            context_texts = []
            for ctx in contexts:
                page = ctx['metadata']['page']
                text = ctx['text'][:1000].strip()
                context_texts.append(f"[page {page}] {text}")
            
            # Create enriched QA pair
            enriched_pair = {
                'question': pair['question'],
                'answer': pair['answer'],
                'context': '\n\n'.join(context_texts)
            }
            
            enriched_pairs.append(enriched_pair)
            logger.debug(f"Added {len(contexts)} contexts for question: {pair['question'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding contexts for question '{pair['question'][:50]}...': {str(e)}")
            # Keep original pair without context if retrieval fails
            enriched_pairs.append(pair)
            
    logger.info(f"Successfully added contexts to {len(enriched_pairs)} QA pairs")
    return enriched_pairs

def create_evaluation_dataset(
    pdf_path: str,
    num_samples: int = 5,
    num_iterations: int = 2,
    k_contexts: int = 3,
    output_dir: str = "eval_outputs"
) -> dict:
    """
    Complete pipeline to create evaluation dataset:
    1. Generate QA pairs from PDF
    2. Convert to list format
    3. Add contexts to each pair
    
    Args:
        pdf_path: Path to PDF file
        num_samples: Number of QA pairs to generate per iteration
        num_iterations: Number of generation iterations
        k_contexts: Number of contexts to retrieve per question
        output_dir: Directory to save evaluation data
        
    Returns:
        dict with evaluation results containing:
        - original_pairs: List of generated QA pairs
        - enriched_pairs: List of QA pairs with contexts
        - stats: Generation statistics
    """
    logger.info(f"Starting evaluation dataset creation for {pdf_path}")
    
    try:
        # 1. Generate initial QA pairs
        logger.info("Step 1: Generating QA pairs...")
        result = generate_eval_dataset_from_pdf(
            pdf_path=pdf_path,
            num_samples=num_samples,
            num_iterations=num_iterations,
            output_dir=output_dir
        )
        
        # 2. Convert to list format
        logger.info("Step 2: Converting to QA list format...")
        output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path)}.eval.json")
        qa_pairs = convert_json_to_qa_list(output_file)
        
        # Convert tuples to dicts for context addition
        qa_dicts = [
            {
                "question": q,
                "answer": a,
                "context": ""  # Empty context to be filled
            }
            for q, a, _ in qa_pairs
        ]
        
        # 3. Add contexts to each pair
        logger.info("Step 3: Adding contexts to QA pairs...")
        enriched_pairs = add_contexts_to_qa_pairs(
            qa_pairs=qa_dicts,
            pdf_path=pdf_path,
            k=k_contexts
        )
        
        # Prepare return data
        stats = {
            "total_pairs": len(enriched_pairs),
            "successful_contexts": len([p for p in enriched_pairs if p.get('context')])
        }
        
        logger.info(f"Completed evaluation dataset creation with {stats['total_pairs']} pairs")
        
        # Save enriched dataset to a new file
        enriched_file = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(pdf_path))[0]}.enriched.json"
        )
        
        enriched_data = {
            "qa_pairs": enriched_pairs
        }
        
        with open(enriched_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved enriched dataset to {enriched_file}")
        
        # Return complete results
        return {
            "pdf": os.path.basename(pdf_path),
            "original_pairs": qa_dicts,
            "enriched_pairs": enriched_pairs,
            "stats": stats,
            "enriched_file": enriched_file
        }
        
    except Exception as e:
        logger.error(f"Error creating evaluation dataset: {str(e)}")
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
    output_dir: str = "eval_outputs"
) -> dict:
    """
    Evaluate faithfulness of QA pairs loaded from a benchmark JSON file.

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

    # Only keep pairs with question, answer, and context (if available)
    filtered_pairs = []
    for pair in qa_pairs:
        if all(k in pair for k in ("question", "answer")):
            # If context missing, set to empty string
            filtered_pairs.append({
                "question": pair["question"],
                "answer": pair["answer"],
                "context": pair.get("context", "")
            })
    qa_pairs = filtered_pairs

    # Faithfulness evaluation logic (unchanged)
    system_prompt = """You are an expert evaluator for question answering systems.
Your task is to evaluate the faithfulness of answers - how well they align with the provided contexts.

Score faithfulness using these exact groups:
1 = Answer completely contradicts or fabricates information not in context
2 = Answer almost entirely unsupported, with only minimal alignment
3 = Answer mostly unsupported, with some minor alignment
4 = Answer partially reflects context but includes several unsupported claims
5 = Answer partially supported, but with notable unsupported additions
6 = Answer mostly supported by context, but with some unsupported additions
7 = Answer well supported, only minor unsupported details
8 = Answer strongly supported, almost no unsupported information
9 = Answer perfectly reflects context, with no unsupported additions
10 = Answer is a flawless, complete reflection of context, with zero unsupported or contradictory information

IMPORTANT: Return ONLY valid JSON with scores and explanations.
Do not include any other text before or after the JSON."""

    all_evaluations = []

    for i, pair in enumerate(qa_pairs, 1):
        try:
            user_prompt = f"""
Evaluate the faithfulness of this QA pair:

Question: {pair['question']}
Context: {pair['context']}
Answer: {pair['answer']}

Analyze:
1. Whether all claims in the answer are supported by the context
2. Whether the answer adds any unsupported information
3. Whether the answer contradicts any context information

Return your evaluation in this exact JSON format:
{{
    "faithfulness_score": score_between_1_and_100,
    "explanation": "Detailed explanation of the score",
    "supported_claims": ["List specific claims that are supported"],
    "unsupported_claims": ["List claims not found in context"],
    "contradictions": ["List any contradictions with context"]
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

            logger.info(f"Evaluated pair {i}/{len(qa_pairs)}: score = {evaluation['faithfulness_score']}")

            # Add delay between evaluations
            if i < len(qa_pairs):
                time.sleep(2)

        except Exception as e:
            logger.error(f"Error evaluating pair {i}: {str(e)}")
            continue

    # Calculate statistics
    scores = [e['faithfulness_score'] for e in all_evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0

    evaluation_results = {
        "evaluations": all_evaluations,
        "statistics": {
            "total_pairs": len(qa_pairs),
            "evaluated_pairs": len(all_evaluations),
            "average_faithfulness": round(avg_score, 2),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"faithfulness_evaluation_{int(time.time())}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation results to {output_file}")
    logger.info(f"Average faithfulness score: {avg_score:.2f}")

    return evaluation_results


def create_and_evaluate_dataset(
    pdf_path: str,
    num_samples: int = 5,
    num_iterations: int = 2,
    k_contexts: int = 3,
    output_dir: str = "eval_outputs"
) -> dict:
    """
    Complete pipeline to:
    1. Create evaluation dataset with contexts
    2. Evaluate faithfulness of the QA pairs
    
    Args:
        pdf_path: Path to PDF file
        num_samples: Number of QA pairs to generate per iteration
        num_iterations: Number of generation iterations
        k_contexts: Number of contexts to retrieve per question
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing both dataset creation and evaluation results
    """
    logger.info(f"Starting complete evaluation pipeline for {pdf_path}")
    
    try:
        # Step 1: Create evaluation dataset
        logger.info("Step 1: Creating evaluation dataset...")
        dataset_results = create_evaluation_dataset(
            pdf_path=pdf_path,
            num_samples=num_samples,
            num_iterations=num_iterations,
            k_contexts=k_contexts,
            output_dir=output_dir
        )
        
        # Step 2: Evaluate faithfulness
        logger.info("Step 2: Evaluating faithfulness...")
        evaluation_results = evaluate_faithfulness(
            qa_pairs=dataset_results["enriched_pairs"],
            output_dir=output_dir
        )
        
        # Combine results
        final_results = {
            "pdf": dataset_results["pdf"],
            "generation_stats": dataset_results["stats"],
            "evaluation_stats": evaluation_results["statistics"],
            "qa_pairs": [
                {
                    **pair,
                    "evaluation": next(
                        (e for e in evaluation_results["evaluations"] if e["question"] == pair["question"]),
                        None
                    )
                }
                for pair in dataset_results["enriched_pairs"]
            ],
            "files": {
                "enriched_dataset": dataset_results["enriched_file"],
                "evaluation_results": evaluation_results.get("output_file")
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save final results
        final_output = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(pdf_path))[0]}_complete_evaluation.json"
        )
        
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved complete evaluation results to {final_output}")
        return final_results
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise

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
