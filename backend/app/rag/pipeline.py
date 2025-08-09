import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai

# Set OPENAI_API_KEY env var for LLM calls
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = os.path.join(os.getcwd(), "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Helpers for chunking

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50):
    # naive chunking by characters (good enough for MVP). Adjust for tokens if needed.
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_tokens, length)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


def ensure_index_for_pdf(pdf_path: str, pages: List[dict] = None):
    """Builds and caches a FAISS index + metadata for the given PDF path.
    If already exists, skip.
    Returns index path.
    """
    key = os.path.basename(pdf_path)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{key}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        return idx_path
    # otherwise build
    from app.pdf.extract import parse_pdf
    if pages is None:
        pages = parse_pdf(pdf_path)
    texts = []
    metadatas = []
    for p in pages:
        page_num = p["page"]
        page_text = p["text"].strip()
        chunks = chunk_text(page_text, max_tokens=1200, overlap=200)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"page": page_num, "chunk_id": i})
    # compute embeddings
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    return idx_path


def _load_index(pdf_path: str):
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
    index, meta = _load_index(pdf_path)
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        text = meta["texts"][idx]
        md = meta["metadatas"][idx]
        results.append({"text": text, "metadata": md})
    return results


def _build_system_prompt(mode: str):
    if mode.lower() == "novice":
        return (
            "You are an assistant that explains scientific papers to a novice. "
            "Give a clear, concise answer with high-level intuition and point to the source pages. "
            "Keep technical jargon minimal and define any introduced terms."
        )
    elif mode.lower() == "reviewer":
        return (
            "You are an assistant that helps peer reviewers. Provide critical analysis, list assumptions, "
            "possible threats to validity, and point to specific pages/paragraphs supporting claims."
        )
    else:
        return (
            "You are an assistant that explains scientific papers to an informed researcher. Provide precise, "
            "technical answers, include equations when relevant, and cite page numbers.")


def call_openai_chat(system_prompt: str, question: str, contexts: List[dict]):
    if not OPENAI_API_KEY:
        # fallback: simple concatenation answer (not ideal). Warn user.
        ctx_text = "\n\n".join([f"[page {c['metadata']['page']}] " + c['text'] for c in contexts])
        return (
            "[NO OPENAI_API_KEY set] -> returning concatenated context as answer. "
            "Set OPENAI_API_KEY to enable LLM generation.",
            contexts,
        )
    # construct prompt
    system = system_prompt
    # Provide top-k contexts with page numbers
    context_blocks = []
    for c in contexts:
        page = c['metadata']['page']
        snippet = c['text'][:1000]
        context_blocks.append(f"[page {page}] {snippet}")
    user_prompt = (
        "You are given the following snippets from a target paper (each labeled by page). "
        "Answer the question using ONLY the provided snippets and cite page numbers inline. "
        f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
    )
    # call OpenAI ChatCompletion
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    out = resp["choices"][0]["message"]["content"].strip()
    return out


def process_question(question: str, mode: str, pdf_path: str, k: int = 3):
    contexts = retrieve_top_k(question, pdf_path, k=k)
    system_prompt = _build_system_prompt(mode)
    answer = call_openai_chat(system_prompt, question, contexts)
    # build sources for frontend: short snippet + page
    sources = []
    for c in contexts:
        sources.append({
            "page": c['metadata']['page'],
            "snippet": c['text'][:800],
        })
    return answer, sources