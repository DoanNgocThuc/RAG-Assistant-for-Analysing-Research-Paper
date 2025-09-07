# app/rag/indexing.py
import os, pickle, faiss
from app.pdf.extract import parse_pdf
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_with_ollama
import numpy as np

EMBEDDINGS_DIR = os.path.join(os.getcwd(), "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def ensure_index_for_pdf(pdf_path: str, pages: list = None):
    print("Ensuring FAISS index for PDF...")
    key = os.path.basename(pdf_path)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{key}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        return idx_path

    if pages is None:
        pages = parse_pdf(pdf_path)

    texts, metadatas = [], []
    for p in pages:
        chunks = chunk_text(p.get("text", ""), max_chars=1200, overlap=200)
        for i, c in enumerate(chunks):
            if not c.strip(): continue
            texts.append(c)
            metadatas.append({"page": p["page"], "chunk_id": i})

    embeddings = embed_with_ollama(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
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
    q_emb = embed_with_ollama([question]).astype("float32")
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        results.append({"text": meta["texts"][idx], "metadata": meta["metadatas"][idx]})
    return results
