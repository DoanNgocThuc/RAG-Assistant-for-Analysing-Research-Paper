import os
import pickle
import numpy as np
from app.rag.indexing import ensure_index_for_pdf
from app.rag.generation import generate_with_ollama
import faiss

EMBEDDINGS_DIR = os.path.join(os.getcwd(), "embeddings")

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