# app/rag/pipeline.py
import os
from app.pdf.extract import parse_pdf
from app.rag.indexing import retrieve_top_k
from app.rag.prompts import _build_system_prompt, explain_context
from app.rag.generation import generate_with_ollama

def process_question(question: str, mode: str, pdf_path: str, k: int = 3):
    contexts = retrieve_top_k(question, pdf_path, k=k)
    system_prompt = _build_system_prompt(mode)
    context_blocks, sources = [], []

    for c in contexts:
        page, snippet = c["metadata"]["page"], c["text"][:1000].strip()
        context_blocks.append(f"[page {page}] {snippet}")
        explanation = explain_context(question, snippet)
        sources.append({"page": page, "snippet": snippet, "explanation": explanation})

    user_prompt = (
        "You are given the following snippets...\n\n"
        f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
    )
    answer_text = generate_with_ollama(system_prompt, user_prompt, max_tokens=800)
    return answer_text, sources

def get_page_text(pdf_path: str, page_number: int):
    pages = parse_pdf(pdf_path)
    for p in pages:
        if p["page"] == page_number:
            return p
    return None

def get_formulas(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = parse_pdf(pdf_path)
    formulas = []
    for p in pages:
        for f in p.get("formulas", []):
            if f and isinstance(f, str):
                formulas.append({"page": p["page"], "formula": f.strip()})
    return formulas
