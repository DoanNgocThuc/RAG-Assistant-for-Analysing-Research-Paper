import fitz  # pymupdf
import re

# Very small PDF parser that returns text per page and detects LaTeX-like formulas

def parse_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        # simple latex formula capture: $...$ or $$...$$ or \[ ... \]
        formulas = re.findall(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]", text, flags=re.S)
        # flatten formula tuples
        formulas = [next((g for g in t if g), "") for t in formulas]
        pages.append({"page": i + 1, "text": text, "formulas": formulas})
    return pages