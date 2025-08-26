import fitz  # PyMuPDF
import re

def parse_pdf(path: str):
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    pages = []

    # Regex for LaTeX-like formulas (ignores escaped \$)
    formula_pattern = re.compile(
        r"(?<!\\)\$\$(.+?)(?<!\\)\$\$"  # $$...$$
        r"|(?<!\\)\$(.+?)(?<!\\)\$"     # $...$
        r"|\\\[(.+?)\\\]",              # \[...\]
        re.S
    )

    for i, page in enumerate(doc):
        # Use "blocks" mode for better text fidelity
        blocks = page.get_text("blocks")

        # Sort blocks by vertical, then horizontal position
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

        # Join block text with proper line breaks
        text = "\n".join(b[4] for b in blocks if b[4].strip())
        print(f"Page {i+1}: {len(text)} characters extracted")

        # Extract formulas
        formulas = formula_pattern.findall(text)

        # Flatten tuple results (filter out empty matches)
        formulas = [next((g for g in t if g), None) for t in formulas]
        formulas = [f for f in formulas if f]  # remove None/empty

        pages.append({
            "page": i + 1,
            "text": text,
            "formulas": formulas
        })

    return pages
