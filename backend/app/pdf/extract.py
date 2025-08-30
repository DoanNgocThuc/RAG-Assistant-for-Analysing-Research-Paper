import fitz  # PyMuPDF
import re

def parse_pdf(path: str):
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    pages = []

    # Original regex for LaTeX-like formulas (ignores escaped \$)
    formula_pattern = re.compile(
        r"(?<!\\)\$\$(.+?)(?<!\\)\$\$"  # $$...$$
        r"|(?<!\\)\$(.+?)(?<!\\)\$"     # $...$
        r"|\\\[(.+?)\\\]",              # \[...\]
        re.S
    )

    # Regex for Unicode math characters
    unicode_math_char_pattern = re.compile(
        r"[\u0370-\u03FF\u2100-\u214F\u2190-\u21FF\u2200-\u22FF\u2300-\u23FF\u25A0-\u25FF"
        r"\u27C0-\u27EF\u27F0-\u27FF\u2900-\u297F\u2980-\u29FF\u2A00-\u2AFF\u2B00-\u2BFF"
        r"\u1D400-\u1D7FF\u1EE00-\u1EEFF\u20D0-\u20FF]",
        re.UNICODE
    )

    for i, page in enumerate(doc):
        # Use "blocks" mode for better text fidelity
        blocks = page.get_text("blocks")

        # Sort blocks by vertical, then horizontal position
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

        # Join block text with proper line breaks
        text = "\n".join(b[4] for b in blocks if b[4].strip())
        print(f"Page {i+1}: {len(text)} characters extracted")

        # Extract LaTeX formulas (original)
        latex_formulas = formula_pattern.findall(text)

        # Flatten tuple results (filter out empty matches)
        latex_formulas = [next((g for g in t if g), None) for t in latex_formulas]
        latex_formulas = [f for f in latex_formulas if f]  # remove None/empty

        # New: Extract Unicode math formulas by processing lines
        lines = text.split("\n")
        unicode_formulas = []
        current_formula = []
        for line in lines:
            line_strip = line.strip()
            if line_strip and unicode_math_char_pattern.search(line_strip) and len(line_strip) < 50:
                current_formula.append(line_strip)
            else:
                if current_formula:
                    unicode_formulas.append(" ".join(current_formula))
                    current_formula = []
        if current_formula:
            unicode_formulas.append(" ".join(current_formula))

        # Combine both types of formulas, deduplicate if needed
        all_formulas = latex_formulas + unicode_formulas
        all_formulas = list(set(all_formulas))  # Optional: remove duplicates if overlap occurs

        pages.append({
            "page": i + 1,
            "text": text,
            "formulas": all_formulas
        })

    return pages