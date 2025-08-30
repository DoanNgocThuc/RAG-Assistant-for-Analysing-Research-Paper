import fitz  # PyMuPDF
import re
import unicodedata

# Regex for LaTeX-like formulas
LATEX_PATTERN = re.compile(
    r"(?<!\\)\$\$(.+?)(?<!\\)\$\$"   # $$...$$
    r"|(?<!\\)\$(.+?)(?<!\\)\$"      # $...$
    r"|\\\[(.+?)\\\]",               # \[...\]
    re.S
)

MATH_OPS = r"=+\-−*/×÷^_<>≤≥≈≠∝∑∏√∫∞∂∇%|"
MATH_OPS_REGEX = re.compile(rf"[{re.escape(MATH_OPS)}]")
DIGIT_REGEX = re.compile(r"\d")
LETTER_REGEX = re.compile(r"[A-Za-z\u0370-\u03FF]")  # Latin + Greek

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2212", "-")  # Unicode minus → ASCII
    s = s.replace("\u2009", " ").replace("\u202F", " ").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_formula(line: str) -> bool:
    if not line or len(line) < 3:
        return False
    if not MATH_OPS_REGEX.search(line):
        return False
    if not (DIGIT_REGEX.search(line) and LETTER_REGEX.search(line)):
        return False
    return True

def parse_pdf(path: str):
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    pages = []

    for i, page in enumerate(doc):
        # ---------- Try rawdict extraction ----------
        lines = []
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = []
                for span in line.get("spans", []):
                    txt = span.get("text")
                    if isinstance(txt, str) and txt.strip():
                        spans.append(txt)
                if spans:
                    line_text = normalize("".join(spans))
                    if line_text:
                        lines.append(line_text)

        # ---------- Fallback if rawdict gave nothing ----------
        if not lines:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
            lines = [normalize(b[4]) for b in blocks if isinstance(b[4], str) and b[4].strip()]

        page_text = "\n".join(lines)
        print(f"Page {i+1}: {len(page_text)} characters extracted")

        # ---------- Extract formulas ----------
        formulas = []

        # LaTeX-like
        latex_hits = LATEX_PATTERN.findall(page_text)
        latex_hits = [next((g for g in t if g), None) for t in latex_hits]
        latex_hits = [normalize(f) for f in latex_hits if f]
        formulas.extend(latex_hits)

        # Unicode/inline math
        for line in lines:
            if is_formula(line):
                formulas.append(line)

        # Deduplicate
        seen, unique = set(), []
        for f in formulas:
            key = re.sub(r"\s+", "", f)
            if key not in seen:
                seen.add(key)
                unique.append(f)

        pages.append({
            "page": i + 1,
            "text": page_text,
            "formulas": unique
        })

    return pages
