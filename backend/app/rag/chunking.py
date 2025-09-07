# app/rag/chunking.py
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """
    Chunk text into overlapping windows by character length.
    Ensures forward progress to avoid infinite loops.
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

        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0
        if start >= end:
            start = end
    return chunks
