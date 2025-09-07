# app/rag/embeddings.py
import os, time, requests
import numpy as np

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"  # make sure you have ollama pull nomic-embed-text

def embed_with_ollama(texts, retries: int = 3, timeout: int = 120) -> np.ndarray:
    print(f"Embedding {len(texts)} texts with Ollama...")
    if not isinstance(texts, list):
        texts = [texts]
    if not texts:
        raise RuntimeError("No texts provided for embedding")

    expected_dim = 768
    embeddings = []

    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)} (len={len(text)})")
        embedding, attempt = None, 0
        while attempt < retries:
            try:
                r = requests.post(
                    OLLAMA_EMBED_ENDPOINT,
                    json={"model": EMBED_MODEL_NAME, "prompt": text},
                    timeout=timeout
                )
                r.raise_for_status()
                data = r.json()
                embedding = data.get("embedding") or (data.get("embeddings") or [None])[0]
                break
            except Exception as e:
                print(f"Embedding attempt {attempt+1} failed: {e}")
                attempt += 1
                if attempt == retries:
                    embedding = [0.0] * expected_dim
                time.sleep(1)

        if not embedding or len(embedding) != expected_dim:
            embeddings.append([0.0] * expected_dim)
        else:
            embeddings.append(embedding)

    arr = np.array(embeddings, dtype="float32")
    return arr
