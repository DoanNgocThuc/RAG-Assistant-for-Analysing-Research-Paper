# app/rag/generation.py
import os, requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_GEN_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
LLM_MODEL_NAME = "llama3.2"

def generate_with_ollama(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    print("Generating text with Ollama...")
    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    r = requests.post(OLLAMA_GEN_ENDPOINT, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        if "response" in data: return data["response"].strip()
        if "text" in data: return data["text"].strip()
        if "outputs" in data and len(data["outputs"]) > 0:
            out0 = data["outputs"][0]
            if "content" in out0: return out0["content"].strip()
            if "message" in out0 and "content" in out0["message"]:
                return out0["message"]["content"].strip()
    raise RuntimeError(f"Unexpected Ollama response: {data}")
