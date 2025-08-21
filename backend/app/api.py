# api.py
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse  
from app.rag.pipeline import process_question, ensure_index_for_pdf
from app.pdf.extract import parse_pdf
import requests
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@router.get("/test-ollama")
async def test_ollama():
    try:
        payload = {
            "model": "mistral",
            "prompt": "Say hello from the backend!",
            "stream": False
        }
        r = requests.post(OLLAMA_API_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return {"reply": data.get("response", "").strip()}
    except Exception as e:
        return {"error": str(e)}

@router.get("/talk")
async def talk():
    try:
        payload = {
            "model": "llama3.2",
            "prompt": "Which model are you?",
            "stream": False
        }
        r = requests.post(OLLAMA_API_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return {"reply": data.get("response", "").strip()}
    except Exception as e:
        return {"error": str(e)}

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    dst = os.path.join(UPLOAD_DIR, file.filename)
    with open(dst, "wb") as f:
        f.write(await file.read())

    # Parse once and build index
    pages = parse_pdf(dst)
    ensure_index_for_pdf(dst, pages)

    return {"message": "uploaded", "filename": file.filename}

@router.get("/ask")
async def ask_question(
    question: str,
    pdf_filename: str,
    mode: str = "Novice",
    k: int = 3,
):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found on server")
    print("Trying to generate asked question...")
    answer, sources = process_question(question, mode, pdf_path, k=k)
    return JSONResponse({"answer": answer, "sources": sources})

@router.get("/context")
async def get_context(pdf_filename: str, page: int):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found on server")
    
    pages = parse_pdf(pdf_path)
    for p in pages:
        if p["page"] == page:
            return {"page": p}
    
    raise HTTPException(status_code=404, detail="Page not found")

@router.get("/get_pdf/{filename}")
async def get_pdf(filename: str):
    # Sanitize filename to prevent path traversal
    from pathlib import Path
    filename = Path(filename).name
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found on server")
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)

