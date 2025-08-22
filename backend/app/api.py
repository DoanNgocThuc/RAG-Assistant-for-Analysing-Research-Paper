# api.py
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse  
from app.rag.pipeline import process_question, ensure_index_for_pdf
from app.pdf.extract import parse_pdf
from app.rag.pipeline import generate_eval_dataset_from_pdf
import requests
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
EMBEDDINGS_DIR = os.path.join(os.getcwd(), "embeddings")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

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

@router.post("/generate_eval")
async def generate_eval_dataset_api(
    pdf_filename: str = Form(...),
    num_samples: int = Form(5),
    output_dir: str = Form("eval_outputs"),
):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found on server")

    # Call the pipeline function to generate evaluation dataset
    result = generate_eval_dataset_from_pdf(pdf_path, num_samples=num_samples, output_dir=output_dir)
    return {"message": "Evaluation dataset generated successfully", "result": result}

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

@router.delete("/delete_pdf/{filename}")
async def delete_pdf(filename: str):
    # Sanitize filename to prevent path traversal
    filename = Path(filename).name
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{filename}.faiss")
    meta_path = os.path.join(EMBEDDINGS_DIR, f"{filename}.pkl")

    deleted = False
    response = {"message": "Deletion completed", "details": []}

    # Delete PDF file
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            response["details"].append(f"PDF file {filename} deleted")
            deleted = True
        except Exception as e:
            response["details"].append(f"Failed to delete PDF file {filename}: {str(e)}")
    else:
        response["details"].append(f"PDF file {filename} not found")

    # Delete FAISS index
    if os.path.exists(idx_path):
        try:
            os.remove(idx_path)
            response["details"].append(f"FAISS index for {filename} deleted")
            deleted = True
        except Exception as e:
            response["details"].append(f"Failed to delete FAISS index for {filename}: {str(e)}")
    else:
        response["details"].append(f"FAISS index for {filename} not found")

    # Delete metadata
    if os.path.exists(meta_path):
        try:
            os.remove(meta_path)
            response["details"].append(f"Metadata file for {filename} deleted")
            deleted = True
        except Exception as e:
            response["details"].append(f"Failed to delete metadata file for {filename}: {str(e)}")
    else:
        response["details"].append(f"Metadata file for {filename} not found")

    if not deleted:
        response["message"] = "No files were deleted"
    
    return response