# api.py
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse  
from app.rag.pipeline import process_question, ensure_index_for_pdf
from app.pdf.extract import parse_pdf
from app.rag.pipeline import generate_eval_dataset_from_pdf
from app.rag.pipeline import convert_json_to_qa_list
import requests
from pathlib import Path
import json

import logging
# Add at top of file with other imports
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    num_samples: int = Form(...),
    num_iterations: int = Form(...),
    output_dir: str = Form("eval_outputs"),
):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found on server")

    # Call the pipeline function to generate evaluation dataset
    result = generate_eval_dataset_from_pdf(pdf_path, num_samples=num_samples, num_iterations=num_iterations, output_dir=output_dir)
    return {"message": "Evaluation dataset generated successfully", "result": result}


@router.get("/qac_list/{pdf_filename}")
async def get_qac_list(pdf_filename: str):
    try:
        # Sanitize filename to prevent path traversal
        filename = Path(pdf_filename).name
        
        # Construct path to eval json file
        eval_file = os.path.join("eval_outputs", f"{filename}.eval.json")
        logger.debug(f"Looking for eval file at: {eval_file}")
        
        if not os.path.exists(eval_file):
            logger.error(f"Eval file not found: {eval_file}")
            raise HTTPException(
                status_code=404, 
                detail=f"Evaluation data not found for PDF: {filename}"
            )
        
        # Use the convert function to get QA list
        try:
            qa_list = convert_json_to_qa_list(eval_file)
            logger.debug(f"Successfully loaded QA list with {len(qa_list)} items")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error parsing evaluation data JSON"
            )
        
        # Convert tuples to dictionaries for JSON response
        formatted_list = [
            {
                "question": q,
                "answer": a,
                "context": c
            }
            for q, a, c in qa_list
        ]
        
        return {"qa_pairs": formatted_list}
        
    except Exception as e:
        logger.exception("Unexpected error in get_qac_list")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing evaluation data: {str(e)}"
        )



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

