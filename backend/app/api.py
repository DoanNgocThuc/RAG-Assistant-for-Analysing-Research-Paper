# api.py
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException,Query
from fastapi.responses import JSONResponse, FileResponse  
from app.rag.pipeline import (
    process_question, 
    ensure_index_for_pdf,
    convert_json_to_qa_list,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    get_candidate_chunks,
    get_relevant_chunks,
    calculate_chunk_precision
)
from app.rag.pipeline import load_rag_benchmark_dataset
from app.pdf.extract import parse_pdf
import requests
from pathlib import Path
from typing import List, Optional

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

        
@router.post("/evaluate_faithfulness_from_file")
async def evaluate_faithfulness_from_file(
    filename: str = Form(...),
    output_dir: str = Form("benchmark_datasets")
):
    """
    Evaluate faithfulness using QA pairs from a JSON file.
    Uses the new evaluate_faithfulness(benchmark_file, output_dir) function.
    """
    try:
        # Input validation
        input_file = os.path.join(output_dir, filename)
        if not os.path.exists(input_file):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}"
            )

        # Run faithfulness evaluation using the new function
        try:
            results = evaluate_faithfulness(
                benchmark_file=input_file,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Error during faithfulness evaluation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during faithfulness evaluation: {str(e)}"
            )

        return {
            "message": "Evaluation completed successfully",
            "input_file": filename,
            "total_evaluated": results["statistics"]["evaluated_pairs"],
            "average_faithfulness": results["statistics"]["average_faithfulness"],
            "min_score": results["statistics"]["min_score"],
            "max_score": results["statistics"]["max_score"],
            "detailed_results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during faithfulness evaluation")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.post("/evaluate_relevance_from_file")
async def evaluate_relevance_from_file(
    filename: str = Form(...),
    output_dir: str = Form("benchmark_datasets")
):
    """
    Evaluate answer relevance using QA pairs from a JSON file.
    Uses the evaluate_answer_relevance(benchmark_file, output_dir) function.
    """
    try:
        # Input validation
        input_file = os.path.join(output_dir, filename)
        if not os.path.exists(input_file):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}"
            )

        # Run relevance evaluation
        try:
            results = evaluate_answer_relevance(
                benchmark_file=input_file,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Error during relevance evaluation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during relevance evaluation: {str(e)}"
            )

        return {
            "message": "Evaluation completed successfully",
            "input_file": filename,
            "total_evaluated": results["statistics"]["evaluated_pairs"],
            "average_relevance": results["statistics"]["average_relevance"],
            "min_score": results["statistics"]["min_score"],
            "max_score": results["statistics"]["max_score"],
            "detailed_results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during relevance evaluation")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
    
@router.post("/benchmark_dataset")
async def get_benchmark_dataset(
    subset: Optional[str] = "train",  # Default to train split
    num_samples: Optional[int] = 10,  # Default to 1 record
    save_to_file: bool = True  # Default to save file
):
    try:
        result = load_rag_benchmark_dataset(subset, num_samples, save_to_file)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))