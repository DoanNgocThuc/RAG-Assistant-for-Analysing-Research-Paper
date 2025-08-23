# api.py
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse  
from app.rag.pipeline import process_question, ensure_index_for_pdf
from app.pdf.extract import parse_pdf
from app.rag.pipeline import generate_eval_dataset_from_pdf
from app.rag.pipeline import convert_json_to_qa_list
from app.rag.pipeline import add_contexts_to_qa_pairs
from app.rag.pipeline import create_evaluation_dataset
from app.rag.pipeline import evaluate_faithfulness
from app.rag.pipeline import create_and_evaluate_dataset
import requests
from pathlib import Path
import json
from pydantic import BaseModel
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

    

@router.post("/create_evaluation")
async def create_evaluation_endpoint(
    pdf_filename: str = Form(...),
    num_samples: int = Form(5),
    num_iterations: int = Form(2),
    k_contexts: int = Form(1),
    output_dir: str = Form("eval_outputs")
):
    """
    Create complete evaluation dataset including generated QA pairs with contexts.
    
    Args:
        pdf_filename: Name of the PDF file to evaluate
        num_samples: Number of QA pairs to generate per iteration (default: 5)
        num_iterations: Number of generation iterations (default: 2)
        k_contexts: Number of contexts to retrieve per question (default: 3)
        output_dir: Directory to save evaluation results (default: eval_outputs)
    
    Returns:
        Dictionary containing:
        - original_pairs: Generated QA pairs
        - enriched_pairs: QA pairs with retrieved contexts
        - stats: Generation and retrieval statistics
    """
    try:
        # Input validation
        if num_samples < 1 or num_samples > 20:
            raise HTTPException(
                status_code=400,
                detail="num_samples must be between 1 and 20"
            )
            
        if num_iterations < 1 or num_iterations > 5:
            raise HTTPException(
                status_code=400,
                detail="num_iterations must be between 1 and 5"
            )
            
        if k_contexts < 1 or k_contexts > 5:
            raise HTTPException(
                status_code=400,
                detail="k_contexts must be between 1 and 5"
            )

        # Validate PDF exists
        pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF file {pdf_filename} not found in uploads directory"
            )
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if evaluation file already exists
        eval_file = os.path.join(output_dir, f"{pdf_filename}.eval.json")
        if os.path.exists(eval_file):
            logger.info(f"Existing evaluation found for {pdf_filename}")
        
        logger.info(f"Starting evaluation for {pdf_filename} with {num_samples} samples x {num_iterations} iterations")
            
        # Run evaluation pipeline
        result = create_evaluation_dataset(
            pdf_path=pdf_path,
            num_samples=num_samples,
            num_iterations=num_iterations,
            k_contexts=k_contexts,
            output_dir=output_dir
        )
        
        # Add metadata to response
        response = {
            "pdf": pdf_filename,
            "parameters": {
                "num_samples": num_samples,
                "num_iterations": num_iterations,
                "k_contexts": k_contexts
            },
            "output_file": eval_file,
            "results": result
        }
        
        logger.info(f"Successfully created evaluation dataset for {pdf_filename}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unexpected error creating evaluation dataset")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
    
@router.post("/evaluate_faithfulness_from_file")
async def evaluate_faithfulness_from_file(
    filename: str = Form(...),
    output_dir: str = Form("eval_outputs")
):
    """
    Evaluate faithfulness using QA pairs from a JSON file.
    Expects file in format created by create_evaluation_dataset.
    """
    try:
        # Input validation
        input_file = os.path.join(output_dir, filename)
        if not os.path.exists(input_file):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}"
            )
            
        # Load QA pairs from file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                qa_pairs = data.get('qa_pairs', [])
                
            if not qa_pairs:
                raise HTTPException(
                    status_code=400,
                    detail="No QA pairs found in file"
                )
                
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON file: {str(e)}"
            )
            
        # Run faithfulness evaluation
        results = evaluate_faithfulness(
            qa_pairs=qa_pairs,
            output_dir=output_dir
        )
        
        return {
            "message": "Evaluation completed successfully",
            "input_file": filename,
            "total_evaluated": results["statistics"]["evaluated_pairs"],
            "average_faithfulness": results["statistics"]["average_faithfulness"],
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
    

@router.post("/create_evaluate_rag")
async def create_evaluate_rag_endpoint(
    pdf_filename: str = Form(...),
    num_samples: int = Form(5),
    num_iterations: int = Form(2),
    k_contexts: int = Form(1),
    output_dir: str = Form("eval_outputs")
):
    """
    Complete pipeline to generate QA pairs and evaluate their faithfulness.
    
    Args:
        pdf_filename: Name of the PDF file to evaluate
        num_samples: Number of QA pairs to generate per iteration (default: 5)
        num_iterations: Number of generation iterations (default: 2)
        k_contexts: Number of contexts per question (default: 3)
        output_dir: Directory to save results (default: eval_outputs)
    """
    try:
        # Input validation
        if num_samples < 1 or num_samples > 20:
            raise HTTPException(
                status_code=400,
                detail="num_samples must be between 1 and 20"
            )
            
        if num_iterations < 1 or num_iterations > 5:
            raise HTTPException(
                status_code=400,
                detail="num_iterations must be between 1 and 5"
            )
            
        if k_contexts < 1 or k_contexts > 5:
            raise HTTPException(
                status_code=400,
                detail="k_contexts must be between 1 and 5"
            )

        # Validate PDF exists
        pdf_path = os.path.join("uploads", pdf_filename)
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF file {pdf_filename} not found"
            )
            
        # Run complete pipeline
        results = create_and_evaluate_dataset(
            pdf_path=pdf_path,
            num_samples=num_samples,
            num_iterations=num_iterations,
            k_contexts=k_contexts,
            output_dir=output_dir
        )
        
        # Return summarized results
        return {
            "message": "Evaluation completed successfully",
            "pdf": results["pdf"],
            "stats": {
                "total_pairs": results["generation_stats"]["total_pairs"],
                "average_faithfulness": results["evaluation_stats"]["average_faithfulness"],
                "min_score": results["evaluation_stats"]["min_score"],
                "max_score": results["evaluation_stats"]["max_score"]
            },
            "output_files": results["files"],
            "qa_pairs": results["qa_pairs"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in evaluation pipeline")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )