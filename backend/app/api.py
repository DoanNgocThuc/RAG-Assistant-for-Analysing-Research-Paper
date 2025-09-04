# api.py
import json
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.rag.pipeline import evaluate_RAG, process_question, ensure_index_for_pdf, get_formulas, suggest_related_papers_with_difference
from app.pdf.extract import parse_pdf
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
    mode: str,
    k: int = 5,
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

@router.get("/formulas")
def formulas(pdf_filename: str):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)  # Adjust path
    try:
        return {"formulas": get_formulas(pdf_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/rag-evaluation")
def rag_evaluation(pdf_filename: str):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)  # Adjust path

    # get question_groundtruth list from question_groundtruth.json
    with open("eval_outputs/question_groundtruth.json", "r", encoding="utf-8") as f:
        question_groundtruth = json.load(f)
    
    data = evaluate_RAG(question_groundtruth, pdf_path)
    faithfulness = data["faithfulness"]
    context_recall = data["context_recall"]
    context_precision = data["context_precision"]
    answer_correctness = data["answer_correctness"]
    answer_relevancy = data["answer_relevancy"]

    # Tính trung bình
    avg_faithfulness = sum(faithfulness) / len(faithfulness)
    avg_recall = sum(context_recall) / len(context_recall)
    avg_precision = sum(context_precision) / len(context_precision)
    avg_answer_correctness = sum(answer_correctness) / len(answer_correctness)
    avg_answer_relevancy = sum(answer_relevancy) / len(answer_relevancy)

    result = {
        "faithfulness": avg_faithfulness,
        "context_recall": avg_recall,
        "context_precision": avg_precision,
        "answer_correctness": avg_answer_correctness,
        "answer_relevancy": avg_answer_relevancy
    }

    # Lưu kết quả vào file
    os.makedirs("evaluation_scores", exist_ok=True)
    with open("evaluation_scores/rag_evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    try:
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/related_papers")
def related_papers(pdf_filename: str):
    print("looking for: ", pdf_filename)
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)  # Adjust path
    try:
        return {"related_papers": suggest_related_papers_with_difference(pdf_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))