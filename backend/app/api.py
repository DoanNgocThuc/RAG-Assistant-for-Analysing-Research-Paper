import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.rag.pipeline import process_question, ensure_index_for_pdf
from app.pdf.extract import parse_pdf

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    dst = os.path.join(UPLOAD_DIR, file.filename)
    with open(dst, "wb") as f:
        f.write(await file.read())
    # parse once and build index
    pages = parse_pdf(dst)
    ensure_index_for_pdf(dst, pages)
    return {"message": "uploaded", "pdf_path": dst}

@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    mode: str = Form("Novice"),
    pdf_path: str = Form(...),
    k: int = Form(3),
):
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="pdf_path not found on server")
    answer, sources = process_question(question, mode, pdf_path, k=k)
    return JSONResponse({"answer": answer, "sources": sources})

@router.get("/context")
async def get_context(pdf_path: str, page: int):
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="pdf_path not found on server")
    pages = parse_pdf(pdf_path)
    for p in pages:
        if p["page"] == page:
            return {"page": p}
    raise HTTPException(status_code=404, detail="page not found")