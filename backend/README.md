1. Create & activate virtualenv:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

2. Install:

```bash
pip install -r requirements.txt
```

3. Set OpenAI key (optional but recommended):

```bash
export OPENAI_API_KEY="sk-..."
```

4. Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

5. Endpoints:
- POST /api/upload (multipart form "file")
- POST /api/ask (form fields: question, mode, pdf_path, k)
- GET /api/context?pdf_path=...&page=1


Notes:
- The backend caches FAISS indexes under `embeddings/` after upload.
- This is an MVP: improve chunking, error handling, and formula parsing for production.
"""
