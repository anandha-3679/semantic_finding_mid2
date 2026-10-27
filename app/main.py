import uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from . import parser, rag

app = FastAPI()
DOCS = {}  # doc_id -> {"chunks": [...], "matrix": np.ndarray}  (in-memory)

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        paras = parser.extract(file.filename or "", await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    chunks = rag.chunk(paras)
    if not chunks:
        raise HTTPException(400, "No readable text found in the file.")
    matrix = rag.embed([c["text"] for c in chunks])
    doc_id = uuid.uuid4().hex
    DOCS[doc_id] = {"chunks": chunks, "matrix": matrix}
    print("chunks:", len(chunks), "| vec diff:", float(abs(matrix[0] - matrix[-1]).sum()))
    return {"doc_id": doc_id, "chunks": chunks}

@app.post("/api/search")
async def search(doc_id: str = Form(...), query: str = Form(...), top_k: int = Form(5)):
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(404, "Upload a document first.")
    hits = rag.search(doc["matrix"], query, top_k)
    results = []
    for rank, (i, score) in enumerate(hits):
        c = doc["chunks"][i]
        results.append({
            "id": c["id"], "page": c["page"], "text": c["text"], "score": round(score, 3),
            "why": rag.explain(query, c["text"]) if rank == 0 else "",
        })
    return {"results": results}

app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[2] / "frontend", html=True))