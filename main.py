"""
RAG Legal Document Generator MVP
Main entry point for the application
"""

import os
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import API_HOST, API_PORT, API_TITLE, API_VERSION, JSON_DIR
from vector_database import VectorDatabase
from simple_vector_db import SimpleVectorDatabase
from gemini_integration import generate_legal_document

# Initialize FastAPI app
app = FastAPI(
	title=API_TITLE,
	version=API_VERSION,
	description="MVP for generating legal documents using RAG architecture",
)

# CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Static & templates
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Try to init full vector DB on startup; fallback to simple
vector_backend = {"type": "", "db": None}

@app.on_event("startup")
async def startup_event():
	global vector_backend
	try:
		vdb = VectorDatabase()
		vdb.load_from_json_files(JSON_DIR)
		vector_backend = {"type": "chroma", "db": vdb}
	except Exception:
		svdb = SimpleVectorDatabase()
		svdb.load_from_json_files(JSON_DIR)
		vector_backend = {"type": "tfidf", "db": svdb}

@app.get("/", response_class=HTMLResponse)
async def root_page(request: Request):
	return templates.TemplateResponse(
		"index.html",
		{"request": request, "api_version": API_VERSION, "backend": vector_backend["type"]},
	)

@app.get("/health")
async def health_check():
	return {
		"status": "healthy",
		"components": {
			"api": "running",
			"vector_backend": vector_backend["type"] or "none",
		},
	}

@app.post("/api/generate")
async def api_generate(payload: Dict[str, Any]):
	"""Body: { "query": str, "document_type": str }
	Returns: { provider, document, snippets }
	"""
	query = (payload.get("query") or "").strip()
	document_type = (payload.get("document_type") or "исковое заявление").strip()
	if not query:
		raise HTTPException(status_code=400, detail="query is required")
	
	# Search
	db = vector_backend["db"]
	if db is None:
		raise HTTPException(status_code=500, detail="vector backend not initialized")
	
	similar = db.search_similar(query, n_results=5)
	
	# Generate
	result = generate_legal_document(query, similar, document_type=document_type)
	
	# Return
	snippets = [
		{
			"text": s.get("text", "")[:500],
			"source_file": s.get("metadata", {}).get("source_file", ""),
			"chunk_type": s.get("metadata", {}).get("chunk_type", ""),
		}
		for s in similar
	]
	return {"provider": result.get("provider"), "document": result.get("document"), "snippets": snippets}

if __name__ == "__main__":
	print(f"🚀 Starting RAG Legal Document Generator MVP")
	print(f"📡 API will be available at: http://{API_HOST}:{API_PORT}")
	print(f"📚 Documentation: http://{API_HOST}:{API_PORT}/docs")
	uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True, log_level="info")
