"""
RAG Legal Document Generator MVP
Main entry point for the application
"""

import os
from typing import List, Dict, Any
from contextlib import asynccontextmanager

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
from loguru import logger

# Try to init full vector DB on startup; fallback to simple
vector_backend = {"type": "", "db": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_backend
    load_json_on_start = os.getenv("LOAD_JSON_ON_START", "0").lower() in ("1", "true", "yes")
    try:
        vdb = VectorDatabase()
        # IMPORTANT: by default do NOT reload JSON on server start — attach to persisted collection for instant startup
        if load_json_on_start:
            vdb.load_from_json_files(JSON_DIR)
        vector_backend = {"type": "chroma", "db": vdb}
    except Exception:
        svdb = SimpleVectorDatabase()
        if load_json_on_start:
            svdb.load_from_json_files(JSON_DIR)
        vector_backend = {"type": "tfidf", "db": svdb}
    
    yield
    
    # Shutdown
    # Cleanup if needed

# Initialize FastAPI app
app = FastAPI(
	title=API_TITLE,
	version=API_VERSION,
	description="MVP for generating legal documents using RAG architecture",
	lifespan=lifespan
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
	
	# Определяем тип спора для фильтрации
	query_lower = query.lower()
	dispute_type = None
	if any(keyword in query_lower for keyword in ["потребитель", "товар", "продавец", "недостаток", "качество"]):
		dispute_type = "consumer_protection"
	elif any(keyword in query_lower for keyword in ["договор", "контракт", "обязательство"]):
		dispute_type = "contract_dispute"
	
	# Используем фильтрацию по типу спора если доступна
	similar = []
	if hasattr(db, 'search_similar') and 'dispute_type' in db.search_similar.__code__.co_varnames:
		# Сначала пробуем с фильтром
		similar = db.search_similar(query, n_results=5, dispute_type=dispute_type)
		
		# Если не нашли с фильтром, пробуем без фильтра
		if not similar and dispute_type:
			logger.warning(f"Не найдено документов с фильтром {dispute_type}, пробую без фильтра")
			similar = db.search_similar(query, n_results=5)
	else:
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
