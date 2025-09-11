"""
RAG Legal Document Generator MVP
Main entry point for the application
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import API_HOST, API_PORT, API_TITLE, API_VERSION

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="MVP for generating legal documents using RAG architecture"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Legal Document Generator MVP",
        "status": "running",
        "version": API_VERSION
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "api": "running",
            "database": "not_initialized",  # Will be updated later
            "gemini": "not_configured"      # Will be updated later
        }
    }

if __name__ == "__main__":
    print(f"🚀 Starting RAG Legal Document Generator MVP")
    print(f"📡 API will be available at: http://{API_HOST}:{API_PORT}")
    print(f"📚 Documentation: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
