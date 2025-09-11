"""
Configuration settings for RAG Legal Document Generator MVP
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Database Settings
CHROMA_DB_PATH = "./data/chroma_db"
VECTOR_COLLECTION_NAME = "legal_documents"

# Model Settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Using available model

# Search Settings
TOP_K_RESULTS = 5
SEARCH_THRESHOLD = 0.7

# File Paths
DATA_DIR = "./data"
PDF_DIR = "./data/pdfs"
JSON_DIR = "./data/json"
OUTPUT_DIR = "./data/output"

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "RAG Legal Document Generator"
API_VERSION = "1.0.0"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/app.log"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)
