"""
Модули обработки документов
"""

from .data_processor import LegalDocumentProcessor
from .gemini_chunker import GeminiChunker
from .chatgpt_chunker import ChatGPTChunker

__all__ = ["LegalDocumentProcessor", "GeminiChunker", "ChatGPTChunker"]
