"""
Двухэтапный стратегический поиск (про/контра) поверх ChromaDB.
"""

from typing import Dict, Any, List
import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
import os
import google.generativeai as genai
from ..utils.config import GEMINI_API_KEY


class StrategicRetriever:
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "vs_rf_universal_practice", api_key: str = None):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key) if api_key else None
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)
        # Настройка Gemini для контраргументов
        self._gemini_ready = False
        self._configure_gemini(api_key or GEMINI_API_KEY)

    def _configure_gemini(self, api_key: str):
        try:
            if api_key:
                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                self._gemini_ready = True
            else:
                self._gemini_model = None
        except Exception as e:
            logger.warning(f"Не удалось инициализировать Gemini для контраргументов: {e}")
            self._gemini_model = None
            self._gemini_ready = False

    def get_potential_counterarguments(self, query: str) -> List[str]:
        if self._gemini_ready:
            try:
                prompt = (
                    "Проанализируй фабулу дела истца. Сформулируй 3 наиболее вероятных и сильных возражения ответчика "
                    "в виде кратких тезисов (по одному на строку) для векторного поиска.\n\n"
                    f"Фабула: \"{query}\"\n\nВозражения:"
                )
                resp = self._gemini_model.generate_content(prompt)
                text = (resp.text or "").strip()
                lines = [l.strip().lstrip("-* ") for l in text.split("\n") if l.strip()]
                return lines[:3] if lines else []
            except Exception as e:
                logger.warning(f"Gemini недоступен для контраргументов, fallback на эвристику: {e}")

        # Fallback эвристика
        hints = [
            "отсутствие доказательств размера ущерба",
            "неправильный способ защиты права",
            "истечение срока исковой давности",
        ]
        return [f"{h}. Фабула: {query}" for h in hints]

    def query(self, query: str, n_results_pro: int = 4, n_results_contra: int = 2) -> Dict[str, Any]:
        pro = self.collection.query(query_texts=[query], n_results=n_results_pro)
        counter_queries = self.get_potential_counterarguments(query)
        contra = self.collection.query(query_texts=counter_queries, n_results=n_results_contra) if counter_queries else None

        context = {"supporting_practice": [], "rebuttal_practice": []}

        if pro and pro.get("metadatas"):
            for meta in pro["metadatas"][0]:
                context["supporting_practice"].append({
                    "source": f"Определение ВС РФ от {meta.get('date')} № {meta.get('case_number')}",
                    "ratio_decidendi": meta.get("ratio_decidendi"),
                    "content": meta.get("quote"),
                })

        if contra and contra.get("metadatas"):
            for metadata_list in contra["metadatas"]:
                for meta in metadata_list:
                    context["rebuttal_practice"].append({
                        "source": f"Определение ВС РФ от {meta.get('date')} № {meta.get('case_number')}",
                        "ratio_decidendi": meta.get("ratio_decidendi"),
                        "content": meta.get("quote"),
                    })

        return context


