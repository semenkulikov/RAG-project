"""
Основной класс для генерации юридических документов
"""

import sys
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from processors import LegalDocumentProcessor
from databases import VectorDatabase, SimpleVectorDatabase
from integrations import generate_legal_document
from utils.config import PDF_DIR, JSON_DIR, GEMINI_API_KEY, OPENAI_API_KEY


class LegalDocumentGenerator:
    """
    Основной класс для генерации юридических документов с использованием RAG
    """

    def __init__(self, use_simple_db: bool = False, use_gemini_chunking: bool = True):
        """
        Инициализация генератора документов

        Args:
            use_simple_db: Использовать упрощенную векторную базу (TF-IDF)
            use_gemini_chunking: Использовать Gemini для чанкования
        """
        self.use_simple_db = use_simple_db
        self.use_gemini_chunking = use_gemini_chunking

        # Инициализируем компоненты
        self._init_processor()
        self._init_vector_database()

        logger.info("✅ LegalDocumentGenerator инициализирован")

    def _init_processor(self):
        """Инициализация процессора документов"""
        try:
            self.processor = LegalDocumentProcessor(
                use_gemini_chunking=self.use_gemini_chunking
            )
            logger.info("✅ Процессор документов инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации процессора: {e}")
            raise

    def _init_vector_database(self):
        """Инициализация векторной базы данных"""
        try:
            if self.use_simple_db:
                self.vector_db = SimpleVectorDatabase()
                logger.info("✅ Упрощенная векторная база инициализирована")
            else:
                self.vector_db = VectorDatabase()
                logger.info("✅ Векторная база данных инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации векторной базы: {e}")
            # Fallback на простую базу
            self.vector_db = SimpleVectorDatabase()
            self.use_simple_db = True
            logger.warning("⚠️ Переключился на упрощенную векторную базу")

    def process_documents(
        self, pdf_directory: str = PDF_DIR, force: bool = False
    ) -> Dict[str, Any]:
        """
        Обрабатывает PDF документы и создает векторную базу

        Args:
            pdf_directory: Директория с PDF файлами
            force: Принудительная переобработка

        Returns:
            Статистика обработки
        """
        logger.info(f"🔄 Начинаю обработку документов из {pdf_directory}")

        try:
            # Обрабатываем PDF файлы
            pdf_stats = self.processor.process_all_pdfs(force=force)

            # Загружаем в векторную базу
            if not self.use_simple_db:
                vector_stats = self.vector_db.load_from_json_files(JSON_DIR)
            else:
                vector_stats = self.vector_db.load_documents_from_json(JSON_DIR)

            logger.info("✅ Обработка документов завершена")

            return {
                "pdf_processing": pdf_stats,
                "vector_loading": vector_stats,
                "total_chunks": pdf_stats.get("total_chunks", 0),
                "total_positions": pdf_stats.get("total_positions", 0),
            }

        except Exception as e:
            logger.error(f"❌ Ошибка обработки документов: {e}")
            raise

    def generate_document(
        self,
        case_description: str,
        document_type: str = "исковое заявление",
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Генерирует юридический документ на основе описания дела

        Args:
            case_description: Описание обстоятельств дела
            document_type: Тип документа для генерации
            n_results: Количество релевантных документов для поиска

        Returns:
            Сгенерированный документ и метаданные
        """
        logger.info(
            f"📝 Генерирую {document_type} для дела: {case_description[:100]}..."
        )

        try:
            # Поиск релевантных документов
            relevant_docs = self.vector_db.search_similar(
                case_description, n_results=n_results
            )

            if not relevant_docs:
                logger.warning("⚠️ Релевантные документы не найдены")
                return {
                    "document": "Не удалось найти релевантные судебные решения для генерации документа.",
                    "relevant_docs": [],
                    "status": "no_results",
                }

            logger.info(f"🔍 Найдено {len(relevant_docs)} релевантных документов")

            # Генерация документа
            generated_doc = generate_legal_document(
                case_description=case_description,
                document_type=document_type,
                relevant_docs=relevant_docs,
            )

            logger.info("✅ Документ сгенерирован успешно")

            return {
                "document": generated_doc,
                "relevant_docs": relevant_docs,
                "document_type": document_type,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"❌ Ошибка генерации документа: {e}")
            return {
                "document": f"Ошибка генерации документа: {str(e)}",
                "relevant_docs": [],
                "status": "error",
            }

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов

        Args:
            query: Поисковый запрос
            n_results: Количество результатов

        Returns:
            Список релевантных документов
        """
        try:
            results = self.vector_db.search_similar(query, n_results=n_results)
            logger.info(f"🔍 Найдено {len(results)} документов для запроса: {query}")
            return results
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику по векторной базе данных

        Returns:
            Статистика базы данных
        """
        try:
            if hasattr(self.vector_db, "get_collection_info"):
                return self.vector_db.get_collection_info()
            else:
                return {
                    "total_documents": (
                        len(self.vector_db.documents)
                        if hasattr(self.vector_db, "documents")
                        else 0
                    ),
                    "database_type": "simple" if self.use_simple_db else "vector",
                }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, bool]:
        """
        Проверка состояния системы

        Returns:
            Статус компонентов системы
        """
        health_status = {"processor": False, "vector_db": False, "api_keys": False}

        try:
            # Проверка процессора
            health_status["processor"] = self.processor is not None

            # Проверка векторной базы
            health_status["vector_db"] = self.vector_db is not None

            # Проверка API ключей
            health_status["api_keys"] = bool(GEMINI_API_KEY or OPENAI_API_KEY)

        except Exception as e:
            logger.error(f"❌ Ошибка проверки состояния: {e}")

        return health_status
