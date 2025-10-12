#!/usr/bin/env python3
"""
Скрипт для переиндексации векторной базы с улучшенной разметкой данных
"""

import os
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from data_processor import LegalDocumentProcessor
from vector_database import VectorDatabase
from config import PDF_DIR, JSON_DIR, CHROMA_DB_PATH
from loguru import logger

def main():
    """Переиндексация с улучшенной разметкой"""
    logger.info("🚀 Начинаю переиндексацию с улучшенной разметкой")
    
    # Очищаем старую векторную базу
    if os.path.exists(CHROMA_DB_PATH):
        logger.info(f"🗑️ Удаляю старую векторную базу: {CHROMA_DB_PATH}")
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Очищаем JSON файлы для переобработки
    if os.path.exists(JSON_DIR):
        logger.info(f"🗑️ Удаляю старые JSON файлы: {JSON_DIR}")
        import shutil
        shutil.rmtree(JSON_DIR)
        os.makedirs(JSON_DIR, exist_ok=True)
    
    # 1. Обработка PDF с улучшенной разметкой
    logger.info("📄 Обрабатываю PDF файлы с улучшенной разметкой...")
    processor = LegalDocumentProcessor(use_gemini_chunking=True)
    result = processor.process_all_pdfs(force=True)
    
    logger.info(f"✅ Обработка завершена: {result['processed']} файлов, {result['skipped']} пропущено, {result['errors']} ошибок")
    
    # 2. Создание новой векторной базы
    logger.info("🔍 Создаю новую векторную базу с улучшенной разметкой...")
    vector_db = VectorDatabase()
    
    # Загружаем JSON файлы
    load_result = vector_db.load_from_json_files(JSON_DIR)
    logger.info(f"📊 Результат загрузки: {load_result}")
    
    # Получаем статистику
    info = vector_db.get_collection_info()
    logger.info(f"✅ Векторная база создана: {info.get('document_count', 0)} документов")
    
    # 3. Тестируем улучшенный поиск
    logger.info("🧪 Тестирую улучшенный поиск...")
    
    test_queries = [
        "потребитель товар недостаток качество возврат",
        "договор купли продажи расторжение",
        "административное правонарушение штраф"
    ]
    
    for query in test_queries:
        logger.info(f"🔍 Тестовый запрос: {query}")
        results = vector_db.search_similar(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            meta = result.get('metadata', {})
            logger.info(f"  {i}. {meta.get('source_file', 'unknown')} - {meta.get('dispute_type', 'unknown')} - {meta.get('legal_area', 'unknown')}")
    
    logger.info("🎉 Переиндексация завершена успешно!")
    logger.info("💡 Теперь система будет находить более релевантные документы")

if __name__ == "__main__":
    main()
