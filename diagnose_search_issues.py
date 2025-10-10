#!/usr/bin/env python3
"""
Скрипт для диагностики проблем с поиском в векторной базе
"""

import os
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from vector_database import VectorDatabase
from config import CHROMA_DB_PATH
from loguru import logger

def main():
    """Диагностика проблем с поиском"""
    logger.info("🔍 Диагностика проблем с поиском в векторной базе")
    
    # Проверяем существование базы
    if not os.path.exists(CHROMA_DB_PATH):
        logger.error(f"❌ Векторная база не найдена: {CHROMA_DB_PATH}")
        logger.info("💡 Запустите: python reindex_with_improved_labeling.py")
        return
    
    try:
        # Инициализируем векторную базу
        vector_db = VectorDatabase()
        
        # Получаем информацию о коллекции
        info = vector_db.get_collection_info()
        logger.info(f"📊 Информация о коллекции: {info}")
        
        if info.get('document_count', 0) == 0:
            logger.error("❌ Коллекция пуста!")
            logger.info("💡 Запустите: python reindex_with_improved_labeling.py")
            return
        
        # Тестируем поиск без фильтров
        logger.info("🔍 Тестирую поиск без фильтров...")
        test_queries = [
            "потребитель товар недостаток",
            "договор купли продажи",
            "суд решение"
        ]
        
        for query in test_queries:
            logger.info(f"📝 Запрос: {query}")
            
            # Поиск без фильтра
            results_no_filter = vector_db.search_similar(query, n_results=3)
            logger.info(f"  Без фильтра: найдено {len(results_no_filter)} документов")
            
            for i, result in enumerate(results_no_filter, 1):
                meta = result.get('metadata', {})
                logger.info(f"    {i}. {meta.get('source_file', 'unknown')} - {meta.get('dispute_type', 'unknown')}")
            
            # Поиск с фильтром consumer_protection
            results_with_filter = vector_db.search_similar(query, n_results=3, dispute_type="consumer_protection")
            logger.info(f"  С фильтром consumer_protection: найдено {len(results_with_filter)} документов")
            
            # Поиск с фильтром contract_dispute
            results_with_filter2 = vector_db.search_similar(query, n_results=3, dispute_type="contract_dispute")
            logger.info(f"  С фильтром contract_dispute: найдено {len(results_with_filter2)} документов")
            
            logger.info("")
        
        # Проверяем метаданные документов
        logger.info("📋 Проверяю метаданные документов...")
        sample_results = vector_db.search_similar("тест", n_results=5)
        
        dispute_types = {}
        legal_areas = {}
        
        for result in sample_results:
            meta = result.get('metadata', {})
            dispute_type = meta.get('dispute_type', 'unknown')
            legal_area = meta.get('legal_area', 'unknown')
            
            dispute_types[dispute_type] = dispute_types.get(dispute_type, 0) + 1
            legal_areas[legal_area] = legal_areas.get(legal_area, 0) + 1
        
        logger.info(f"📊 Типы споров в выборке: {dispute_types}")
        logger.info(f"📊 Правовые области в выборке: {legal_areas}")
        
        # Если нет документов с нужными метаданными
        if not dispute_types.get('consumer_protection', 0):
            logger.warning("⚠️ Не найдено документов с dispute_type='consumer_protection'")
            logger.info("💡 Возможно, нужно переиндексировать с улучшенной разметкой")
        
        logger.info("✅ Диагностика завершена")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при диагностике: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
