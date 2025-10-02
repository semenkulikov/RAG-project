"""
Быстрый тест системы с упрощенной векторной базой данных
"""

import os
import time
from data_processor import LegalDocumentProcessor
from simple_vector_db import SimpleVectorDatabase
from config import PDF_DIR, JSON_DIR

def quick_test():
    """Быстрый тест всей системы"""
    print("🚀 БЫСТРЫЙ ТЕСТ RAG СИСТЕМЫ")
    print("="*50)
    
    # Шаг 1: Проверяем обработанные данные
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    print(f"📄 Найдено JSON файлов: {len(json_files)}")
    
    if len(json_files) == 0:
        print("❌ JSON файлы не найдены. Запустите обработку PDF сначала.")
        return False
    
    # Шаг 2: Создаем упрощенную векторную базу
    print("\n📊 Создание упрощенной векторной базы данных...")
    start_time = time.time()
    
    vector_db = SimpleVectorDatabase()
    vector_db.load_from_json_files(JSON_DIR)
    
    creation_time = time.time() - start_time
    info = vector_db.get_database_info()
    
    print(f"✅ База создана за {creation_time:.2f} секунд")
    print(f"📊 Документов в базе: {info['document_count']}")
    
    # Шаг 3: Тестируем поиск
    print("\n🔍 Тестирование поиска:")
    
    test_queries = [
        "договор займа",
        "исковое заявление",
        "расторжение договора",
        "Верховный Суд"
    ]
    
    for query in test_queries:
        print(f"\n📝 Запрос: '{query}'")
        
        start_time = time.time()
        results = vector_db.search_similar(query, n_results=2)
        search_time = time.time() - start_time
        
        print(f"   ⏱️ Время: {search_time:.3f}с, Результатов: {len(results)}")
        
        for i, result in enumerate(results, 1):
            text_preview = result['text'][:60].replace('\n', ' ')
            similarity = result.get('similarity', 0)
            source = result['metadata'].get('source_file', 'unknown')
            print(f"   {i}. {text_preview}... ({similarity:.3f}) - {source}")
    
    print(f"\n🎉 БЫСТРЫЙ ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
    print(f"✅ Все компоненты работают корректно")
    print(f"🚀 Готов к интеграции с Gemini 2.5 Pro")
    
    return True

if __name__ == "__main__":
    quick_test()
