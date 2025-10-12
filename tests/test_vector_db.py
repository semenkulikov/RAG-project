"""
Тестирование векторной базы данных
"""

import os
from vector_database import VectorDatabase
from config import JSON_DIR

def test_vector_database():
    """Тестирует работу векторной базы данных"""
    print("🧪 Тестирование векторной базы данных...")
    
    try:
        # Создаем векторную базу
        print("📊 Инициализация векторной базы данных...")
        vector_db = VectorDatabase()
        
        # Загружаем документы из JSON файлов
        print("📁 Загрузка документов из JSON файлов...")
        vector_db.load_from_json_files(JSON_DIR)
        
        # Получаем информацию о коллекции
        info = vector_db.get_collection_info()
        print(f"✅ Информация о коллекции:")
        print(f"   - Название: {info['name']}")
        print(f"   - Количество документов: {info['document_count']}")
        print(f"   - Путь к БД: {info['db_path']}")
        
        # Тестируем поиск
        test_queries = [
            "договор займа",
            "расписка деньги",
            "суд установил",
            "правовая позиция"
        ]
        
        print(f"\n🔍 Тестирование поиска:")
        for query in test_queries:
            print(f"\n📝 Запрос: '{query}'")
            results = vector_db.search_similar(query, n_results=2)
            
            if results:
                for i, result in enumerate(results):
                    print(f"   {i+1}. {result['text'][:80]}...")
                    print(f"      Расстояние: {result['distance']:.4f}")
                    print(f"      Тип: {result['metadata'].get('chunk_type', 'unknown')}")
            else:
                print("   ❌ Результаты не найдены")
        
        print(f"\n🎉 Тест векторной базы данных прошел успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_database()
    if not success:
        print("\n❌ Тест не прошел. Проверьте ошибки выше.")
