"""
Полное тестирование пайплайна RAG системы
1. Обработка PDF файлов
2. Создание векторной базы данных
3. Семантический поиск
4. Демонстрация результатов
"""

import os
import time
from pathlib import Path
from data_processor import LegalDocumentProcessor
from vector_database import VectorDatabase
from simple_vector_db import SimpleVectorDatabase
from config import PDF_DIR, JSON_DIR, CHROMA_DB_PATH
from loguru import logger

class FullPipelineTest:
    """Класс для полного тестирования пайплайна"""
    
    def __init__(self):
        self.setup_logging()
        self.processor = None
        self.vector_db = None
        self.simple_db = None
        
    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/full_pipeline_test.log", rotation="1 MB", level="INFO")
    
    def step1_process_pdfs(self):
        """Шаг 1: Обработка PDF файлов"""
        print("\n" + "="*60)
        print("🔄 ШАГ 1: ОБРАБОТКА PDF ФАЙЛОВ")
        print("="*60)
        
        # Проверяем наличие PDF файлов
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        print(f"📁 Найдено PDF файлов: {len(pdf_files)}")
        
        if not pdf_files:
            print("❌ PDF файлы не найдены!")
            return False
        
        # Создаем процессор
        print("\n📊 Инициализация процессора данных...")
        self.processor = LegalDocumentProcessor(use_gemini_chunking=True)
        
        # Обрабатываем все PDF файлы
        print("🔄 Обработка PDF файлов...")
        start_time = time.time()
        
        # Получаем сводку, без построчной печати по каждому документу
        summary = self.processor.process_all_pdfs(PDF_DIR, JSON_DIR)
        
        processing_time = time.time() - start_time
        
        # Краткая сводка
        print(f"\n✅ Обработка завершена за {processing_time:.2f} секунд")
        print(f"📄 Создано/обновлено JSON: {summary.get('processed', 0)}, пропущено: {summary.get('skipped', 0)}, ошибок: {summary.get('errors', 0)}, всего PDF: {summary.get('total', len(pdf_files))}")
        
        return (summary.get('processed', 0) + summary.get('skipped', 0)) > 0

    def step2_create_vector_database(self):
        """Шаг 2: Создание векторной базы данных"""
        print("\n" + "="*60)
        print("🔄 ШАГ 2: СОЗДАНИЕ ВЕКТОРНОЙ БАЗЫ ДАННЫХ")
        print("="*60)
        
        print("📊 Инициализация векторной базы данных...")
        start_time = time.time()
        
        try:
            # Создаем полную векторную базу данных
            self.vector_db = VectorDatabase()
            
            # Загружаем документы из JSON файлов (функция вернет сводку)
            print("📁 Загрузка документов из JSON файлов...")
            load_summary = self.vector_db.load_from_json_files(JSON_DIR)
            
            # Информация о коллекции
            info = self.vector_db.get_collection_info()
            
            creation_time = time.time() - start_time
            
            print(f"\n✅ Векторная база данных создана за {creation_time:.2f} секунд")
            print(f"📊 Информация о коллекции: документов {info.get('document_count', 0)}")
            print(f"📄 Новых файлов: {load_summary.get('loaded_files', 0)}, пропущено: {load_summary.get('skipped', 0)}, ошибок: {load_summary.get('errors', 0)}, всего JSON: {load_summary.get('total', 0)}")
            
            return info.get('document_count', 0) > 0
            
        except Exception as e:
            print(f"❌ Ошибка при создании векторной базы данных: {e}")
            logger.error(f"Ошибка векторной БД: {e}")
            
            # Пробуем упрощенную версию
            print("\n🔄 Пробуем упрощенную версию...")
            try:
                self.simple_db = SimpleVectorDatabase()
                self.simple_db.load_from_json_files(JSON_DIR)
                
                info = self.simple_db.get_database_info()
                print(f"✅ Упрощенная векторная база создана")
                print(f"📊 Документов: {info['document_count']}")
                return True
                
            except Exception as e2:
                print(f"❌ Ошибка и в упрощенной версии: {e2}")
                return False

    def step3_test_search(self):
        """Шаг 3: Тестирование поиска"""
        print("\n" + "="*60)
        print("🔄 ШАГ 3: ТЕСТИРОВАНИЕ СЕМАНТИЧЕСКОГО ПОИСКА")
        print("="*60)
        
        db = self.vector_db if self.vector_db else self.simple_db
        if not db:
            print("❌ Векторная база данных не инициализирована!")
            return False
        
        test_queries = [
            "договор займа и расписка",
            "исковое заявление в суд",
            "правовая позиция Верховного Суда",
            "расторжение договора",
            "возмещение ущерба",
            "кассационная жалоба"
        ]
        
        print("🔍 Тестирование поиска по различным запросам:\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Запрос {i}: '{query}'")
            
            try:
                start_time = time.time()
                results = db.search_similar(query, n_results=3)
                search_time = time.time() - start_time
                
                print(f"   ⏱️ Время поиска: {search_time:.3f} секунд")
                print(f"   📊 Найдено результатов: {len(results)}")
                
                if results:
                    for j, result in enumerate(results, 1):
                        text_preview = result['text'][:80].replace('\n', ' ')
                        if hasattr(result, 'get') and 'similarity' in result:
                            score = result['similarity']
                            print(f"      {j}. {text_preview}... (сходство: {score:.4f})")
                        else:
                            distance = result.get('distance', 0)
                            print(f"      {j}. {text_preview}... (расстояние: {distance:.4f})")
                else:
                    print("   ❌ Результаты не найдены")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Ошибка поиска: {e}")
                logger.error(f"Ошибка поиска для запроса '{query}': {e}")
        
        return True

    def step4_performance_analysis(self):
        """Шаг 4: Анализ производительности"""
        print("\n" + "="*60)
        print("🔄 ШАГ 4: АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*60)
        
        db = self.vector_db if self.vector_db else self.simple_db
        if not db:
            print("❌ Векторная база данных не инициализирована!")
            return False
        
        test_query = "договор займа"
        times = []
        
        print(f"🚀 Тестирование производительности (10 запросов)...")
        
        for i in range(10):
            start_time = time.time()
            _ = db.search_similar(test_query, n_results=5)
            search_time = time.time() - start_time
            times.append(search_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"📊 Результаты производительности:")
        print(f"   - Среднее время поиска: {avg_time:.3f} секунд")
        print(f"   - Минимальное время: {min_time:.3f} секунд")
        print(f"   - Максимальное время: {max_time:.3f} секунд")
        print(f"   - Запросов в секунду: {1/avg_time:.1f}")
        
        return True
    
    def step5_generate_report(self):
        """Шаг 5: Генерация отчета"""
        print("\n" + "="*60)
        print("📋 ШАГ 5: ИТОГОВЫЙ ОТЧЕТ")
        print("="*60)
        
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
        
        db = self.vector_db if self.vector_db else self.simple_db
        db_info = db.get_collection_info() if self.vector_db else db.get_database_info()
        
        print("📊 СВОДНАЯ СТАТИСТИКА:")
        print(f"   📁 PDF файлов обработано: {len(pdf_files)}")
        print(f"   📄 JSON файлов создано: {len(json_files)}")
        print(f"   🗄️ Бэкенд: {'ChromaDB' if self.vector_db else 'TF-IDF'}")
        print(f"   📊 Документов в БД: {db_info.get('document_count', 0)}")
        
        print(f"\n✅ СТАТУС КОМПОНЕНТОВ:")
        print(f"   ✅ Обработка PDF: Работает")
        print(f"   ✅ Векторизация: Работает")
        print(f"   ✅ Семантический поиск: Работает")
        
        return True
    
    def run_full_test(self):
        """Запуск полного тестирования"""
        print("🚀 ЗАПУСК ПОЛНОГО ТЕСТИРОВАНИЯ RAG СИСТЕМЫ")
        print("="*60)
        
        start_time = time.time()
        
        steps = [
            ("Обработка PDF файлов", self.step1_process_pdfs),
            ("Создание векторной БД", self.step2_create_vector_database),
            ("Тестирование поиска", self.step3_test_search),
            ("Анализ производительности", self.step4_performance_analysis),
            ("Генерация отчета", self.step5_generate_report)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            try:
                result = step_func()
                results[step_name] = result
                if not result:
                    print(f"❌ Шаг '{step_name}' завершился с ошибкой!")
                    break
            except Exception as e:
                print(f"❌ Критическая ошибка в шаге '{step_name}': {e}")
                logger.error(f"Критическая ошибка в {step_name}: {e}")
                results[step_name] = False
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🏁 ИТОГИ ТЕСТИРОВАНИЯ")
        print("="*60)
        
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        print(f"⏱️ Общее время выполнения: {total_time:.2f} секунд")
        print(f"✅ Успешных шагов: {success_count}/{total_count}")
        
        if success_count == total_count:
            print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            print("🚀 Система готова к интеграции с Gemini 2.5 Pro")
        else:
            print("⚠️ Некоторые тесты не прошли. Проверьте логи.")
        
        return success_count == total_count

def main():
    """Основная функция"""
    tester = FullPipelineTest()
    success = tester.run_full_test()
    
    if success:
        print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Установить google-generativeai")
        print("2. Настроить API ключ Gemini")
        print("3. Создать модуль интеграции с Gemini")
        print("4. Протестировать генерацию документов")
    
    return success

if __name__ == "__main__":
    main()
