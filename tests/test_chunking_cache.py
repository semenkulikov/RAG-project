#!/usr/bin/env python3
"""
Скрипт для тестирования кэширования результатов чанкования
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.processors.gemini_chunker import GeminiChunker

def test_caching():
    """Тестирует работу кэширования"""
    
    test_text = """
    ВЕРХОВНЫЙ СУД РОССИЙСКОЙ ФЕДЕРАЦИИ
    ОПРЕДЕЛЕНИЕ
    от 15 августа 2023 г. № 44-КГ23-11-К7
    
    Судебная коллегия по гражданским делам Верховного Суда Российской Федерации рассмотрела в судебном заседании кассационную жалобу...
    
    Установлено, что истец приобрел у ответчика товар ненадлежащего качества. Ответчик отказался принять товар обратно, ссылаясь на отсутствие гарантийного срока.
    
    Суд первой инстанции пришел к выводу, что на истце лежит обязанность доказать наличие недостатков товара. Однако данная позиция противоречит закону.
    
    В соответствии со статьей 18 Закона РФ "О защите прав потребителей", продавец обязан принять товар ненадлежащего качества и провести проверку качества за свой счет.
    
    Верховный Суд РФ указал: «Разрешая спор, суд пришёл к выводу, что на истце лежит обязанность доказать, что качество приобретённого ею товара не соответствовало условиям договора... Между тем, в соответствии с приведёнными выше положениями действующего законодательства обязанность доказать добросовестность своих действий лежит именно на продавце товара».
    """
    
    logger.info("🧪 Тестирую кэширование результатов чанкования...")
    
    # Создаем экземпляр чанкера
    chunker = GeminiChunker()
    
    # Первый запрос (должен обработаться через API)
    logger.info("📡 Первый запрос (через API)...")
    start_time = time.time()
    chunks1 = chunker.chunk_document(test_text, "test_document.pdf")
    api_time = time.time() - start_time
    
    logger.info(f"✅ Получено {len(chunks1)} чанков за {api_time:.2f} секунд")
    
    # Второй запрос (должен использовать кэш)
    logger.info("💾 Второй запрос (из кэша)...")
    start_time = time.time()
    chunks2 = chunker.chunk_document(test_text, "test_document.pdf")
    cache_time = time.time() - start_time
    
    logger.info(f"✅ Получено {len(chunks2)} чанков за {cache_time:.2f} секунд")
    
    # Проверяем результаты
    if chunks1 == chunks2:
        logger.info("✅ Кэширование работает корректно!")
        if cache_time > 0:
            logger.info(f"⚡ Ускорение: {api_time/cache_time:.1f}x")
        else:
            logger.info("⚡ Ускорение: мгновенное (из кэша)")
    else:
        logger.error("❌ Кэширование работает некорректно!")
    
    # Проверяем размер кэша
    cache_size = len(chunker.cache)
    logger.info(f"📊 Размер кэша: {cache_size} записей")
    
    # Тестируем с разными файлами (но тот же текст)
    logger.info("🔄 Тестирую с разными файлами...")
    
    chunks3 = chunker.chunk_document(test_text, "different_file.pdf")
    cache_size_after = len(chunker.cache)
    
    # Кэш не должен обновиться, так как текст тот же
    if cache_size_after == cache_size:
        logger.info("✅ Кэш работает корректно - одинаковый текст не создает новую запись")
    else:
        logger.warning("⚠️ Кэш обновился для того же текста")
    
    # Тестируем с другим текстом
    logger.info("🔄 Тестирую с другим текстом...")
    different_text = "Это совершенно другой юридический документ для тестирования кэширования."
    chunks4 = chunker.chunk_document(different_text, "different_text.pdf")
    cache_size_final = len(chunker.cache)
    
    if cache_size_final > cache_size_after:
        logger.info("✅ Кэш обновился для нового текста")
    else:
        logger.warning("⚠️ Кэш не обновился для нового текста")
    
    # Оценка экономии
    if api_time > 0 and cache_time > 0:
        savings_percent = (api_time - cache_time) / api_time * 100
        logger.info(f"💰 Экономия времени: {savings_percent:.1f}%")
        
        # Примерная экономия токенов для большого корпуса
        estimated_documents = 1000
        estimated_savings = estimated_documents * (api_time - cache_time)
        logger.info(f"📈 Для {estimated_documents} документов экономия: {estimated_savings:.1f} секунд")

def test_cache_persistence():
    """Тестирует сохранение кэша между сессиями"""
    logger.info("💾 Тестирую сохранение кэша между сессиями...")
    
    test_text = "Тестовый текст для проверки сохранения кэша."
    
    # Первая сессия
    chunker1 = GeminiChunker()
    chunks1 = chunker1.chunk_document(test_text, "persistence_test.pdf")
    logger.info(f"Сессия 1: {len(chunks1)} чанков")
    
    # Вторая сессия (новый экземпляр)
    chunker2 = GeminiChunker()
    chunks2 = chunker2.chunk_document(test_text, "persistence_test.pdf")
    logger.info(f"Сессия 2: {len(chunks2)} чанков")
    
    if chunks1 == chunks2:
        logger.info("✅ Кэш сохраняется между сессиями!")
    else:
        logger.warning("⚠️ Кэш не сохраняется между сессиями")

def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестирования кэширования")
    
    # Проверяем наличие API ключа
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY не установлен. Тестируем только логику кэширования.")
        test_cache_persistence()
        return
    
    try:
        test_caching()
        test_cache_persistence()
        logger.info("🎉 Тестирование кэширования завершено!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
