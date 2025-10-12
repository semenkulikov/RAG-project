#!/usr/bin/env python3
"""
Тестирование асинхронной обработки на небольшом наборе документов
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from loguru import logger

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from async_reindex import AsyncReindexer

async def test_async_processing():
    """Тестирует асинхронную обработку на небольшом наборе"""
    logger.info("🧪 Тестирую асинхронную обработку...")
    
    # Создаем тестовый переиндексатор с 2 потоками
    reindexer = AsyncReindexer(max_workers=2)
    
    try:
        start_time = time.time()
        
        # Тестируем только обработку PDF (без создания векторной базы)
        pdf_results = await reindexer.process_all_pdfs_async()
        
        total_time = time.time() - start_time
        
        logger.info("🎉 Тестирование завершено!")
        logger.info(f"⏱️ Время обработки: {total_time:.2f} секунд")
        logger.info(f"📊 Результаты:")
        logger.info(f"   📄 Обработано: {pdf_results['processed']} файлов")
        logger.info(f"   ⏭️ Пропущено: {pdf_results['skipped']} файлов")
        logger.info(f"   ❌ Ошибок: {pdf_results['errors']} файлов")
        logger.info(f"   📊 Чанков: {pdf_results['total_chunks']}")
        logger.info(f"   ⚡ Скорость: {pdf_results['processed']/total_time:.2f} файлов/сек")
        
        # Сравниваем с синхронной обработкой
        logger.info("\n🔄 Для сравнения запустите синхронную обработку:")
        logger.info("   python reindex_with_improved_labeling.py")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await reindexer.cleanup()

async def main():
    """Основная функция"""
    logger.info("🚀 Запуск тестирования асинхронной обработки")
    
    try:
        await test_async_processing()
    except KeyboardInterrupt:
        logger.info("⏹️ Тестирование прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    # Настраиваем логирование
    logger.add("./logs/test_async.log", rotation="1 MB", level="INFO")
    
    # Запускаем тестирование
    asyncio.run(main())
