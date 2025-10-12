#!/usr/bin/env python3
"""
Скрипт для тестирования ChatGPT-чанкования
"""

import os
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from gemini_chunker import GeminiChunker
from data_processor import LegalDocumentProcessor
from loguru import logger

def test_chatgpt_chunking():
    """Тестирует ChatGPT чанкование на примере"""
    
    # Тестовый текст из Model.txt
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
    
    logger.info("🧪 Тестирую ChatGPT чанкование...")
    
    # Тестируем Gemini чанкование (с резервом на ChatGPT)
    chunker = GeminiChunker()
    chunks = chunker.chunk_document(test_text, "test_document.pdf")
    
    logger.info(f"✅ Создано {len(chunks)} чанков:")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"  {i}. {chunk.get('title', 'Без названия')} ({chunk.get('type', 'unknown')})")
        logger.info(f"     Текст: {chunk.get('text', '')[:100]}...")
        if chunk.get('key_articles'):
            logger.info(f"     Статьи: {chunk['key_articles']}")
        logger.info("")
    
    # Тестируем интеграцию с data_processor
    logger.info("🔗 Тестирую интеграцию с data_processor...")
    processor = LegalDocumentProcessor(use_gemini_chunking=True)
    chunks_from_processor = processor.chunk_text(test_text, "test_integration.pdf")
    
    logger.info(f"✅ Data processor создал {len(chunks_from_processor)} чанков")
    
    # Оценка стоимости
    cost_estimate = chunker.estimate_cost(test_text)
    logger.info(f"💰 Оценка стоимости: ${cost_estimate['estimated_cost_usd']:.4f}")
    logger.info(f"   Входные токены: {cost_estimate['input_tokens']:.0f}")
    logger.info(f"   Выходные токены: {cost_estimate['output_tokens']:.0f}")

def test_fallback_chunking():
    """Тестирует резервное чанкование"""
    logger.info("🔄 Тестирую резервное чанкование...")
    
    processor = LegalDocumentProcessor(use_gemini_chunking=False)
    test_text = "Абзац 1.\n\nАбзац 2.\n\nАбзац 3."
    
    chunks = processor.fallback_chunking(test_text, "test_fallback.pdf")
    
    logger.info(f"✅ Резервное чанкование создало {len(chunks)} чанков:")
    for chunk in chunks:
        logger.info(f"  - {chunk['title']}: {chunk['text'][:50]}...")

def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестирования ChatGPT-чанкования")
    
    # Проверяем наличие API ключа
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY не установлен. Тестируем только резервное чанкование.")
        test_fallback_chunking()
        return
    
    try:
        test_chatgpt_chunking()
        test_fallback_chunking()
        logger.info("🎉 Тестирование завершено успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
