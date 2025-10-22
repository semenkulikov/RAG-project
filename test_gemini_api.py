#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работоспособности Gemini API ключа
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def test_gemini_api():
    """Тестирует подключение к Gemini API"""
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Получаем API ключ
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("❌ GEMINI_API_KEY не найден в .env файле")
        logger.info("🔧 Создайте файл .env в корне проекта со строкой:")
        logger.info("   GEMINI_API_KEY=ваш_ключ_здесь")
        return False
    
    logger.info(f"✅ API ключ найден: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Настраиваем Gemini
        genai.configure(api_key=api_key)
        
        # Создаем модель
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        logger.info("🔄 Тестируем подключение к Gemini API...")
        
        # Простой тест
        test_prompt = "Привет! Ответь коротко: работает ли API?"
        response = model.generate_content(test_prompt)
        
        if response.text:
            logger.success("✅ Gemini API работает корректно!")
            logger.info(f"📝 Ответ модели: {response.text.strip()}")
            return True
        else:
            logger.error("❌ Gemini API вернул пустой ответ")
            return False
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Ошибка подключения к Gemini API: {e}")
        
        # Детальная диагностика
        if "403" in error_msg or "Forbidden" in error_msg:
            logger.error("🔧 Проблема с авторизацией (403):")
            logger.error("   1. Проверьте корректность API ключа")
            logger.error("   2. Убедитесь, что ключ активен в Google AI Studio")
            logger.error("   3. Проверьте региональные ограничения")
            logger.error("   4. Включите биллинг если необходимо")
            logger.error("")
            logger.error("🔍 Запустите расширенную диагностику:")
            logger.error("   python diagnose_gemini_403.py")
            logger.error("")
            logger.error("🌍 Проверьте регион:")
            logger.error("   python check_gemini_region.py")
            
        elif "User location is not supported" in error_msg:
            logger.error("🌍 Проблема с регионом:")
            logger.error("   1. Ваш регион не поддерживается для бесплатного доступа")
            logger.error("   2. Включите биллинг в Google Cloud Console")
            logger.error("   3. Или используйте VPN из поддерживаемого региона")
            
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            logger.error("📊 Проблема с квотами:")
            logger.error("   1. Превышены лимиты запросов")
            logger.error("   2. Проверьте квоты в Google Cloud Console")
            logger.error("   3. Запросите увеличение лимитов")
            
        elif "invalid" in error_msg.lower() or "malformed" in error_msg.lower():
            logger.error("🔑 Проблема с ключом:")
            logger.error("   1. API ключ некорректный или поврежден")
            logger.error("   2. Создайте новый ключ в Google AI Studio")
            logger.error("   3. Убедитесь, что скопировали ключ полностью")
            
        else:
            logger.error("❓ Неизвестная ошибка:")
            logger.error(f"   {error_msg}")
            
        return False

def test_advanced_features():
    """Тестирует расширенные возможности Gemini"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        logger.info("🔄 Тестируем расширенные возможности...")
        
        # Тест структурирования JSON
        json_test_prompt = """
        Проанализируй следующий текст и представь в формате JSON:
        
        Текст: "Суд постановил взыскать с ответчика 100000 рублей в пользу истца."
        
        Верни JSON с полями: decision, amount, currency, parties.
        """
        
        response = model.generate_content(json_test_prompt)
        
        if response.text and "{" in response.text:
            logger.success("✅ JSON структурирование работает!")
            logger.info(f"📝 JSON ответ: {response.text.strip()[:200]}...")
        else:
            logger.warning("⚠️ JSON структурирование работает частично")
            
        # Тест длинного текста
        long_text = "Это тестовый текст. " * 100  # ~2000 символов
        long_prompt = f"Проанализируй этот текст и дай краткое резюме: {long_text}"
        
        response = model.generate_content(long_prompt)
        
        if response.text:
            logger.success("✅ Обработка длинных текстов работает!")
        else:
            logger.warning("⚠️ Проблемы с длинными текстами")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в расширенных тестах: {e}")
        return False

def main():
    """Основная функция тестирования"""
    
    logger.info("🚀 Запуск тестирования Gemini API")
    logger.info("=" * 50)
    
    # Базовый тест
    basic_test = test_gemini_api()
    
    if basic_test:
        logger.info("=" * 50)
        # Расширенные тесты
        advanced_test = test_advanced_features()
        
        logger.info("=" * 50)
        if advanced_test:
            logger.success("🎉 Все тесты пройдены! Gemini API полностью готов к работе.")
        else:
            logger.warning("⚠️ Базовые функции работают, но есть проблемы с расширенными возможностями.")
    else:
        logger.error("💥 Базовые тесты не пройдены. Исправьте проблемы с API ключом.")
        
    logger.info("=" * 50)
    logger.info("📋 Инструкции по созданию API ключа:")
    logger.info("   1. Зайдите на https://aistudio.google.com/")
    logger.info("   2. Создайте новый проект или выберите существующий")
    logger.info("   3. Перейдите в раздел 'Get API key'")
    logger.info("   4. Создайте новый ключ")
    logger.info("   5. Скопируйте ключ в файл .env как GEMINI_API_KEY=ваш_ключ")

if __name__ == "__main__":
    main()
