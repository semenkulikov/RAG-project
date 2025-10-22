#!/usr/bin/env python3
"""
Расширенная диагностика ошибки 403 Gemini API
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def check_api_key_format():
    """Проверяет формат API ключа"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("❌ GEMINI_API_KEY не найден в .env")
        return False
    
    logger.info(f"✅ API ключ найден: {api_key[:10]}...{api_key[-4:]}")
    
    # Проверяем формат
    if not api_key.startswith("AIza"):
        logger.error("❌ Неправильный формат ключа! Должен начинаться с 'AIza'")
        return False
    
    if len(api_key) < 35:
        logger.error("❌ Ключ слишком короткий! Должен быть ~39 символов")
        return False
    
    logger.success("✅ Формат ключа корректный")
    return True

def check_region_restrictions():
    """Проверяет региональные ограничения"""
    logger.info("🌍 Проверяем региональные ограничения...")
    
    try:
        # Проверяем доступность Google AI Studio
        response = requests.get("https://aistudio.google.com/", timeout=10)
        if response.status_code == 200:
            logger.success("✅ Google AI Studio доступен")
        else:
            logger.warning(f"⚠️ Google AI Studio недоступен: {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Не удается проверить Google AI Studio: {e}")
    
    # Проверяем IP через внешний сервис
    try:
        ip_response = requests.get("https://ipapi.co/json/", timeout=5)
        if ip_response.status_code == 200:
            ip_data = ip_response.json()
            country = ip_data.get('country_name', 'Unknown')
            logger.info(f"📍 Ваш IP: {ip_data.get('ip', 'Unknown')} ({country})")
            
            # Список проблемных регионов
            restricted_regions = ['Russia', 'Belarus', 'Iran', 'North Korea', 'Cuba']
            if country in restricted_regions:
                logger.error(f"❌ Ваш регион ({country}) может быть заблокирован")
                logger.error("🔧 Решения:")
                logger.error("   1. Используйте VPN (рекомендуется: США, Европа)")
                logger.error("   2. Включите биллинг в Google Cloud Console")
                return False
            else:
                logger.success(f"✅ Регион {country} должен поддерживаться")
        else:
            logger.warning("⚠️ Не удается определить регион")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка проверки региона: {e}")
    
    return True

def test_direct_api_call():
    """Тестирует прямой вызов API"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return False
    
    logger.info("🔄 Тестируем прямой вызов Gemini API...")
    
    try:
        # Настраиваем Gemini
        genai.configure(api_key=api_key)
        
        # Создаем модель
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Простой тест
        response = model.generate_content("Hi")
        
        if response.text:
            logger.success("✅ Прямой вызов API работает!")
            return True
        else:
            logger.error("❌ API вернул пустой ответ")
            return False
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Ошибка прямого вызова: {e}")
        
        # Детальный анализ ошибки
        if "403" in error_msg:
            logger.error("🔍 Детальный анализ ошибки 403:")
            logger.error("   Возможные причины:")
            logger.error("   1. Региональные ограничения")
            logger.error("   2. Неправильный формат ключа")
            logger.error("   3. Ключ не активирован")
            logger.error("   4. Превышены квоты")
            logger.error("   5. Блокировка IP адреса")
            
        elif "quota" in error_msg.lower():
            logger.error("📊 Проблема с квотами")
            
        elif "permission" in error_msg.lower():
            logger.error("🔐 Проблема с разрешениями")
            
        return False

def suggest_solutions():
    """Предлагает решения проблемы"""
    logger.info("💡 Рекомендуемые решения:")
    logger.info("")
    
    logger.info("🔧 Решение 1: Проверьте ключ в Google AI Studio")
    logger.info("   1. Зайдите на https://aistudio.google.com/")
    logger.info("   2. Убедитесь, что ключ активен")
    logger.info("   3. Проверьте, что проект не заблокирован")
    logger.info("")
    
    logger.info("🌍 Решение 2: Региональные ограничения")
    logger.info("   1. Используйте VPN (рекомендуется: США, Канада, Европа)")
    logger.info("   2. Или включите биллинг в Google Cloud Console")
    logger.info("")
    
    logger.info("💳 Решение 3: Включите биллинг")
    logger.info("   1. Зайдите на https://console.cloud.google.com/")
    logger.info("   2. Выберите ваш проект")
    logger.info("   3. Перейдите в 'Биллинг'")
    logger.info("   4. Включите биллинг (даже для бесплатного тарифа)")
    logger.info("")
    
    logger.info("🔄 Решение 4: Создайте новый проект")
    logger.info("   1. Создайте новый проект в Google Cloud Console")
    logger.info("   2. Включите Vertex AI API")
    logger.info("   3. Создайте новый API ключ")
    logger.info("   4. Обновите .env файл")
    logger.info("")
    
    logger.info("🛠️ Решение 5: Альтернативные методы")
    logger.info("   1. Используйте ChatGPT API (уже настроен как резерв)")
    logger.info("   2. Или локальные модели (Ollama, etc.)")

def main():
    """Основная функция диагностики"""
    logger.info("🔍 Расширенная диагностика ошибки 403 Gemini API")
    logger.info("=" * 60)
    
    # Проверка 1: Формат ключа
    logger.info("1️⃣ Проверка формата API ключа")
    key_ok = check_api_key_format()
    logger.info("")
    
    # Проверка 2: Региональные ограничения
    logger.info("2️⃣ Проверка региональных ограничений")
    region_ok = check_region_restrictions()
    logger.info("")
    
    # Проверка 3: Прямой вызов API
    logger.info("3️⃣ Тестирование прямого вызова API")
    api_ok = test_direct_api_call()
    logger.info("")
    
    # Итоги
    logger.info("=" * 60)
    logger.info("📊 ИТОГИ ДИАГНОСТИКИ:")
    
    if key_ok and region_ok and api_ok:
        logger.success("🎉 Все проверки пройдены! API должен работать.")
    else:
        logger.error("❌ Обнаружены проблемы. Смотрите рекомендации ниже.")
        logger.info("")
        suggest_solutions()
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
