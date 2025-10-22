#!/usr/bin/env python3
"""
Быстрое решение проблемы 403 через проверку доступности
"""

import os
import requests
from dotenv import load_dotenv
from loguru import logger

def check_gemini_availability():
    """Проверяет доступность Gemini API из вашего региона"""
    
    logger.info("🌍 Проверяем доступность Gemini API...")
    
    # Проверяем статус Google AI Studio
    try:
        response = requests.get("https://aistudio.google.com/", timeout=10)
        if response.status_code == 200:
            logger.success("✅ Google AI Studio доступен")
        else:
            logger.error(f"❌ Google AI Studio недоступен: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к Google AI Studio: {e}")
        return False
    
    # Проверяем ваш IP и регион
    try:
        ip_response = requests.get("https://ipapi.co/json/", timeout=5)
        if ip_response.status_code == 200:
            ip_data = ip_response.json()
            country = ip_data.get('country_name', 'Unknown')
            logger.info(f"📍 Ваш регион: {country}")
            
            # Проверяем, есть ли ограничения
            restricted_regions = ['Russia', 'Belarus', 'Iran', 'North Korea', 'Cuba']
            if country in restricted_regions:
                logger.error(f"❌ Ваш регион ({country}) заблокирован для Gemini API")
                logger.info("🔧 РЕШЕНИЕ: Используйте VPN")
                logger.info("   Рекомендуемые страны: США, Канада, Германия, Великобритания")
                return False
            else:
                logger.success(f"✅ Регион {country} должен поддерживаться")
                return True
        else:
            logger.warning("⚠️ Не удается определить регион")
            return True
    except Exception as e:
        logger.warning(f"⚠️ Ошибка проверки региона: {e}")
        return True

def main():
    """Основная функция"""
    logger.info("🔍 Проверка доступности Gemini API")
    logger.info("=" * 50)
    
    available = check_gemini_availability()
    
    logger.info("=" * 50)
    if available:
        logger.success("✅ Gemini API должен быть доступен из вашего региона")
        logger.info("🔧 Если все еще ошибка 403:")
        logger.info("   1. Проверьте корректность API ключа")
        logger.info("   2. Включите биллинг в Google Cloud Console")
        logger.info("   3. Создайте новый проект и ключ")
    else:
        logger.error("❌ Ваш регион заблокирован для Gemini API")
        logger.info("🔧 РЕШЕНИЯ:")
        logger.info("   1. Используйте VPN (США/Европа)")
        logger.info("   2. Включите биллинг в Google Cloud Console")
        logger.info("   3. Используйте ChatGPT API (уже настроен как резерв)")

if __name__ == "__main__":
    main()
