#!/usr/bin/env python3
"""
Быстрая проверка Gemini API ключа
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def quick_test():
    """Быстрый тест API ключа"""
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Получаем API ключ
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY не найден в .env файле")
        print("🔧 Создайте файл .env со строкой: GEMINI_API_KEY=ваш_ключ")
        return False
    
    print(f"✅ API ключ найден: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Настраиваем Gemini
        genai.configure(api_key=api_key)
        
        # Создаем модель
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        print("🔄 Тестируем подключение...")
        
        # Простой тест
        response = model.generate_content("Ответь: работает ли API?")
        
        if response.text:
            print("✅ Gemini API работает!")
            print(f"📝 Ответ: {response.text.strip()}")
            return True
        else:
            print("❌ Пустой ответ от API")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Ошибка: {e}")
        
        if "403" in error_msg:
            print("🔧 Проблема с авторизацией:")
            print("   - Проверьте корректность API ключа")
            print("   - Убедитесь, что ключ активен в Google AI Studio")
            print("   - Проверьте региональные ограничения")
            
        elif "User location is not supported" in error_msg:
            print("🌍 Регион не поддерживается:")
            print("   - Включите биллинг в Google Cloud Console")
            print("   - Или используйте VPN")
            
        return False

if __name__ == "__main__":
    quick_test()
