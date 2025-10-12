"""
Тестирование модуля обработки данных
"""

import os
import json
from data_processor import LegalDocumentProcessor
from config import PDF_DIR, JSON_DIR

def test_data_processing():
    """Тестирует обработку данных"""
    print("🧪 Тестирование модуля обработки данных...")
    
    # Создаем процессор
    processor = LegalDocumentProcessor()
    
    # Путь к тестовому PDF
    test_pdf = os.path.join(PDF_DIR, "test_legal_document.pdf")
    
    if not os.path.exists(test_pdf):
        print("❌ Тестовый PDF не найден!")
        return False
    
    print(f"📄 Обрабатываю файл: {test_pdf}")
    
    # Обрабатываем PDF
    result = processor.process_pdf_to_json(test_pdf)
    
    if not result:
        print("❌ Ошибка при обработке PDF")
        return False
    
    # Проверяем результат
    print("✅ Обработка завершена успешно!")
    print(f"📊 Статистика:")
    print(f"   - Номер дела: {result['metadata']['case_number']}")
    print(f"   - Суд: {result['metadata']['court']}")
    print(f"   - Тип документа: {result['metadata']['document_type']}")
    print(f"   - Количество чанков: {result['processing_info']['total_chunks']}")
    print(f"   - Правовых позиций: {result['processing_info']['total_positions']}")
    print(f"   - Длина текста: {result['processing_info']['text_length']} символов")
    
    # Сохраняем результат
    output_file = os.path.join(JSON_DIR, "test_result.json")
    processor.save_json(result, output_file)
    print(f"💾 Результат сохранен в: {output_file}")
    
    # Показываем пример чанка
    if result['chunks']:
        print(f"\n📝 Пример чанка:")
        print(f"   {result['chunks'][0]['text'][:100]}...")
    
    # Показываем правовые позиции
    if result['legal_positions']:
        print(f"\n⚖️ Найденные правовые позиции:")
        for i, pos in enumerate(result['legal_positions'][:2]):  # Показываем первые 2
            print(f"   {i+1}. {pos['text'][:80]}...")
    
    return True

if __name__ == "__main__":
    success = test_data_processing()
    if success:
        print("\n🎉 Тест прошел успешно! Модуль обработки данных работает корректно.")
    else:
        print("\n❌ Тест не прошел. Проверьте ошибки выше.")
