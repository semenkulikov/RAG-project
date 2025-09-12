"""
Создание тестового PDF файла для демонстрации работы системы
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from config import PDF_DIR

def create_test_legal_document():
    """Создает тестовый юридический документ в формате PDF"""
    
    # Создаем директорию если её нет
    os.makedirs(PDF_DIR, exist_ok=True)
    
    # Путь к файлу
    pdf_path = os.path.join(PDF_DIR, "test_legal_document.pdf")
    
    # Создаем PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Заголовок
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "ОПРЕДЕЛЕНИЕ")
    c.drawString(50, height - 70, "Верховный Суд Российской Федерации")
    
    # Номер дела
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "№ 5-КГ25-89-К2")
    c.drawString(50, height - 120, "12 августа 2025 г.")
    
    # Судьи
    c.drawString(50, height - 150, "Судьи:")
    c.drawString(50, height - 170, "Асташов С.В. (председательствующий)")
    c.drawString(50, height - 190, "Киселёв А.П.")
    c.drawString(50, height - 210, "Петрушкин В.А.")
    
    # Стороны
    c.drawString(50, height - 250, "Стороны по делу:")
    c.drawString(50, height - 270, "Истец: Ларицкий Владислав")
    c.drawString(50, height - 290, "Ответчики: Сенин Сергей Витальевич, Хафизов Андрей Шэвкатович")
    
    # Основной текст
    c.drawString(50, height - 330, "Суд установил следующие обстоятельства:")
    c.drawString(50, height - 350, "Между сторонами был заключен договор займа на сумму 1,200,000 долларов США.")
    c.drawString(50, height - 370, "Ответчик Хафизов А.Ш. собственноручно написал расписку о получении")
    c.drawString(50, height - 390, "77,280,000 рублей.")
    
    c.drawString(50, height - 420, "Суд пришел к выводу, что нахождение долговой расписки у займодавца")
    c.drawString(50, height - 440, "подтверждает неисполнение денежного обязательства со стороны заемщика,")
    c.drawString(50, height - 460, "если им не будет доказано иное (ст. 408 ГК РФ).")
    
    c.drawString(50, height - 490, "Суд считает, что в споре по займу кредитор доказывает факт передачи")
    c.drawString(50, height - 510, "денег, а заемщик - факт возврата или безденежность.")
    
    c.drawString(50, height - 540, "Суд указывает, что подписание договора займа в день, отличный от")
    c.drawString(50, height - 560, "даты фактической передачи денег, само по себе не является основанием")
    c.drawString(50, height - 580, "для признания договора безденежным (ст. 807 ГК РФ).")
    
    # Резолютивная часть
    c.drawString(50, height - 620, "РЕШИЛ:")
    c.drawString(50, height - 640, "Апелляционное определение отменить, направить дело на новое рассмотрение.")
    
    # Сохраняем PDF
    c.save()
    print(f"Тестовый PDF создан: {pdf_path}")

if __name__ == "__main__":
    create_test_legal_document()
