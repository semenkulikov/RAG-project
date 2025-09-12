"""
Модуль обработки данных для RAG системы
Извлекает текст из PDF, структурирует и подготавливает для векторизации
"""

import os
import json
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF
from pathlib import Path
from loguru import logger
from config import PDF_DIR, JSON_DIR, DATA_DIR

class LegalDocumentProcessor:
    """Класс для обработки юридических документов"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/data_processor.log", rotation="1 MB", level="INFO")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Извлекает текст из PDF документа
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Извлеченный текст
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += f"\n--- Страница {page_num + 1} ---\n{text}\n"
            
            doc.close()
            logger.info(f"Успешно извлечен текст из {pdf_path}")
            return full_text
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из {pdf_path}: {e}")
            return ""
    
    def extract_legal_metadata(self, text: str) -> Dict[str, Any]:
        """
        Извлекает метаданные из юридического документа
        
        Args:
            text: Текст документа
            
        Returns:
            Словарь с метаданными
        """
        metadata = {
            "case_number": "",
            "court": "",
            "date": "",
            "judges": [],
            "parties": [],
            "document_type": ""
        }
        
        # Извлечение номера дела
        case_patterns = [
            r"№\s*(\d+[-\w]+\d+)",
            r"дело\s*№\s*(\d+[-\w]+\d+)",
            r"№\s*(\d+-\w+-\d+-\w+)"
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["case_number"] = match.group(1)
                break
        
        # Извлечение суда
        court_patterns = [
            r"Верховный\s+Суд\s+Российской\s+Федерации",
            r"Арбитражный\s+суд\s+[^,\n]+",
            r"Районный\s+суд\s+[^,\n]+",
            r"Городской\s+суд\s+[^,\n]+"
        ]
        
        for pattern in court_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["court"] = match.group(0)
                break
        
        # Извлечение даты
        date_patterns = [
            r"(\d{1,2}\s+\w+\s+\d{4})\s*г\.",
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{1,2}\.\d{1,2}\.\d{4})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["date"] = match.group(1)
                break
        
        # Извлечение судей
        judge_pattern = r"([А-Я][а-я]+\s+[А-Я]\.\s*[А-Я]\.)"
        judges = re.findall(judge_pattern, text)
        metadata["judges"] = list(set(judges))
        
        # Определение типа документа
        if "определение" in text.lower():
            metadata["document_type"] = "Определение"
        elif "решение" in text.lower():
            metadata["document_type"] = "Решение"
        elif "постановление" in text.lower():
            metadata["document_type"] = "Постановление"
        
        return metadata
    
    def extract_legal_positions(self, text: str) -> List[Dict[str, str]]:
        """
        Извлекает правовые позиции из текста
        
        Args:
            text: Текст документа
            
        Returns:
            Список правовых позиций
        """
        positions = []
        
        # Поиск правовых позиций по ключевым словам
        position_keywords = [
            "суд установил",
            "суд пришел к выводу",
            "суд считает",
            "суд полагает",
            "суд указывает",
            "суд подчеркивает"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in position_keywords):
                # Извлечение ссылок на статьи
                articles = re.findall(r'ст\.\s*\d+[^\s]*', sentence)
                
                positions.append({
                    "text": sentence,
                    "articles": articles,
                    "type": "legal_position"
                })
        
        return positions
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки для векторизации
        
        Args:
            text: Текст для разбиения
            chunk_size: Размер чанка в символах
            overlap: Перекрытие между чанками
            
        Returns:
            Список чанков с метаданными
        """
        chunks = []
        
        # Разбиение по абзацам
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Если абзац слишком длинный, разбиваем его
            if len(paragraph) > chunk_size:
                words = paragraph.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk + " " + word) > chunk_size:
                        if temp_chunk:
                            chunks.append({
                                "id": chunk_id,
                                "text": temp_chunk.strip(),
                                "type": "legal_text",
                                "length": len(temp_chunk)
                            })
                            chunk_id += 1
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word
                
                if temp_chunk:
                    chunks.append({
                        "id": chunk_id,
                        "text": temp_chunk.strip(),
                        "type": "legal_text",
                        "length": len(temp_chunk)
                    })
                    chunk_id += 1
            else:
                # Если текущий чанк + абзац не превышают размер, добавляем
                if len(current_chunk + " " + paragraph) <= chunk_size:
                    current_chunk += " " + paragraph
                else:
                    # Сохраняем текущий чанк
                    if current_chunk.strip():
                        chunks.append({
                            "id": chunk_id,
                            "text": current_chunk.strip(),
                            "type": "legal_text",
                            "length": len(current_chunk)
                        })
                        chunk_id += 1
                    
                    # Начинаем новый чанк
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk.strip():
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "type": "legal_text",
                "length": len(current_chunk)
            })
        
        logger.info(f"Создано {len(chunks)} чанков из текста")
        return chunks
    
    def process_pdf_to_json(self, pdf_path: str) -> Dict[str, Any]:
        """
        Обрабатывает PDF файл и создает структурированный JSON
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Структурированные данные в формате JSON
        """
        logger.info(f"Начинаю обработку файла: {pdf_path}")
        
        # Извлечение текста
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"Не удалось извлечь текст из {pdf_path}")
            return {}
        
        # Извлечение метаданных
        metadata = self.extract_legal_metadata(text)
        
        # Извлечение правовых позиций
        legal_positions = self.extract_legal_positions(text)
        
        # Разбиение на чанки
        chunks = self.chunk_text(text)
        
        # Формирование итоговой структуры
        result = {
            "metadata": metadata,
            "legal_positions": legal_positions,
            "chunks": chunks,
            "full_text": text,
            "source_file": os.path.basename(pdf_path),
            "processing_info": {
                "total_chunks": len(chunks),
                "total_positions": len(legal_positions),
                "text_length": len(text)
            }
        }
        
        logger.info(f"Обработка завершена: {len(chunks)} чанков, {len(legal_positions)} позиций")
        return result
    
    def save_json(self, data: Dict[str, Any], output_path: str):
        """
        Сохраняет данные в JSON файл
        
        Args:
            data: Данные для сохранения
            output_path: Путь для сохранения
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Данные сохранены в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении в {output_path}: {e}")
    
    def process_all_pdfs(self, input_dir: str = PDF_DIR, output_dir: str = JSON_DIR):
        """
        Обрабатывает все PDF файлы в директории
        
        Args:
            input_dir: Директория с PDF файлами
            output_dir: Директория для сохранения JSON
        """
        if not os.path.exists(input_dir):
            logger.warning(f"Директория {input_dir} не существует")
            return
        
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"PDF файлы не найдены в {input_dir}")
            return
        
        logger.info(f"Найдено {len(pdf_files)} PDF файлов для обработки")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            json_file = pdf_file.replace('.pdf', '.json')
            json_path = os.path.join(output_dir, json_file)
            
            try:
                data = self.process_pdf_to_json(pdf_path)
                if data:
                    self.save_json(data, json_path)
            except Exception as e:
                logger.error(f"Ошибка при обработке {pdf_file}: {e}")

def main():
    """Основная функция для тестирования"""
    processor = LegalDocumentProcessor()
    
    # Создаем тестовый PDF файл если его нет
    test_pdf_path = os.path.join(PDF_DIR, "test_document.pdf")
    
    if not os.path.exists(test_pdf_path):
        logger.info("Создаю тестовый PDF файл...")
        # Здесь можно создать простой тестовый PDF
        logger.warning("Тестовый PDF не найден. Поместите PDF файлы в папку data/pdfs/")
        return
    
    # Обрабатываем все PDF файлы
    processor.process_all_pdfs()

if __name__ == "__main__":
    main()
