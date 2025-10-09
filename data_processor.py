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
			"document_type": "",
			"legal_area": "",
			"key_articles": [],
			"dispute_type": "",
			"consumer_protection": False,
			"contract_dispute": False,
			"administrative": False,
			"criminal": False
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
		elif "исковое заявление" in text.lower():
			metadata["document_type"] = "исковое заявление"
		elif "апелляционная жалоба" in text.lower():
			metadata["document_type"] = "апелляционная жалоба"
		elif "кассационная жалоба" in text.lower():
			metadata["document_type"] = "кассационная жалоба"
		
		# Классификация по типу спора
		text_lower = text.lower()
		
		# Потребительские споры
		consumer_keywords = [
			"защита прав потребителей", "зозпп", "потребитель", "продавец", 
			"изготовитель", "исполнитель", "недостаток товара", "ненадлежащее качество",
			"возврат товара", "замена товара", "устранение недостатков", "неустойка",
			"компенсация морального вреда", "штраф", "ст. 18", "ст. 25", "ст. 15"
		]
		
		if any(keyword in text_lower for keyword in consumer_keywords):
			metadata["consumer_protection"] = True
			metadata["dispute_type"] = "consumer_protection"
			metadata["legal_area"] = "защита прав потребителей"
		
		# Договорные споры
		contract_keywords = [
			"договор", "контракт", "соглашение", "обязательство", "исполнение договора",
			"нарушение договора", "расторжение договора", "взыскание долга", "неустойка"
		]
		
		if any(keyword in text_lower for keyword in contract_keywords):
			metadata["contract_dispute"] = True
			if not metadata["dispute_type"]:
				metadata["dispute_type"] = "contract_dispute"
				metadata["legal_area"] = "договорное право"
		
		# Административные дела
		admin_keywords = [
			"административное дело", "административное правонарушение", "штраф гибдд",
			"лишение прав", "административная ответственность", "коап"
		]
		
		if any(keyword in text_lower for keyword in admin_keywords):
			metadata["administrative"] = True
			if not metadata["dispute_type"]:
				metadata["dispute_type"] = "administrative"
				metadata["legal_area"] = "административное право"
		
		# Уголовные дела
		criminal_keywords = [
			"уголовное дело", "преступление", "уголовная ответственность", "наказание",
			"суд присяжных", "обвинение", "защита", "прокурор"
		]
		
		if any(keyword in text_lower for keyword in criminal_keywords):
			metadata["criminal"] = True
			if not metadata["dispute_type"]:
				metadata["dispute_type"] = "criminal"
				metadata["legal_area"] = "уголовное право"
		
		# Извлечение ключевых статей
		article_patterns = [
			r"ст\.\s*(\d+)\s*(?:п\.\s*(\d+))?\s*(?:ч\.\s*(\d+))?\s*(?:ГК|ГПК|АПК|УК|КоАП|ЗоЗПП)",
			r"статья\s*(\d+)\s*(?:пункт\s*(\d+))?\s*(?:часть\s*(\d+))?\s*(?:ГК|ГПК|АПК|УК|КоАП|ЗоЗПП)",
			r"(\d+)\s*статья\s*(?:пункт\s*(\d+))?\s*(?:часть\s*(\d+))?\s*(?:ГК|ГПК|АПК|УК|КоАП|ЗоЗПП)"
		]
		
		for pattern in article_patterns:
			matches = re.findall(pattern, text, re.IGNORECASE)
			for match in matches:
				article_ref = f"ст. {match[0]}"
				if match[1]:
					article_ref += f" п. {match[1]}"
				if match[2]:
					article_ref += f" ч. {match[2]}"
				metadata["key_articles"].append(article_ref)
		
		return metadata
	
	def extract_legal_positions(self, text: str) -> List[Dict[str, str]]:
		"""
		Извлекает правовые позиции из текста
		"""
		positions = []
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
		"""
		chunks = []
		paragraphs = text.split('\n\n')
		current_chunk = ""
		chunk_id = 0
		for paragraph in paragraphs:
			paragraph = paragraph.strip()
			if not paragraph:
				continue
			if len(paragraph) > chunk_size:
				words = paragraph.split()
				temp_chunk = ""
				for word in words:
					if len(temp_chunk + " " + word) > chunk_size:
						if temp_chunk:
							chunks.append({"id": chunk_id, "text": temp_chunk.strip(), "type": "legal_text", "length": len(temp_chunk)})
							chunk_id += 1
						temp_chunk = word
					else:
						temp_chunk += " " + word
				if temp_chunk:
					chunks.append({"id": chunk_id, "text": temp_chunk.strip(), "type": "legal_text", "length": len(temp_chunk)})
					chunk_id += 1
			else:
				if len(current_chunk + " " + paragraph) <= chunk_size:
					current_chunk += " " + paragraph
				else:
					if current_chunk.strip():
						chunks.append({"id": chunk_id, "text": current_chunk.strip(), "type": "legal_text", "length": len(current_chunk)})
						chunk_id += 1
					current_chunk = paragraph
		if current_chunk.strip():
			chunks.append({"id": chunk_id, "text": current_chunk.strip(), "type": "legal_text", "length": len(current_chunk)})
		logger.info(f"Создано {len(chunks)} чанков из текста")
		return chunks
	
	def process_pdf_to_json(self, pdf_path: str) -> Dict[str, Any]:
		"""Обрабатывает PDF файл и создает структурированный JSON"""
		logger.info(f"Начинаю обработку файла: {pdf_path}")
		text = self.extract_text_from_pdf(pdf_path)
		if not text:
			logger.error(f"Не удалось извлечь текст из {pdf_path}")
			return {}
		metadata = self.extract_legal_metadata(text)
		legal_positions = self.extract_legal_positions(text)
		chunks = self.chunk_text(text)
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
		"""Сохраняет данные в JSON файл"""
		try:
			with open(output_path, 'w', encoding='utf-8') as f:
				json.dump(data, f, ensure_ascii=False, indent=2)
			logger.info(f"Данные сохранены в {output_path}")
		except Exception as e:
			logger.error(f"Ошибка при сохранении в {output_path}: {e}")
	
	def process_all_pdfs(self, input_dir: str = PDF_DIR, output_dir: str = JSON_DIR, force: bool = False):
		"""
		Обрабатывает все PDF файлы в директории
		- Если JSON уже существует (и новее PDF), пропускаем
		- Если force=True, пересобираем
		Возвращает сводку: { 'processed': int, 'skipped': int, 'errors': int, 'total': int, 'processed_files': [json_name,...] }
		"""
		if not os.path.exists(input_dir):
			logger.warning(f"Директория {input_dir} не существует")
			return { 'processed': 0, 'skipped': 0, 'errors': 0, 'total': 0, 'processed_files': [] }
		os.makedirs(output_dir, exist_ok=True)
		pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
		if not pdf_files:
			logger.warning(f"PDF файлы не найдены в {input_dir}")
			return { 'processed': 0, 'skipped': 0, 'errors': 0, 'total': 0, 'processed_files': [] }
		logger.info(f"Найдено {len(pdf_files)} PDF файлов для обработки")
		processed = 0
		skipped = 0
		errors = 0
		processed_files: List[str] = []
		for idx, pdf_file in enumerate(pdf_files, 1):
			pdf_path = os.path.join(input_dir, pdf_file)
			json_file = pdf_file.replace('.pdf', '.json')
			json_path = os.path.join(output_dir, json_file)
			if not force and os.path.exists(json_path):
				try:
					if os.path.getmtime(json_path) >= os.path.getmtime(pdf_path):
						skipped += 1
						if skipped % 1000 == 0:
							logger.info(f"Пропущено уже-конвертированных файлов: {skipped}")
						continue
				except Exception:
					pass
			try:
				data = self.process_pdf_to_json(pdf_path)
				if data:
					self.save_json(data, json_path)
					processed += 1
					processed_files.append(os.path.basename(json_path))
			except Exception as e:
				errors += 1
				logger.error(f"Ошибка при обработке {pdf_file}: {e}")
			if (processed + skipped) % 500 == 0:
				logger.info(f"Прогресс: обработано/пропущено {processed}/{skipped} из {len(pdf_files)}")
		logger.info(f"ИТОГО: обработано {processed}, пропущено {skipped}, ошибок {errors}, всего {len(pdf_files)}")
		return { 'processed': processed, 'skipped': skipped, 'errors': errors, 'total': len(pdf_files), 'processed_files': processed_files }


def main():
	"""Основная функция для тестирования"""
	processor = LegalDocumentProcessor()
	# Обрабатываем все PDF файлы (по умолчанию безопасный режим: пропуск уже обработанных)
	summary = processor.process_all_pdfs()
	logger.info(f"ИТОГО: обработано {summary['processed']}, пропущено {summary['skipped']}, ошибок {summary['errors']}, всего {summary['total']}")

if __name__ == "__main__":
	main()
