"""
Модуль интеграции с LLM: Gemini 2.5 Pro с резервом OpenAI GPT-5
Функции:
- compose_prompt: создать промпт из фактов и найденных документов
- generate_with_gemini: генерация через Gemini
- generate_with_openai: резервная генерация через OpenAI
- generate_legal_document: обертка с автоматическим фолбэком
"""
from typing import List, Dict, Any, Optional
import os
from loguru import logger
from dotenv import load_dotenv

import google.generativeai as genai
import openai

from config import GEMINI_API_KEY, OPENAI_API_KEY, GEMINI_MODEL, OPENAI_MODEL

load_dotenv()

# Configure clients
if GEMINI_API_KEY:
	genai.configure(api_key=GEMINI_API_KEY)
else:
	logger.warning("GEMINI_API_KEY is empty. Gemini calls will fail.")

if OPENAI_API_KEY:
	openai.api_key = OPENAI_API_KEY
else:
	logger.warning("OPENAI_API_KEY is empty. OpenAI fallback will fail.")

def compose_prompt(case_description: str, similar_docs: List[Dict[str, Any]], document_type: str = "исковое заявление") -> str:
	"""Формирует промпт для LLM на основе найденных документов с улучшенной фильтрацией.
	Ожидается, что similar_docs — это список словарей, как из VectorDatabase.search_similar.
	"""
	# Фильтруем документы по релевантности
	relevant_docs = []
	for doc in similar_docs[:10]:  # Берем топ-10
		meta = doc.get("metadata", {})
		dispute_type = meta.get("dispute_type", "")
		legal_area = meta.get("legal_area", "")
		
		# Приоритет потребительским спорам для данного случая
		if "потребитель" in case_description.lower() or "товар" in case_description.lower():
			if dispute_type == "consumer_protection" or "защита прав потребителей" in legal_area:
				relevant_docs.append(doc)
		else:
			# Для других типов споров используем все документы
			relevant_docs.append(doc)
	
	# Если не нашли специфичных документов, берем все
	if not relevant_docs:
		relevant_docs = similar_docs[:5]
	
	context_lines: List[str] = []
	for i, doc in enumerate(relevant_docs, 1):
		meta = doc.get("metadata", {})
		source = meta.get("source_file", "unknown")
		chunk_type = meta.get("chunk_type", "legal_text")
		legal_area = meta.get("legal_area", "")
		dispute_type = meta.get("dispute_type", "")
		
		# Добавляем информацию о типе спора в контекст
		type_info = f" ({legal_area}, {dispute_type})" if legal_area else ""
		
		context_lines.append(
			f"Дело {i} (источник: {source}, тип: {chunk_type}{type_info}):\n{doc.get('text','')[:1500]}"
		)
	
	context = "\n\n".join(context_lines) if context_lines else "Нет релевантного контекста"
	
	# Определяем тип спора из описания дела
	case_lower = case_description.lower()
	if any(keyword in case_lower for keyword in ["потребитель", "товар", "продавец", "недостаток", "качество"]):
		dispute_category = "потребительский спор"
		legal_framework = "Закон РФ 'О защите прав потребителей'"
	elif any(keyword in case_lower for keyword in ["договор", "контракт", "обязательство"]):
		dispute_category = "договорный спор"
		legal_framework = "Гражданский кодекс РФ"
	else:
		dispute_category = "гражданский спор"
		legal_framework = "соответствующее законодательство РФ"
	
	prompt = f"""
Ты — опытный юрист-процессуалист. Составь {document_type} с подробной мотивировочной частью.

ОБСТОЯТЕЛЬСТВА ДЕЛА:
{case_description}

ТИП СПОРА: {dispute_category}
ПРАВОВАЯ ОСНОВА: {legal_framework}

РЕЛЕВАНТНАЯ СУДЕБНАЯ ПРАКТИКА (из найденных документов):
{context}

ТРЕБОВАНИЯ К ОТВЕТУ:
1) Структура: вводная часть, обстоятельства, правовая квалификация, ссылки на нормы и позиции ВС РФ, просительная часть, приложения.
2) В мотивировочной части опирайся ТОЛЬКО на приведенные релевантные выдержки; цитируй нормы и правовые позиции явно.
3) Если найденные документы не относятся к данному типу спора, укажи это и используй общие принципы права.
4) Стиль официальный, без воды. Русский язык. Укажи ссылки на источники (файл/позиция), если это уместно.
5) Верни только финальный текст документа без пояснений.
"""
	return prompt.strip()


def generate_with_gemini(prompt: str) -> str:
	"""Генерация через Gemini."""
	logger.info("Запрос к Gemini...")
	model = genai.GenerativeModel(GEMINI_MODEL)
	resp = model.generate_content(prompt)
	return getattr(resp, "text", "") or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")


def generate_with_openai(prompt: str) -> str:
	"""Резервная генерация через OpenAI GPT-5."""
	logger.info("Фолбэк в OpenAI GPT-5...")
	# Унифицированный chat-completions стиль
	completion = openai.chat.completions.create(
		model=OPENAI_MODEL,
		messages=[
			{"role": "system", "content": "Ты — опытный юрист-процессуалист. Пиши строго структурированные процессуальные документы."},
			{"role": "user", "content": prompt},
		],
		temperature=0.2,
	)
	return completion.choices[0].message.content


def generate_legal_document(case_description: str,
							similar_docs: List[Dict[str, Any]],
							document_type: str = "исковое заявление") -> Dict[str, Any]:
	"""Собирает промпт, вызывает Gemini, при ошибке — OpenAI.
	Возвращает словарь с результатом и источником ("gemini" или "openai").
	"""
	prompt = compose_prompt(case_description, similar_docs, document_type)
	try:
		text = generate_with_gemini(prompt)
		if not text or len(text.strip()) < 50:
			raise RuntimeError("Пустой или слишком короткий ответ Gemini")
		return {"provider": "gemini", "document": text}
	except Exception as e:
		logger.error(f"Gemini ошибка: {e}. Перехожу на OpenAI.")
		try:
			text = generate_with_openai(prompt)
			return {"provider": "openai", "document": text}
		except Exception as e2:
			logger.error(f"OpenAI ошибка: {e2}")
			raise
