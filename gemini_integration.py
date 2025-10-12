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
	"""Формирует специализированный промпт на основе образца Gemini 2.5 Pro"""
	
	# Фильтруем и структурируем документы
	relevant_docs = []
	for doc in similar_docs[:8]:  # Берем топ-8 для качества
		meta = doc.get("metadata", {})
		dispute_type = meta.get("dispute_type", "")
		legal_area = meta.get("legal_area", "")
		
		# Приоритет потребительским спорам
		if "потребитель" in case_description.lower() or "товар" in case_description.lower():
			if dispute_type == "consumer_protection" or "защита прав потребителей" in legal_area:
				relevant_docs.append(doc)
		else:
			relevant_docs.append(doc)
	
	if not relevant_docs:
		relevant_docs = similar_docs[:5]
	
	# Структурируем контекст по типам
	context_sections = {
		"factual_circumstances": [],
		"legal_positions": [],
		"citations": [],
		"conclusions": []
	}
	
	for i, doc in enumerate(relevant_docs, 1):
		meta = doc.get("metadata", {})
		chunk_type = meta.get("chunk_type", "legal_text")
		source = meta.get("source_file", "unknown")
		
		# Классифицируем по типу чанка
		doc_text = doc.get('text', '')[:1200]
		if chunk_type == "factual_circumstances":
			context_sections["factual_circumstances"].append(f"Контекст {i} (из {source}):\n{doc_text}")
		elif chunk_type == "legal_position":
			context_sections["legal_positions"].append(f"Правовая позиция {i} (из {source}):\n{doc_text}")
		elif "ст." in doc_text or "статья" in doc_text.lower():
			context_sections["citations"].append(f"Цитата {i} (из {source}):\n{doc_text}")
		else:
			context_sections["conclusions"].append(f"Вывод {i} (из {source}):\n{doc_text}")
	
	# Формируем структурированный контекст
	context_parts = []
	if context_sections["factual_circumstances"]:
		context_parts.append("ФАКТИЧЕСКИЕ ОБСТОЯТЕЛЬСТВА ИЗ СУДЕБНОЙ ПРАКТИКИ:\n" + "\n\n".join(context_sections["factual_circumstances"]))
	if context_sections["legal_positions"]:
		context_parts.append("ПРАВОВЫЕ ПОЗИЦИИ ВЕРХОВНОГО СУДА РФ:\n" + "\n\n".join(context_sections["legal_positions"]))
	if context_sections["citations"]:
		context_parts.append("ЦИТАТЫ ИЗ ЗАКОНОВ И ПОСТАНОВЛЕНИЙ:\n" + "\n\n".join(context_sections["citations"]))
	if context_sections["conclusions"]:
		context_parts.append("ВЫВОДЫ И РЕШЕНИЯ СУДОВ:\n" + "\n\n".join(context_sections["conclusions"]))
	
	context = "\n\n".join(context_parts) if context_parts else "Нет релевантного контекста"
	
	# Определяем тип спора
	case_lower = case_description.lower()
	if any(keyword in case_lower for keyword in ["потребитель", "товар", "продавец", "недостаток", "качество"]):
		dispute_category = "потребительский спор"
		legal_framework = "Закон РФ 'О защите прав потребителей'"
	else:
		dispute_category = "гражданский спор"
		legal_framework = "соответствующее законодательство РФ"
	
	prompt = f"""
Ты — элитный юрист-аналитик, специализирующийся на составлении процессуальных документов на основе судебной практики Верховного Суда РФ.

ЗАДАЧА: Подготовить проект {document_type} на основе следующих фактических обстоятельств.

ФАКТИЧЕСКИЕ ОБСТОЯТЕЛЬСТВА КЛИЕНТА:
{case_description}

ТИП СПОРА: {dispute_category}
ПРАВОВАЯ ОСНОВА: {legal_framework}

АНАЛИТИЧЕСКАЯ СПРАВКА (релевантные правовые позиции из базы данных ВС РФ):
{context}

ИНСТРУКЦИИ:
1. Внимательно изучи обстоятельства клиента и предоставленный контекст.
2. Используй ТОЛЬКО информацию из предоставленной аналитической справки для формирования правовой позиции. Не придумывай ничего от себя и не используй свои общие знания.
3. Критически оцени релевантность каждого контекста. Если какой-то из найденных фрагментов НЕ относится к делу, проигнорируй его и укажи в своих рассуждениях, почему ты его проигнорил.
4. Напиши структурированное {document_type}, явно цитируя наиболее подходящие правовые позиции из контекста и ссылаясь на номера дел.
5. В мотивировочной части максимально подробно раскрой суть нарушенного права и надлежащий способ его защиты судом.
6. Объясни, что должен сделать суд и чем это предусмотрено законом.
7. Стиль: официальный, убедительный, без воды. Русский язык.
8. Верни только финальный текст документа без пояснений.

СТРУКТУРА ДОКУМЕНТА:
- Вводная часть (стороны, предмет спора)
- Фактические обстоятельства дела
- Правовое обоснование с цитатами из ВС РФ
- Просительная часть
- Приложения
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
