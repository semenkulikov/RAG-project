"""
Демonstration script:
1) Берет описание дела
2) Ищет похожие фрагменты в векторной БД (полной, при наличии)
3) Генерирует документ через Gemini, при ошибке — через OpenAI
"""

import os
from loguru import logger
from dotenv import load_dotenv

from config import JSON_DIR
from vector_database import VectorDatabase
from simple_vector_db import SimpleVectorDatabase
from gemini_integration import generate_legal_document

load_dotenv()


def main():
	case_description = (
		"Истец выдал заем ответчику. Имеется расписка. Срок возврата истек, деньги не возвращены. "
		"Ответчик заявляет о безденежности. Нужен проект искового заявления о взыскании задолженности."
	)
	
	# Пробуем полную БД, при ошибке — упрощенную
	similar_docs = []
	try:
		logger.info("Поиск похожих документов (полная БД)...")
		vdb = VectorDatabase()
		vdb.load_from_json_files(JSON_DIR)
		similar_docs = vdb.search_similar("договор займа расписка взыскание задолженности", n_results=5)
	except Exception as e:
		logger.error(f"Полная БД недоступна: {e}. Перехожу на упрощенную.")
		svdb = SimpleVectorDatabase()
		svdb.load_from_json_files(JSON_DIR)
		similar_docs = svdb.search_similar("договор займа расписка взыскание задолженности", n_results=5)
	
	logger.info(f"Найдено фрагментов: {len(similar_docs)}")
	
	result = generate_legal_document(case_description, similar_docs, document_type="исковое заявление")
	provider = result.get("provider")
	document = result.get("document", "")
	
	print("\n=== Провайдер ===")
	print(provider)
	print("\n=== Сгенерированный документ ===\n")
	print(document)


if __name__ == "__main__":
	main()
