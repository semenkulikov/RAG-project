"""
Модуль для работы с векторной базой данных
Использует ChromaDB для хранения и поиска векторных представлений документов
"""

import os
import json
from typing import List, Dict, Any, Optional, Iterable
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import CHROMA_DB_PATH, VECTOR_COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_RESULTS

# Параметры производительности (можно подстроить под железо)
INGEST_MANIFEST = os.path.join(CHROMA_DB_PATH, "ingested_files.txt")
DEFAULT_TEXTS_BATCH = 2000  # размер партии текстов для эмбеддинга (увеличено)
DEFAULT_FILES_BATCH = 50    # количество JSON файлов на партию (увеличено)

# Задействуем все доступные ядра BLAS/МКL (если не выставлено снаружи)
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() or 8)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, os.cpu_count() or 8)))
try:
	import torch
	torch.set_num_threads(max(1, os.cpu_count() or 8))
	_HAS_TORCH = True
except Exception:
	_HAS_TORCH = False

device_str = "cpu"
if _HAS_TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():
	device_str = "cuda"

class VectorDatabase:
	"""Класс для работы с векторной базой данных"""
	
	def __init__(self, db_path: str = CHROMA_DB_PATH):
		self.db_path = db_path
		self.embedding_model = None
		self.client = None
		self.collection = None
		self._ingested: set[str] = set()
		self.setup_logging()
		self.initialize_database()
		self._load_ingest_manifest()
		self.load_embedding_model()
	
	def setup_logging(self):
		logger.add("./logs/vector_database.log", rotation="1 MB", level="INFO")
	
	def initialize_database(self):
		try:
			os.makedirs(self.db_path, exist_ok=True)
			self.client = chromadb.PersistentClient(
				path=self.db_path,
				settings=Settings(
					anonymized_telemetry=False,
					allow_reset=True
				)
			)
			try:
				self.collection = self.client.get_collection(name=VECTOR_COLLECTION_NAME)
				logger.info(f"Загружена существующая коллекция: {VECTOR_COLLECTION_NAME}")
			except Exception:
				self.collection = self.client.create_collection(
					name=VECTOR_COLLECTION_NAME,
					metadata={"description": "Коллекция юридических документов для RAG системы"}
				)
				logger.info(f"Создана новая коллекция: {VECTOR_COLLECTION_NAME}")
		except Exception as e:
			logger.error(f"Ошибка при инициализации базы данных: {e}")
			raise
	
	def _load_ingest_manifest(self):
		try:
			if os.path.exists(INGEST_MANIFEST):
				with open(INGEST_MANIFEST, 'r', encoding='utf-8') as f:
					self._ingested = set(line.strip() for line in f if line.strip())
				logger.info(f"Манифест: уже загружено файлов: {len(self._ingested)}")
		except Exception as e:
			logger.warning(f"Не удалось загрузить манифест: {e}")
			self._ingested = set()
	
	def _append_ingested(self, filename: str):
		try:
			with open(INGEST_MANIFEST, 'a', encoding='utf-8') as f:
				f.write(filename + "\n")
			self._ingested.add(filename)
		except Exception as e:
			logger.warning(f"Не удалось обновить манифест {filename}: {e}")
	
	def load_embedding_model(self):
		try:
			logger.info(f"Загружаю модель эмбеддингов: {EMBEDDING_MODEL} (device={device_str})")
			# sentence-transformers поддерживает параметр device
			self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device_str)
			logger.info("Модель эмбеддингов загружена успешно")
		except Exception as e:
			logger.error(f"Ошибка при загрузке модели эмбеддингов: {e}")
			raise
	
	def _encode_batch(self, texts: List[str]) -> List[List[float]]:
		# Увеличенные батчи, разные для CPU/GPU
		batch_size = 256 if device_str == "cuda" else 128
		embeddings = self.embedding_model.encode(
			texts,
			batch_size=batch_size,
			convert_to_tensor=False,
			show_progress_bar=False,
			normalize_embeddings=False,
		)
		return embeddings.tolist()
	
	def add_documents(self, documents: List[Dict[str, Any]]):
		"""Добавляет документы в коллекцию батчами."""
		if not documents:
			logger.warning("Нет документов для добавления")
			return
		texts: List[str] = []
		metadatas: List[Dict[str, Any]] = []
		ids: List[str] = []
		for doc in documents:
			if 'chunks' in doc:
				for chunk in doc['chunks']:
					texts.append(chunk['text'])
					metadatas.append({
						'source_file': doc.get('source_file', 'unknown'),
						'case_number': doc.get('metadata', {}).get('case_number', ''),
						'court': doc.get('metadata', {}).get('court', ''),
						'document_type': doc.get('metadata', {}).get('document_type', ''),
						'chunk_id': chunk['id'],
						'chunk_type': chunk.get('type', 'legal_text')
					})
					ids.append(f"{doc.get('source_file', 'unknown')}_{chunk['id']}")
			if 'legal_positions' in doc:
				for j, position in enumerate(doc['legal_positions']):
					texts.append(position['text'])
					metadatas.append({
						'source_file': doc.get('source_file', 'unknown'),
						'case_number': doc.get('metadata', {}).get('case_number', ''),
						'court': doc.get('metadata', {}).get('court', ''),
						'document_type': doc.get('metadata', {}).get('document_type', ''),
						'chunk_id': f"position_{j}",
						'chunk_type': 'legal_position',
						'articles': ', '.join(position.get('articles', []))
					})
					ids.append(f"{doc.get('source_file', 'unknown')}_position_{j}")
		if not texts:
			logger.warning("Нет текстов для векторизации")
			return
		added = 0
		for start in range(0, len(texts), DEFAULT_TEXTS_BATCH):
			end = min(start + DEFAULT_TEXTS_BATCH, len(texts))
			batch_texts = texts[start:end]
			batch_metadatas = metadatas[start:end]
			batch_ids = ids[start:end]
			embeddings = self._encode_batch(batch_texts)
			self.collection.add(embeddings=embeddings, documents=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
			added += len(batch_texts)
			if added % 5000 == 0:
				logger.info(f"В коллекцию добавлено {added}/{len(texts)} фрагментов")
		logger.info(f"Добавлено {len(texts)} документов в векторную базу")
	
	def search_similar(self, query: str, n_results: int = TOP_K_RESULTS, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
		try:
			query_embedding = self._encode_batch([query])[0]
			results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results, where=filter_metadata)
			similar_docs: List[Dict[str, Any]] = []
			for i in range(len(results['documents'][0])):
				similar_docs.append({
					'text': results['documents'][0][i],
					'metadata': results['metadatas'][0][i],
					'distance': results['distances'][0][i],
					'id': results['ids'][0][i]
				})
			logger.info(f"Найдено {len(similar_docs)} похожих документов")
			return similar_docs
		except Exception as e:
			logger.error(f"Ошибка при поиске: {e}")
			raise
	
	def get_collection_info(self) -> Dict[str, Any]:
		try:
			count = self.collection.count()
			return {'name': VECTOR_COLLECTION_NAME, 'document_count': count, 'db_path': self.db_path}
		except Exception as e:
			logger.error(f"Ошибка при получении информации о коллекции: {e}")
			return {}
	
	def clear_collection(self):
		try:
			self.client.delete_collection(name=VECTOR_COLLECTION_NAME)
			self.collection = self.client.create_collection(name=VECTOR_COLLECTION_NAME, metadata={"description": "Коллекция юридических документов для RAG системы"})
			logger.info("Коллекция очищена")
		except Exception as e:
			logger.error(f"Ошибка при очистке коллекции: {e}")
			raise
	
	def load_from_json_files(self, json_dir: str, files_batch: int = DEFAULT_FILES_BATCH):
		"""
		Загружает документы из JSON в батчах с резюмом.
		- Пропускает JSON файлы, которые уже были загружены ранее (манифест)
		- Обрабатывает партиями, чтобы не держать всё в памяти
		Возвращает сводку: { 'new_files': [...], 'loaded_files': int, 'skipped': int, 'errors': int, 'total': int }
		"""
		if not os.path.exists(json_dir):
			logger.warning(f"Директория {json_dir} не существует")
			return { 'new_files': [], 'loaded_files': 0, 'skipped': 0, 'errors': 0, 'total': 0 }
		json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
		if not json_files:
			logger.warning(f"JSON файлы не найдены в {json_dir}")
			return { 'new_files': [], 'loaded_files': 0, 'skipped': 0, 'errors': 0, 'total': 0 }
		logger.info(f"К загрузке JSON файлов: всего {len(json_files)}")
		batch_docs: List[Dict[str, Any]] = []
		loaded_files = 0
		errors = 0
		skipped = 0
		new_files: List[str] = []
		for idx, json_file in enumerate(json_files, 1):
			if json_file in self._ingested:
				skipped += 1
				continue
			json_path = os.path.join(json_dir, json_file)
			try:
				with open(json_path, 'r', encoding='utf-8') as f:
					doc_data = json.load(f)
				batch_docs.append(doc_data)
				new_files.append(json_file)
				loaded_files += 1
				if loaded_files % files_batch == 0:
					logger.info(f"Добавление партии: файлов {files_batch}, прогресс файлов {idx}/{len(json_files)}")
					self.add_documents(batch_docs)
					for d in batch_docs:
						self._append_ingested(d.get('source_file', json_file))
					batch_docs = []
			except Exception as e:
				errors += 1
				logger.error(f"Ошибка при загрузке {json_file}: {e}")
		if batch_docs:
			logger.info(f"Добавление финальной партии: файлов {len(batch_docs)}")
			self.add_documents(batch_docs)
			for d in batch_docs:
				self._append_ingested(d.get('source_file', 'unknown'))
		logger.info(f"Сводка загрузки: новых файлов {loaded_files}, пропущено {skipped}, ошибок {errors}, всего {len(json_files)}")
		return { 'new_files': new_files, 'loaded_files': loaded_files, 'skipped': skipped, 'errors': errors, 'total': len(json_files) }


def main():
	logger.info("Инициализация векторной базы данных...")
	vector_db = VectorDatabase()
	from config import JSON_DIR
	vector_db.load_from_json_files(JSON_DIR)
	info = vector_db.get_collection_info()
	logger.info(f"Информация о коллекции: {info}")

if __name__ == "__main__":
	main()
