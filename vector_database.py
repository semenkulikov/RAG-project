"""
Модуль для работы с векторной базой данных
Использует ChromaDB для хранения и поиска векторных представлений документов
"""

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import CHROMA_DB_PATH, VECTOR_COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_RESULTS

class VectorDatabase:
    """Класс для работы с векторной базой данных"""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH):
        """
        Инициализация векторной базы данных
        
        Args:
            db_path: Путь к базе данных ChromaDB
        """
        self.db_path = db_path
        self.embedding_model = None
        self.client = None
        self.collection = None
        
        self.setup_logging()
        self.initialize_database()
        self.load_embedding_model()
    
    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/vector_database.log", rotation="1 MB", level="INFO")
    
    def initialize_database(self):
        """Инициализация ChromaDB"""
        try:
            # Создаем директорию если её нет
            os.makedirs(self.db_path, exist_ok=True)
            
            # Инициализируем клиент ChromaDB
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Получаем или создаем коллекцию
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
    
    def load_embedding_model(self):
        """Загрузка модели для создания эмбеддингов"""
        try:
            logger.info(f"Загружаю модель эмбеддингов: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Модель эмбеддингов загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели эмбеддингов: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Список векторов
        """
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            logger.info(f"Создано {len(embeddings)} эмбеддингов")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Добавляет документы в векторную базу данных
        
        Args:
            documents: Список документов с метаданными
        """
        try:
            if not documents:
                logger.warning("Нет документов для добавления")
                return
            
            # Подготавливаем данные
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                # Извлекаем текст из чанков
                if 'chunks' in doc:
                    for j, chunk in enumerate(doc['chunks']):
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
                
                # Добавляем правовые позиции отдельно
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
            
            # Создаем эмбеддинги
            embeddings = self.create_embeddings(texts)
            
            # Добавляем в коллекцию
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Добавлено {len(texts)} документов в векторную базу")
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = TOP_K_RESULTS, 
                      filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Ищет похожие документы по запросу
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            filter_metadata: Фильтр по метаданным
            
        Returns:
            Список похожих документов с метаданными
        """
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.create_embeddings([query])[0]
            
            # Выполняем поиск
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Формируем результат
            similar_docs = []
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
        """
        Получает информацию о коллекции
        
        Returns:
            Информация о коллекции
        """
        try:
            count = self.collection.count()
            return {
                'name': VECTOR_COLLECTION_NAME,
                'document_count': count,
                'db_path': self.db_path
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            return {}
    
    def clear_collection(self):
        """Очищает коллекцию"""
        try:
            self.client.delete_collection(name=VECTOR_COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=VECTOR_COLLECTION_NAME,
                metadata={"description": "Коллекция юридических документов для RAG системы"}
            )
            logger.info("Коллекция очищена")
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}")
            raise
    
    def load_from_json_files(self, json_dir: str):
        """
        Загружает документы из JSON файлов
        
        Args:
            json_dir: Директория с JSON файлами
        """
        try:
            if not os.path.exists(json_dir):
                logger.warning(f"Директория {json_dir} не существует")
                return
            
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            
            if not json_files:
                logger.warning(f"JSON файлы не найдены в {json_dir}")
                return
            
            documents = []
            for json_file in json_files:
                json_path = os.path.join(json_dir, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                        documents.append(doc_data)
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {json_file}: {e}")
            
            if documents:
                self.add_documents(documents)
                logger.info(f"Загружено {len(documents)} документов из JSON файлов")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке из JSON файлов: {e}")
            raise

def main():
    """Основная функция для тестирования"""
    logger.info("Инициализация векторной базы данных...")
    
    # Создаем векторную базу
    vector_db = VectorDatabase()
    
    # Загружаем документы из JSON файлов
    from config import JSON_DIR
    vector_db.load_from_json_files(JSON_DIR)
    
    # Получаем информацию о коллекции
    info = vector_db.get_collection_info()
    logger.info(f"Информация о коллекции: {info}")
    
    # Тестируем поиск
    test_query = "договор займа расписка"
    results = vector_db.search_similar(test_query, n_results=3)
    
    logger.info(f"Результаты поиска по запросу '{test_query}':")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['text'][:100]}... (расстояние: {result['distance']:.4f})")

if __name__ == "__main__":
    main()
