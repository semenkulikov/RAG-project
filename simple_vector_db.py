"""
Упрощенная векторная база данных для MVP
Использует простые эмбеддинги без загрузки больших моделей
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from loguru import logger
from config import CHROMA_DB_PATH, TOP_K_RESULTS

class SimpleVectorDatabase:
    """Упрощенная векторная база данных для MVP"""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH):
        """
        Инициализация упрощенной векторной базы данных
        
        Args:
            db_path: Путь к базе данных
        """
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Для русского языка
            ngram_range=(1, 2)
        )
        self.documents = []
        self.embeddings = None
        self.is_fitted = False
        
        self.setup_logging()
        self.initialize_database()
    
    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/simple_vector_db.log", rotation="1 MB", level="INFO")
    
    def initialize_database(self):
        """Инициализация базы данных"""
        try:
            # Создаем директорию если её нет
            os.makedirs(self.db_path, exist_ok=True)
            
            # Пытаемся загрузить существующие данные
            self.load_database()
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации базы данных: {e}")
    
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
            
            # Подготавливаем тексты
            texts = []
            metadatas = []
            
            for doc in documents:
                # Извлекаем текст из чанков
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
                
                # Добавляем правовые позиции отдельно
                if 'legal_positions' in doc:
                    for position in doc['legal_positions']:
                        texts.append(position['text'])
                        metadatas.append({
                            'source_file': doc.get('source_file', 'unknown'),
                            'case_number': doc.get('metadata', {}).get('case_number', ''),
                            'court': doc.get('metadata', {}).get('court', ''),
                            'document_type': doc.get('metadata', {}).get('document_type', ''),
                            'chunk_id': f"position_{len(texts)}",
                            'chunk_type': 'legal_position',
                            'articles': ', '.join(position.get('articles', []))
                        })
            
            if not texts:
                logger.warning("Нет текстов для векторизации")
                return
            
            # Добавляем к существующим документам
            self.documents.extend([{
                'text': text,
                'metadata': metadata
            } for text, metadata in zip(texts, metadatas)])
            
            # Пересоздаем эмбеддинги
            self._create_embeddings()
            
            # Сохраняем базу данных
            self.save_database()
            
            logger.info(f"Добавлено {len(texts)} документов в векторную базу")
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise
    
    def _create_embeddings(self):
        """Создает эмбеддинги для всех документов"""
        try:
            if not self.documents:
                logger.warning("Нет документов для векторизации")
                return
            
            texts = [doc['text'] for doc in self.documents]
            
            # Создаем TF-IDF эмбеддинги
            self.embeddings = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            
            logger.info(f"Создано {self.embeddings.shape[0]} эмбеддингов")
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = TOP_K_RESULTS, 
                      filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Ищет похожие документы по запросу
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            filter_metadata: Фильтр по метаданным (не реализован в упрощенной версии)
            
        Returns:
            Список похожих документов с метаданными
        """
        try:
            if not self.is_fitted or self.embeddings is None:
                logger.warning("База данных не инициализирована")
                return []
            
            # Создаем эмбеддинг для запроса
            query_embedding = self.vectorizer.transform([query])
            
            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            
            # Получаем индексы наиболее похожих документов
            top_indices = similarities.argsort()[-n_results:][::-1]
            
            # Формируем результат
            similar_docs = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Только документы с положительным сходством
                    similar_docs.append({
                        'text': self.documents[idx]['text'],
                        'metadata': self.documents[idx]['metadata'],
                        'similarity': float(similarities[idx]),
                        'id': f"doc_{idx}"
                    })
            
            logger.info(f"Найдено {len(similar_docs)} похожих документов")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Получает информацию о базе данных
        
        Returns:
            Информация о базе данных
        """
        return {
            'document_count': len(self.documents),
            'is_fitted': self.is_fitted,
            'embedding_shape': self.embeddings.shape if self.embeddings is not None else None,
            'db_path': self.db_path
        }
    
    def save_database(self):
        """Сохраняет базу данных на диск"""
        try:
            db_file = os.path.join(self.db_path, "simple_vector_db.pkl")
            
            data = {
                'documents': self.documents,
                'vectorizer': self.vectorizer,
                'embeddings': self.embeddings,
                'is_fitted': self.is_fitted
            }
            
            with open(db_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"База данных сохранена в {db_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении базы данных: {e}")
    
    def load_database(self):
        """Загружает базу данных с диска"""
        try:
            db_file = os.path.join(self.db_path, "simple_vector_db.pkl")
            
            if not os.path.exists(db_file):
                logger.info("База данных не найдена, создается новая")
                return
            
            with open(db_file, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', [])
            self.vectorizer = data.get('vectorizer', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)))
            self.embeddings = data.get('embeddings')
            self.is_fitted = data.get('is_fitted', False)
            
            logger.info(f"База данных загружена: {len(self.documents)} документов")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке базы данных: {e}")
    
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
    logger.info("Инициализация упрощенной векторной базы данных...")
    
    # Создаем векторную базу
    vector_db = SimpleVectorDatabase()
    
    # Загружаем документы из JSON файлов
    from config import JSON_DIR
    vector_db.load_from_json_files(JSON_DIR)
    
    # Получаем информацию о базе данных
    info = vector_db.get_database_info()
    logger.info(f"Информация о базе данных: {info}")
    
    # Тестируем поиск
    test_query = "договор займа расписка"
    results = vector_db.search_similar(test_query, n_results=3)
    
    logger.info(f"Результаты поиска по запросу '{test_query}':")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['text'][:100]}... (сходство: {result['similarity']:.4f})")

if __name__ == "__main__":
    main()
