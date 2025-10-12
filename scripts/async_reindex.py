#!/usr/bin/env python3
"""
Асинхронная переиндексация с прогресс-баром для больших корпусов документов
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import shutil

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.processors.data_processor import LegalDocumentProcessor
from src.databases.vector_database import VectorDatabase
from src.utils.config import PDF_DIR, JSON_DIR, CHROMA_DB_PATH
from loguru import logger
from tqdm.asyncio import tqdm

class AsyncReindexer:
    """Асинхронный переиндексатор с прогресс-баром"""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.processor = LegalDocumentProcessor(use_gemini_chunking=True)
        self.vector_db = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def clear_old_data(self):
        """Очищает старые данные"""
        logger.info("🗑️ Очищаю старые данные...")
        
        # Очищаем векторную базу
        if os.path.exists(CHROMA_DB_PATH):
            logger.info(f"🗑️ Удаляю старую векторную базу: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)
        
        # Очищаем JSON файлы
        if os.path.exists(JSON_DIR):
            logger.info(f"🗑️ Удаляю старые JSON файлы: {JSON_DIR}")
            shutil.rmtree(JSON_DIR)
            os.makedirs(JSON_DIR, exist_ok=True)
    
    async def get_pdf_files(self) -> List[str]:
        """Получает список PDF файлов для обработки"""
        if not os.path.exists(PDF_DIR):
            logger.error(f"Директория {PDF_DIR} не существует")
            return []
        
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        logger.info(f"📄 Найдено {len(pdf_files)} PDF файлов для обработки")
        return pdf_files
    
    async def process_single_pdf(self, pdf_file: str) -> Dict[str, Any]:
        """Асинхронно обрабатывает один PDF файл"""
        start_time = time.time()
        try:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            json_file = pdf_file.replace('.pdf', '.json')
            json_path = os.path.join(JSON_DIR, json_file)
            
            # Проверяем, нужно ли обрабатывать файл
            if os.path.exists(json_path):
                try:
                    if os.path.getmtime(json_path) >= os.path.getmtime(pdf_path):
                        return {
                            'file': pdf_file,
                            'status': 'skipped',
                            'chunks': 0,
                            'positions': 0,
                            'error': None,
                            'processing_time': 0
                        }
                except OSError:
                    pass  # Файл поврежден, переобрабатываем
            
            # Обрабатываем файл в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_pdf_sync,
                pdf_path
            )
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            logger.info(f"✅ {pdf_file} обработан за {processing_time:.2f}с")
            
            return {
                'file': pdf_file,
                'status': 'processed',
                'chunks': result.get('processing_info', {}).get('total_chunks', 0),
                'positions': result.get('processing_info', {}).get('total_positions', 0),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_file}: {e}")
            return {
                'file': pdf_file,
                'status': 'error',
                'chunks': 0,
                'positions': 0,
                'error': str(e)
            }
    
    def _process_pdf_sync(self, pdf_path: str) -> Dict[str, Any]:
        """Синхронная обработка PDF файла"""
        return self.processor.process_pdf_to_json(pdf_path)
    
    async def process_all_pdfs_async(self) -> Dict[str, Any]:
        """Асинхронно обрабатывает все PDF файлы"""
        pdf_files = await self.get_pdf_files()
        
        if not pdf_files:
            return {
                'processed': 0,
                'skipped': 0,
                'errors': 0,
                'total_chunks': 0,
                'total_positions': 0,
                'processed_files': []
            }
        
        logger.info(f"🚀 Начинаю асинхронную обработку {len(pdf_files)} файлов")
        logger.info(f"⚡ Используется {self.max_workers} параллельных потоков")
        
        # Создаем задачи для асинхронной обработки
        tasks = [self.process_single_pdf(pdf_file) for pdf_file in pdf_files]
        
        # Обрабатываем с прогресс-баром
        results = []
        with tqdm(total=len(tasks), desc="📄 Обработка PDF", unit="файл") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                
                # Обновляем прогресс-бар
                if result['status'] == 'processed':
                    pbar.set_postfix({
                        'chunks': result['chunks'],
                        'positions': result['positions']
                    })
                elif result['status'] == 'skipped':
                    pbar.set_postfix({'status': 'skipped'})
                elif result['status'] == 'error':
                    pbar.set_postfix({'status': 'error'})
                
                pbar.update(1)
        
        # Подсчитываем статистику
        processed = sum(1 for r in results if r['status'] == 'processed')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        errors = sum(1 for r in results if r['status'] == 'error')
        total_chunks = sum(r['chunks'] for r in results)
        total_positions = sum(r['positions'] for r in results)
        processed_files = [r['file'] for r in results if r['status'] == 'processed']
        
        logger.info(f"✅ Обработка завершена:")
        logger.info(f"   📄 Обработано: {processed} файлов")
        logger.info(f"   ⏭️ Пропущено: {skipped} файлов")
        logger.info(f"   ❌ Ошибок: {errors} файлов")
        logger.info(f"   📊 Всего чанков: {total_chunks}")
        logger.info(f"   📊 Всего позиций: {total_positions}")
        
        return {
            'processed': processed,
            'skipped': skipped,
            'errors': errors,
            'total_chunks': total_chunks,
            'total_positions': total_positions,
            'processed_files': processed_files
        }
    
    async def create_vector_database(self):
        """Создает векторную базу данных"""
        logger.info("🔍 Создаю векторную базу данных...")
        
        loop = asyncio.get_event_loop()
        self.vector_db = await loop.run_in_executor(
            self.executor,
            VectorDatabase
        )
        
        # Загружаем JSON файлы в векторную базу
        json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
        
        if not json_files:
            logger.warning("JSON файлы не найдены")
            return
        
        logger.info(f"📚 Загружаю {len(json_files)} JSON файлов в векторную базу...")
        
        # Загружаем файлы с прогресс-баром
        with tqdm(total=len(json_files), desc="📚 Загрузка в векторную БД", unit="файл") as pbar:
            for json_file in json_files:
                json_path = os.path.join(JSON_DIR, json_file)
                
                try:
                    await loop.run_in_executor(
                        self.executor,
                        self._load_single_json,
                        json_path
                    )
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Ошибка загрузки {json_file}: {e}")
                    pbar.update(1)
        
        logger.info("✅ Векторная база данных создана")
    
    def _load_single_json(self, json_path: str):
        """Загружает один JSON файл в векторную базу"""
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Добавляем документ в векторную базу
        self.vector_db.add_documents([data])
    
    async def test_search(self):
        """Тестирует поиск в векторной базе"""
        if not self.vector_db:
            logger.error("Векторная база не создана")
            return
        
        logger.info("🔍 Тестирую поиск в векторной базе...")
        
        test_queries = [
            "защита прав потребителей",
            "расторжение договора",
            "недостаток товара",
            "административная ответственность"
        ]
        
        for query in test_queries:
            try:
                results = self.vector_db.search_similar(query, n_results=3)
                logger.info(f"🔍 Запрос: '{query}' -> найдено {len(results)} результатов")
                
                for i, result in enumerate(results[:2], 1):
                    logger.info(f"   {i}. {result.get('title', 'Без названия')[:50]}...")
                    
            except Exception as e:
                logger.error(f"Ошибка поиска для '{query}': {e}")
    
    async def cleanup(self):
        """Очищает ресурсы"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def run_full_reindex(self):
        """Запускает полную переиндексацию"""
        start_time = time.time()
        
        try:
            # 1. Очищаем старые данные
            await self.clear_old_data()
            
            # 2. Обрабатываем PDF файлы
            pdf_results = await self.process_all_pdfs_async()
            
            # 3. Создаем векторную базу
            await self.create_vector_database()
            
            # 4. Тестируем поиск
            await self.test_search()
            
            # 5. Выводим итоговую статистику
            total_time = time.time() - start_time
            logger.info("🎉 Переиндексация завершена успешно!")
            logger.info(f"⏱️ Общее время: {total_time:.2f} секунд")
            logger.info(f"📊 Статистика:")
            logger.info(f"   📄 Обработано файлов: {pdf_results['processed']}")
            logger.info(f"   📊 Создано чанков: {pdf_results['total_chunks']}")
            logger.info(f"   📊 Извлечено позиций: {pdf_results['total_positions']}")
            logger.info(f"   ⚡ Скорость: {pdf_results['processed']/total_time:.2f} файлов/сек")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при переиндексации: {e}")
            raise
        finally:
            await self.cleanup()

async def main():
    """Основная функция"""
    logger.info("🚀 Запуск асинхронной переиндексации")
    
    # Определяем количество потоков на основе CPU
    import multiprocessing
    max_workers = min(multiprocessing.cpu_count(), 8)  # Максимум 8 потоков
    
    reindexer = AsyncReindexer(max_workers=max_workers)
    
    try:
        await reindexer.run_full_reindex()
    except KeyboardInterrupt:
        logger.info("⏹️ Переиндексация прервана пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await reindexer.cleanup()

if __name__ == "__main__":
    # Настраиваем логирование
    logger.add("./logs/async_reindex.log", rotation="10 MB", level="INFO")
    
    # Запускаем асинхронную переиндексацию
    asyncio.run(main())
