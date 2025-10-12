#!/usr/bin/env python3
"""
Многопроцессная переиндексация с настоящим параллелизмом
"""

import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any
import shutil
from tqdm import tqdm

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.processors.data_processor import LegalDocumentProcessor
from src.databases.vector_database import VectorDatabase
from src.utils.config import PDF_DIR, JSON_DIR, CHROMA_DB_PATH
from loguru import logger

def process_single_pdf_worker(pdf_file: str) -> Dict[str, Any]:
    """Рабочая функция для обработки одного PDF файла"""
    try:
        start_time = time.time()
        
        # Создаем процессор в каждом процессе
        processor = LegalDocumentProcessor(use_gemini_chunking=True)
        
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
        
        # Обрабатываем файл
        data = processor.process_pdf_to_json(pdf_path)
        if data:
            processor.save_json(data, json_path)
            processing_time = time.time() - start_time
            
            return {
                'file': pdf_file,
                'status': 'processed',
                'chunks': data.get('processing_info', {}).get('total_chunks', 0),
                'positions': data.get('processing_info', {}).get('total_positions', 0),
                'error': None,
                'processing_time': processing_time
            }
        else:
            return {
                'file': pdf_file,
                'status': 'error',
                'chunks': 0,
                'positions': 0,
                'error': 'Не удалось обработать файл',
                'processing_time': time.time() - start_time
            }
            
    except Exception as e:
        return {
            'file': pdf_file,
            'status': 'error',
            'chunks': 0,
            'positions': 0,
            'error': str(e),
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0
        }

def clear_old_data():
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

def get_pdf_files() -> List[str]:
    """Получает список PDF файлов для обработки"""
    if not os.path.exists(PDF_DIR):
        logger.error(f"Директория {PDF_DIR} не существует")
        return []
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    logger.info(f"📄 Найдено {len(pdf_files)} PDF файлов для обработки")
    return pdf_files

def process_pdfs_multiprocess(pdf_files: List[str], num_processes: int = None) -> List[Dict[str, Any]]:
    """Обрабатывает PDF файлы в многопроцессном режиме"""
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), len(pdf_files))
    
    logger.info(f"🚀 Начинаю многопроцессную обработку {len(pdf_files)} файлов")
    logger.info(f"⚡ Используется {num_processes} процессов")
    
    start_time = time.time()
    
    # Создаем пул процессов
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Обрабатываем файлы с прогресс-баром
        results = []
        with tqdm(total=len(pdf_files), desc="📄 Обработка PDF", unit="файл") as pbar:
            for result in pool.imap(process_single_pdf_worker, pdf_files):
                results.append(result)
                pbar.update(1)
                
                # Обновляем прогресс-бар
                if result['status'] == 'processed':
                    pbar.set_postfix({
                        'chunks': result['chunks'],
                        'positions': result['positions'],
                        'time': f"{result['processing_time']:.1f}s"
                    })
                elif result['status'] == 'skipped':
                    pbar.set_postfix({'status': 'skipped'})
                else:
                    pbar.set_postfix({'status': 'error'})
    
    total_time = time.time() - start_time
    logger.info(f"⏱️ Общее время обработки: {total_time:.2f}с")
    
    return results

def load_to_vector_database():
    """Загружает обработанные JSON файлы в векторную базу"""
    logger.info("📚 Загружаю данные в векторную базу...")
    
    try:
        vector_db = VectorDatabase()
        vector_db.initialize_database()
        
        # Загружаем JSON файлы
        stats = vector_db.load_from_json_files()
        
        logger.info("✅ Векторная база обновлена:")
        logger.info(f"   📄 Новых файлов: {stats['new_files']}")
        logger.info(f"   📚 Загружено: {stats['loaded']}")
        logger.info(f"   ⏭️ Пропущено: {stats['skipped']}")
        logger.info(f"   ❌ Ошибок: {stats['errors']}")
        logger.info(f"   📊 Всего чанков: {stats['total_chunks']}")
        logger.info(f"   📍 Всего позиций: {stats['total_positions']}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке в векторную базу: {e}")
        return None

def print_statistics(results: List[Dict[str, Any]]):
    """Выводит статистику обработки"""
    processed = sum(1 for r in results if r['status'] == 'processed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    total_chunks = sum(r['chunks'] for r in results)
    total_positions = sum(r['positions'] for r in results)
    
    avg_time = sum(r['processing_time'] for r in results if r['processing_time'] > 0) / max(processed, 1)
    
    logger.info("📊 СТАТИСТИКА ОБРАБОТКИ:")
    logger.info(f"   ✅ Обработано: {processed}")
    logger.info(f"   ⏭️ Пропущено: {skipped}")
    logger.info(f"   ❌ Ошибок: {errors}")
    logger.info(f"   📄 Всего файлов: {len(results)}")
    logger.info(f"   📚 Всего чанков: {total_chunks}")
    logger.info(f"   📍 Всего позиций: {total_positions}")
    logger.info(f"   ⏱️ Среднее время на файл: {avg_time:.2f}с")
    
    if errors > 0:
        logger.warning("❌ Файлы с ошибками:")
        for result in results:
            if result['status'] == 'error':
                logger.warning(f"   - {result['file']}: {result['error']}")

def main():
    """Основная функция"""
    logger.info("🚀 Запуск многопроцессной переиндексации")
    
    # Очищаем старые данные
    clear_old_data()
    
    # Получаем список PDF файлов
    pdf_files = get_pdf_files()
    if not pdf_files:
        logger.error("PDF файлы не найдены")
        return
    
    # Обрабатываем PDF файлы в многопроцессном режиме
    results = process_pdfs_multiprocess(pdf_files)
    
    # Выводим статистику
    print_statistics(results)
    
    # Загружаем в векторную базу
    vector_stats = load_to_vector_database()
    
    if vector_stats:
        logger.info("🎉 Переиндексация завершена успешно!")
    else:
        logger.error("❌ Ошибка при загрузке в векторную базу")

if __name__ == "__main__":
    main()
