"""
Модуль для семантического чанкования документов через ChatGPT API
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import openai
from loguru import logger
from config import OPENAI_API_KEY

# Настройка OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY не установлен. ChatGPT чанкование недоступно.")

class ChatGPTChunker:
    """Класс для семантического чанкования через ChatGPT"""
    
    def __init__(self):
        self.setup_logging()
        self.cache_file = "./data/chunking_cache.json"
        self.cache = self.load_cache()
        
    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/chatgpt_chunker.log", rotation="1 MB", level="INFO")
    
    def load_cache(self) -> Dict[str, Any]:
        """Загружает кэш результатов чанкования"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить кэш: {e}")
        return {}
    
    def save_cache(self):
        """Сохраняет кэш результатов чанкования"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")
    
    def get_cache_key(self, text: str) -> str:
        """Генерирует ключ кэша для текста"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def chunk_document(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Разбивает документ на семантические чанки через ChatGPT
        
        Args:
            text: Текст документа
            source_file: Имя исходного файла
            
        Returns:
            Список чанков с метаданными
        """
        # Проверяем кэш
        cache_key = self.get_cache_key(text)
        if cache_key in self.cache:
            logger.info(f"Используем кэшированный результат для {source_file}")
            return self.cache[cache_key]
        
        if not OPENAI_API_KEY:
            logger.error("OpenAI API ключ не установлен")
            return self.fallback_chunking(text, source_file)
        
        try:
            # Если документ слишком большой, разбиваем на части
            max_tokens = 100000  # Ограничение для GPT-4
            if len(text) > max_tokens:
                return self.chunk_large_document(text, source_file)
            
            prompt = self.create_chunking_prompt(text)
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "Ты — эксперт по анализу юридических документов. Твоя задача — разбить судебное решение на семантически завершенные блоки."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            chunks = self.parse_chunking_result(result, source_file)
            
            # Сохраняем в кэш
            self.cache[cache_key] = chunks
            self.save_cache()
            
            logger.info(f"Обработано {len(chunks)} чанков для {source_file}")
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка при чанковании через ChatGPT: {e}")
            return self.fallback_chunking(text, source_file)
    
    def chunk_large_document(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Обрабатывает большие документы по частям"""
        logger.info(f"Обрабатываем большой документ {source_file} по частям")
        
        # Разбиваем на части по 50k символов с перекрытием
        chunk_size = 50000
        overlap = 5000
        all_chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            part = text[i:i + chunk_size]
            part_chunks = self.chunk_document(part, f"{source_file}_part_{i//chunk_size}")
            all_chunks.extend(part_chunks)
        
        return all_chunks
    
    def create_chunking_prompt(self, text: str) -> str:
        """Создает промпт для чанкования"""
        return f"""
Проанализируй следующий судебный документ и разбей его на семантически завершенные блоки.

ТРЕБОВАНИЯ:
1. Каждый чанк должен содержать одну законченную правовую позицию или аргумент
2. Сохраняй контекст - не разрывай связанные факты и выводы
3. Выделяй отдельно:
   - Фактические обстоятельства дела
   - Правовые позиции суда
   - Цитаты из законов и постановлений
   - Выводы и решения
4. Размер чанка: 200-800 слов (оптимально для векторного поиска)

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "chunks": [
    {{
      "id": "chunk_1",
      "type": "factual_circumstances|legal_position|citation|conclusion",
      "title": "Краткое название блока",
      "text": "Полный текст чанка",
      "key_articles": ["ст. 18 ЗоЗПП", "ст. 15 ГК РФ"],
      "legal_concepts": ["недостаток товара", "права потребителя"]
    }}
  ]
}}

ДОКУМЕНТ:
{text[:80000]}  # Ограничиваем размер для API
"""
    
    def parse_chunking_result(self, result: str, source_file: str) -> List[Dict[str, Any]]:
        """Парсит результат чанкования от ChatGPT"""
        try:
            # Извлекаем JSON из ответа
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                chunks = data.get('chunks', [])
            else:
                # Fallback: разбиваем по абзацам
                chunks = self.fallback_chunking(result, source_file)
                return chunks
            
            # Добавляем метаданные
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'id': f"{source_file}_chunk_{i}",
                    'type': chunk.get('type', 'legal_text'),
                    'title': chunk.get('title', ''),
                    'text': chunk.get('text', ''),
                    'key_articles': chunk.get('key_articles', []),
                    'legal_concepts': chunk.get('legal_concepts', []),
                    'source_file': source_file,
                    'chunk_index': i
                })
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге результата чанкования: {e}")
            return self.fallback_chunking(result, source_file)
    
    def fallback_chunking(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Резервное чанкование при ошибках API"""
        logger.warning(f"Используем резервное чанкование для {source_file}")
        
        # Простое разбиение по абзацам
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 100:  # Только значимые абзацы
                chunks.append({
                    'id': f"{source_file}_fallback_{i}",
                    'type': 'legal_text',
                    'title': f"Абзац {i+1}",
                    'text': paragraph.strip(),
                    'key_articles': [],
                    'legal_concepts': [],
                    'source_file': source_file,
                    'chunk_index': i
                })
        
        return chunks
    
    def estimate_cost(self, text: str) -> Dict[str, float]:
        """Оценивает стоимость обработки текста"""
        # Примерная оценка токенов (1 токен ≈ 4 символа)
        input_tokens = len(text) / 4
        output_tokens = 1000  # Примерный размер ответа
        
        # Цены GPT-4 (на момент написания)
        input_cost_per_1k = 0.03
        output_cost_per_1k = 0.06
        
        total_cost = (input_tokens / 1000 * input_cost_per_1k + 
                     output_tokens / 1000 * output_cost_per_1k)
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost_usd': total_cost
        }

def main():
    """Тестирование модуля"""
    chunker = ChatGPTChunker()
    
    # Тестовый текст
    test_text = """
    ВЕРХОВНЫЙ СУД РОССИЙСКОЙ ФЕДЕРАЦИИ
    ОПРЕДЕЛЕНИЕ
    от 15 августа 2023 г. № 44-КГ23-11-К7
    
    Судебная коллегия по гражданским делам Верховного Суда Российской Федерации рассмотрела в судебном заседании кассационную жалобу...
    
    Установлено, что истец приобрел у ответчика товар ненадлежащего качества. Ответчик отказался принять товар обратно, ссылаясь на отсутствие гарантийного срока.
    
    Суд первой инстанции пришел к выводу, что на истце лежит обязанность доказать наличие недостатков товара. Однако данная позиция противоречит закону.
    
    В соответствии со статьей 18 Закона РФ "О защите прав потребителей", продавец обязан принять товар ненадлежащего качества и провести проверку качества за свой счет.
    """
    
    chunks = chunker.chunk_document(test_text, "test_document.pdf")
    
    print(f"Создано {len(chunks)} чанков:")
    for chunk in chunks:
        print(f"- {chunk['title']}: {chunk['text'][:100]}...")

if __name__ == "__main__":
    main()
