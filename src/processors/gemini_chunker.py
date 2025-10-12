"""
Модуль для семантического чанкования документов через Gemini с резервом на ChatGPT
"""

import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from loguru import logger
from ..utils.config import GEMINI_API_KEY, OPENAI_API_KEY

# Настройка Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY не установлен. Gemini чанкование недоступно.")

# Настройка OpenAI для резерва
if OPENAI_API_KEY:
    import openai

    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None
    logger.warning("OPENAI_API_KEY не установлен. ChatGPT резерв недоступен.")


class GeminiChunker:
    """Класс для семантического чанкования через Gemini с резервом на ChatGPT"""

    def __init__(self):
        self.setup_logging()
        self.cache_file = "./data/gemini_chunking_cache.json"
        self.cache = self.load_cache()

    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/gemini_chunker.log", rotation="1 MB", level="INFO")

    def load_cache(self) -> Dict[str, Any]:
        """Загружает кэш результатов чанкования"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить кэш: {e}")
        return {}

    def save_cache(self):
        """Сохраняет кэш результатов чанкования"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")

    def get_cache_key(self, text: str) -> str:
        """Генерирует ключ кэша для текста"""
        import hashlib

        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def extract_json_from_response(self, response_text: str) -> str:
        """Безопасно извлекает JSON из ответа модели"""
        try:
            # Пытаемся найти JSON в различных форматах
            json_patterns = [
                r"```json\s*(.*?)\s*```",
                r"```\s*(.*?)\s*```",
                r"\{.*\}",
            ]

            import re

            for pattern in json_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    json_text = matches[0].strip()
                    # Проверяем, что это валидный JSON
                    json.loads(json_text)
                    return json_text

            # Если не нашли в блоках кода, ищем JSON в тексте
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                # Проверяем валидность
                json.loads(json_text)
                return json_text

            return ""

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            logger.error(f"Полный ответ: {response_text[:500]}...")
            if len(response_text) > 500:
                logger.error(f"Окончание ответа: ...{response_text[-500:]}")
            return ""

    def chunk_with_gemini(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Чанкование через Gemini"""
        try:
            # Если текст слишком большой, разбиваем на части
            max_chars = 30000  # Ограничение для Gemini
            if len(text) > max_chars:
                return self.chunk_large_document_gemini(text, source_file)

            prompt = f"""
            Разбей следующий юридический документ на семантически завершенные чанки. 
            Каждый чанк должен содержать одну правовую позицию или рассуждение по одной норме права.
            
            Документ: {source_file}
            
            Текст:
            {text}
            
            Верни результат в формате JSON:
            {{
                "chunks": [
                    {{
                        "title": "Название чанка",
                        "text": "Текст чанка",
                        "type": "factual_circumstances|legal_position|citation|conclusion",
                        "key_articles": ["ст. 18 ЗоЗПП", "ст. 15 ГК РФ"],
                        "legal_concepts": ["недостаток товара", "права потребителя"]
                    }}
                ]
            }}
            """

            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(prompt)

            # Извлекаем JSON из ответа
            response_text = response.text
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text

            result = json.loads(json_text)
            chunks = result.get("chunks", [])

            # Добавляем метаданные
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append(
                    {
                        "id": f"{source_file}_gemini_chunk_{i}",
                        "text": chunk.get("text", ""),
                        "type": chunk.get("type", "legal_text"),
                        "title": chunk.get("title", f"Чанк {i + 1}"),
                        "key_articles": chunk.get("key_articles", []),
                        "legal_concepts": chunk.get("legal_concepts", []),
                        "source_file": source_file,
                        "chunking_method": "gemini",
                    }
                )

            return processed_chunks

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от Gemini: {e}")
            return self.chunk_with_chatgpt(text, source_file)
        except Exception as e:
            error_msg = str(e)
            if "User location is not supported" in error_msg:
                logger.warning(
                    "⚠️ Gemini API недоступен в вашем регионе. Переходим на ChatGPT."
                )
                return self.chunk_with_chatgpt(text, source_file)
            else:
                logger.error(f"Ошибка Gemini чанкования: {e}")
                return self.chunk_with_chatgpt(text, source_file)

    def chunk_large_document_gemini(
        self, text: str, source_file: str
    ) -> List[Dict[str, Any]]:
        """Обрабатывает большие документы через Gemini по частям"""
        logger.info(
            f"⚠️ Документ {source_file} превышает максимальный контекст Gemini. Разбиваю на части для анализа."
        )

        # Разбиваем на части по 20k символов с перекрытием
        chunk_size = 20000
        overlap = 1000
        all_chunks = []

        # Подсчитываем общее количество частей
        total_parts = (len(text) + chunk_size - overlap - 1) // (chunk_size - overlap)
        logger.info(
            f"📊 Документ будет разбит на {total_parts} частей для полного анализа"
        )

        start = 0
        part_num = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            part_text = text[start:end]

            logger.info(
                f"📄 Анализирую часть {part_num + 1}/{total_parts} документа {source_file}"
            )

            try:
                part_chunks = self.chunk_with_gemini(
                    part_text, f"{source_file}_part_{part_num}"
                )

                # Добавляем информацию о части документа
                for chunk in part_chunks:
                    chunk["id"] = f"{source_file}_gemini_part_{part_num}_{chunk['id']}"
                    chunk["title"] = (
                        f"{chunk['title']} (часть {part_num + 1}/{total_parts})"
                    )
                    chunk["part_info"] = f"часть {part_num + 1}/{total_parts}"

                all_chunks.extend(part_chunks)

            except Exception as e:
                logger.error(f"Ошибка обработки части {part_num + 1}: {e}")
                # Используем ChatGPT для этой части
                try:
                    chatgpt_chunks = self.chunk_with_chatgpt(
                        part_text, f"{source_file}_part_{part_num}"
                    )
                    all_chunks.extend(chatgpt_chunks)
                except Exception as e2:
                    logger.error(f"Ошибка ChatGPT для части {part_num + 1}: {e2}")
                    # Используем fallback для этой части
                    fallback_chunks = self.fallback_chunking(
                        part_text, f"{source_file}_part_{part_num}"
                    )
                    all_chunks.extend(fallback_chunks)

            part_num += 1
            start = end - overlap  # Перекрытие для сохранения контекста

        logger.info(
            f"✅ Полный анализ завершен: {len(all_chunks)} чанков"
        )
        return all_chunks

    def chunk_with_chatgpt(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Резервное чанкование через ChatGPT"""
        if not openai_client:
            logger.error("OpenAI клиент не доступен")
            return self.fallback_chunking(text, source_file)

        try:
            # Если текст слишком большой, разбиваем на части
            max_chars = 5000  # Безопасный лимит для ChatGPT
            if len(text) > max_chars:
                return self.chunk_large_document_chatgpt(text, source_file)

            # Используем простой метод для небольших документов
            return self.chunk_with_chatgpt_simple(text, source_file)

        except Exception as e:
            error_msg = str(e)
            if (
                "unsupported_country_region_territory" in error_msg
                or "Country, region, or territory not supported" in error_msg
            ):
                logger.warning(
                    "⚠️ OpenAI API недоступен в вашем регионе. Используем резервное чанкование."
                )
            else:
                logger.error(f"Ошибка ChatGPT чанкования: {e}")
            return self.fallback_chunking(text, source_file)

    def chunk_large_document_chatgpt(
        self, text: str, source_file: str
    ) -> List[Dict[str, Any]]:
        """Обрабатывает большие документы через ChatGPT по частям"""
        logger.info(
            f"⚠️ Документ {source_file} превышает максимальный контекст ChatGPT. Разбиваю на части для анализа."
        )

        # Разбиваем на части по 3k символов с перекрытием (безопасный размер)
        chunk_size = 3000
        overlap = 200
        all_chunks = []

        start = 0
        part_num = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            part_text = text[start:end]
            part_num += 1

            logger.info(
                f"📄 Анализирую часть {part_num} документа {source_file} ({len(part_text)} символов)"
            )

            try:
                # Простое чанкование каждой части отдельно
                part_chunks = self.chunk_with_chatgpt_simple(
                    part_text, f"{source_file}_part_{part_num}"
                )

                # Добавляем информацию о части документа
                for chunk in part_chunks:
                    chunk["id"] = f"{source_file}_chatgpt_part_{part_num}_{chunk['id']}"
                    chunk["title"] = f"{chunk['title']} (часть {part_num})"

                all_chunks.extend(part_chunks)
                logger.info(f"✅ Часть {part_num} обработана: {len(part_chunks)} чанков")

            except Exception as e:
                logger.error(f"Ошибка обработки части {part_num}: {e}")
                # Используем fallback для этой части
                fallback_chunks = self.fallback_chunking(
                    part_text, f"{source_file}_part_{part_num}"
                )
                all_chunks.extend(fallback_chunks)

            start = end - overlap  # Перекрытие для сохранения контекста

        logger.info(
            f"✅ Полный анализ завершен: {len(all_chunks)} чанков из {part_num} частей"
        )
        return all_chunks

    def chunk_with_chatgpt_simple(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Простое чанкование через ChatGPT без сложной логики"""
        if not openai_client:
            logger.error("OpenAI клиент не доступен")
            return self.fallback_chunking(text, source_file)

        try:
            prompt = f"""
            Разбей следующий фрагмент юридического документа на семантически завершенные чанки. 
            Каждый чанк должен содержать одну правовую позицию или рассуждение по одной норме права.
            
            ВАЖНО: Верни ПОЛНЫЙ JSON ответ без обрезания. Если текст длинный, создай больше чанков, но обязательно заверши JSON корректно.
            
            Документ: {source_file}
            
            Текст:
            {text}
            
            Верни результат в формате JSON (ОБЯЗАТЕЛЬНО ПОЛНЫЙ):
            {{
                "chunks": [
                    {{
                        "title": "Название чанка",
                        "text": "Текст чанка",
                        "type": "factual_circumstances|legal_position|citation|conclusion",
                        "key_articles": ["ст. 18 ЗоЗПП", "ст. 15 ГК РФ"],
                        "legal_concepts": ["недостаток товара", "права потребителя"]
                    }}
                ]
            }}
            """

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — эксперт по анализу юридических документов. Твоя задача — разбить судебное решение на семантически завершенные блоки.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=3000,  # Увеличиваем для полных ответов
                timeout=60,
            )

            response_text = response.choices[0].message.content

            # Извлекаем JSON из ответа
            json_text = self.extract_json_from_response(response_text)

            if not json_text:
                logger.error("Не удалось извлечь JSON из ответа ChatGPT")
                logger.error(f"Полный ответ ChatGPT: {response_text}")
                return self.fallback_chunking(text, source_file)

            result = json.loads(json_text)
            chunks = result.get("chunks", [])

            # Добавляем метаданные
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append(
                    {
                        "id": f"{source_file}_chatgpt_chunk_{i}",
                        "text": chunk.get("text", ""),
                        "type": chunk.get("type", "legal_text"),
                        "title": chunk.get("title", f"Чанк {i + 1}"),
                        "key_articles": chunk.get("key_articles", []),
                        "legal_concepts": chunk.get("legal_concepts", []),
                        "source_file": source_file,
                        "chunking_method": "chatgpt",
                    }
                )

            return processed_chunks

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от ChatGPT: {e}")
            return self.fallback_chunking(text, source_file)
        except Exception as e:
            error_msg = str(e)
            if (
                "unsupported_country_region_territory" in error_msg
                or "Country, region, or territory not supported" in error_msg
            ):
                logger.warning(
                    "⚠️ OpenAI API недоступен в вашем регионе. Используем резервное чанкование."
                )
            else:
                logger.error(f"Ошибка ChatGPT чанкования: {e}")
            return self.fallback_chunking(text, source_file)


    def fallback_chunking(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Резервное чанкование по абзацам"""
        chunks = []
        paragraphs = text.split("\n\n")
        chunk_id = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) > 100:  # Только значимые абзацы
                chunks.append(
                    {
                        "id": f"{source_file}_fallback_chunk_{chunk_id}",
                        "text": paragraph,
                        "type": "legal_text",
                        "title": f"Абзац {chunk_id + 1}",
                        "key_articles": [],
                        "legal_concepts": [],
                        "source_file": source_file,
                        "chunking_method": "fallback",
                    }
                )
                chunk_id += 1

        return chunks

    def chunk_document(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Разбивает документ на семантические чанки через Gemini с резервом на ChatGPT

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

        # Оптимизация для больших документов
        text_length = len(text)
        if text_length > 50000:
            logger.info(f"🚀 Большой документ {source_file}: {text_length:,} символов")
            logger.info("⚡ Используем ChatGPT для ускорения обработки")
            try:
                chunks = self.chunk_with_chatgpt(text, source_file)
                self.cache[cache_key] = chunks
                self.save_cache()
                logger.info(f"✅ Обработано {len(chunks)} чанков для {source_file}")
                return chunks
            except Exception as e:
                logger.warning(f"Ошибка ChatGPT для большого документа: {e}")

        try:
            # Приоритетно используем Gemini
            logger.info(f"Используем Gemini чанкование для {source_file}")
            chunks = self.chunk_with_gemini(text, source_file)

            # Сохраняем в кэш
            self.cache[cache_key] = chunks
            self.save_cache()

            logger.info(f"✅ Обработано {len(chunks)} чанков для {source_file}")
            return chunks

        except Exception as e:
            logger.error(f"Ошибка при чанковании: {e}")
            return self.fallback_chunking(text, source_file)

    def estimate_cost(self, text: str) -> Dict[str, float]:
        """Оценивает стоимость обработки"""
        # Примерная оценка токенов (1 токен ≈ 4 символа для русского текста)
        input_tokens = len(text) / 4
        output_tokens = input_tokens * 0.3  # Примерно 30% от входных токенов

        # Стоимость Gemini (примерная)
        gemini_cost = (input_tokens * 0.000075 + output_tokens * 0.0003) / 1000

        # Стоимость ChatGPT (примерная)
        chatgpt_cost = (input_tokens * 0.03 + output_tokens * 0.06) / 1000

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_gemini_usd": gemini_cost,
            "estimated_cost_chatgpt_usd": chatgpt_cost,
            "estimated_cost_usd": min(
                gemini_cost, chatgpt_cost
            ),  # Берем меньшую стоимость
        }
