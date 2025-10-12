#!/usr/bin/env python3
"""
Скрипт для сравнения качества чанкования между ChatGPT и Gemini
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
import openai
from loguru import logger

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from config import GEMINI_API_KEY, OPENAI_API_KEY

# Настройка API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY не установлен")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY не установлен")

class ChunkingComparison:
    """Класс для сравнения качества чанкования между моделями"""
    
    def __init__(self):
        self.setup_logging()
        self.results_file = "./data/chunking_comparison_results.json"
        self.test_documents = self.load_test_documents()
        
    def setup_logging(self):
        """Настройка логирования"""
        logger.add("./logs/chunking_comparison.log", rotation="1 MB", level="INFO")
    
    def load_test_documents(self) -> List[Dict[str, str]]:
        """Загружает тестовые документы для сравнения"""
        return [
            {
                "name": "consumer_protection_case",
                "text": """
                ВЕРХОВНЫЙ СУД РОССИЙСКОЙ ФЕДЕРАЦИИ
                ОПРЕДЕЛЕНИЕ
                от 15 августа 2023 г. № 44-КГ23-11-К7
                
                Судебная коллегия по гражданским делам Верховного Суда Российской Федерации рассмотрела в судебном заседании кассационную жалобу...
                
                Установлено, что истец приобрел у ответчика товар ненадлежащего качества. Ответчик отказался принять товар обратно, ссылаясь на отсутствие гарантийного срока.
                
                Суд первой инстанции пришел к выводу, что на истце лежит обязанность доказать наличие недостатков товара. Однако данная позиция противоречит закону.
                
                В соответствии со статьей 18 Закона РФ "О защите прав потребителей", продавец обязан принять товар ненадлежащего качества и провести проверку качества за свой счет.
                
                Верховный Суд РФ указал: «Разрешая спор, суд пришёл к выводу, что на истце лежит обязанность доказать, что качество приобретённого ею товара не соответствовало условиям договора... Между тем, в соответствии с приведёнными выше положениями действующего законодательства обязанность доказать добросовестность своих действий лежит именно на продавце товара».
                """
            },
            {
                "name": "contract_dispute_case", 
                "text": """
                АРБИТРАЖНЫЙ СУД ГОРОДА МОСКВЫ
                РЕШЕНИЕ
                от 20 сентября 2023 г. по делу № А40-123456/2023
                
                Арбитражный суд города Москвы в составе судьи Иванова И.И. рассмотрел в судебном заседании дело по иску ООО "Поставщик" к ООО "Покупатель" о взыскании задолженности по договору поставки.
                
                Истец обратился в суд с иском о взыскании с ответчика задолженности в размере 1 500 000 рублей по договору поставки товаров от 15.03.2023 г.
                
                Ответчик иск не признал, ссылаясь на ненадлежащее исполнение истцом обязательств по договору и наличие встречных требований.
                
                Суд установил, что между сторонами заключен договор поставки, согласно которому истец обязался поставить товар, а ответчик - принять и оплатить его.
                
                В соответствии со статьей 454 ГК РФ по договору купли-продажи продавец обязуется передать товар в собственность покупателя, а покупатель обязуется принять этот товар и уплатить за него определенную денежную сумму.
                
                Суд пришел к выводу, что истец надлежащим образом исполнил свои обязательства по поставке товара, что подтверждается товарными накладными и актами приема-передачи.
                """
            },
            {
                "name": "administrative_case",
                "text": """
                ВЕРХОВНЫЙ СУД РОССИЙСКОЙ ФЕДЕРАЦИИ
                ОПРЕДЕЛЕНИЕ
                от 10 октября 2023 г. № 18-КГ23-15-К4
                
                Судебная коллегия по административным делам Верховного Суда Российской Федерации рассмотрела в судебном заседании административное дело по жалобе гражданина Петрова П.П. на постановление о привлечении к административной ответственности.
                
                Заявитель обратился в суд с жалобой на постановление инспектора ГИБДД о привлечении к административной ответственности по части 4 статьи 12.15 КоАП РФ за выезд на встречную полосу движения.
                
                Суд первой инстанции отказал в удовлетворении жалобы, посчитав действия инспектора правомерными.
                
                В соответствии со статьей 12.15 КоАП РФ выезд в нарушение Правил дорожного движения на сторону проезжей части дороги, предназначенную для встречного движения, влечет наложение административного штрафа.
                
                Однако Верховный Суд РФ указал, что при рассмотрении дела суд должен проверить законность и обоснованность вынесенного постановления, а также соблюдение процедуры привлечения к административной ответственности.
                
                Суд установил, что инспектор не предоставил достаточных доказательств нарушения, а также не учел объяснения водителя о вынужденном характере маневра.
                """
            }
        ]
    
    def chunk_with_chatgpt(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Чанкование через ChatGPT"""
        try:
            prompt = f"""
            Разбей следующий юридический документ на семантически завершенные чанки. 
            Каждый чанк должен содержать одну правовую позицию или рассуждение по одной норме права.
            
            Документ: {document_name}
            
            Текст:
            {text}
            
            Верни результат в формате JSON:
            {{
                "chunks": [
                    {{
                        "title": "Название чанка",
                        "text": "Текст чанка",
                        "type": "facts|legal_position|citation|conclusion",
                        "key_articles": ["ст. 18 ЗоЗПП", "ст. 454 ГК РФ"],
                        "legal_concepts": ["защита прав потребителей", "договор поставки"]
                    }}
                ]
            }}
            """
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("chunks", [])
            
        except Exception as e:
            logger.error(f"Ошибка ChatGPT чанкования: {e}")
            return []
    
    def chunk_with_gemini(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Чанкование через Gemini"""
        try:
            prompt = f"""
            Разбей следующий юридический документ на семантически завершенные чанки. 
            Каждый чанк должен содержать одну правовую позицию или рассуждение по одной норме права.
            
            Документ: {document_name}
            
            Текст:
            {text}
            
            Верни результат в формате JSON:
            {{
                "chunks": [
                    {{
                        "title": "Название чанка",
                        "text": "Текст чанка",
                        "type": "facts|legal_position|citation|conclusion",
                        "key_articles": ["ст. 18 ЗоЗПП", "ст. 454 ГК РФ"],
                        "legal_concepts": ["защита прав потребителей", "договор поставки"]
                    }}
                ]
            }}
            """
            
            # Используем более стабильную модель Gemini
            model = genai.GenerativeModel('gemini-1.5-flash')
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
            return result.get("chunks", [])
            
        except Exception as e:
            error_msg = str(e)
            if "User location is not supported" in error_msg:
                logger.warning("⚠️ Gemini API недоступен в вашем регионе. Используем только ChatGPT.")
                return []
            else:
                logger.error(f"Ошибка Gemini чанкования: {e}")
                return []
    
    def evaluate_chunks(self, chunks: List[Dict[str, Any]], document_name: str) -> Dict[str, Any]:
        """Оценивает качество чанков"""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_length": 0,
                "semantic_completeness": 0,
                "legal_relevance": 0,
                "structure_quality": 0
            }
        
        # Базовая статистика
        total_chunks = len(chunks)
        total_length = sum(len(chunk.get("text", "")) for chunk in chunks)
        avg_chunk_length = total_length / total_chunks if total_chunks > 0 else 0
        
        # Оценка семантической завершенности
        semantic_completeness = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            # Проверяем наличие начала и конца предложения
            if text.strip().endswith(('.', '!', '?')) and len(text.split()) > 10:
                semantic_completeness += 1
        semantic_completeness = semantic_completeness / total_chunks if total_chunks > 0 else 0
        
        # Оценка правовой релевантности
        legal_relevance = 0
        legal_keywords = ['статья', 'закон', 'суд', 'право', 'обязанность', 'ответственность', 'договор']
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            keyword_count = sum(1 for keyword in legal_keywords if keyword in text)
            if keyword_count >= 2:  # Минимум 2 правовых термина
                legal_relevance += 1
        legal_relevance = legal_relevance / total_chunks if total_chunks > 0 else 0
        
        # Оценка качества структуры
        structure_quality = 0
        for chunk in chunks:
            has_title = bool(chunk.get("title", "").strip())
            has_type = bool(chunk.get("type", "").strip())
            has_articles = bool(chunk.get("key_articles", []))
            if has_title and has_type and has_articles:
                structure_quality += 1
        structure_quality = structure_quality / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "avg_chunk_length": avg_chunk_length,
            "semantic_completeness": semantic_completeness,
            "legal_relevance": legal_relevance,
            "structure_quality": structure_quality
        }
    
    def compare_models(self) -> Dict[str, Any]:
        """Сравнивает качество чанкования между моделями"""
        logger.info("🔍 Начинаю сравнение моделей чанкования...")
        
        results = {
            "timestamp": time.time(),
            "documents": {},
            "summary": {
                "chatgpt": {"total_score": 0, "documents_tested": 0},
                "gemini": {"total_score": 0, "documents_tested": 0}
            }
        }
        
        for doc in self.test_documents:
            doc_name = doc["name"]
            doc_text = doc["text"]
            
            logger.info(f"📄 Тестирую документ: {doc_name}")
            
            # Тестируем ChatGPT
            logger.info("🤖 Тестирую ChatGPT...")
            chatgpt_chunks = self.chunk_with_chatgpt(doc_text, doc_name)
            chatgpt_evaluation = self.evaluate_chunks(chatgpt_chunks, doc_name)
            
            # Тестируем Gemini
            logger.info("🧠 Тестирую Gemini...")
            gemini_chunks = self.chunk_with_gemini(doc_text, doc_name)
            gemini_evaluation = self.evaluate_chunks(gemini_chunks, doc_name)
            
            # Если Gemini недоступен, пропускаем сравнение
            if not gemini_chunks and "User location is not supported" in str(gemini_chunks):
                logger.warning(f"⚠️ Пропускаю документ {doc_name} - Gemini недоступен")
                continue
            
            # Сохраняем результаты
            results["documents"][doc_name] = {
                "chatgpt": {
                    "chunks": chatgpt_chunks,
                    "evaluation": chatgpt_evaluation
                },
                "gemini": {
                    "chunks": gemini_chunks,
                    "evaluation": gemini_evaluation
                }
            }
            
            # Обновляем сводку
            chatgpt_score = (
                chatgpt_evaluation["semantic_completeness"] + 
                chatgpt_evaluation["legal_relevance"] + 
                chatgpt_evaluation["structure_quality"]
            ) / 3
            
            gemini_score = (
                gemini_evaluation["semantic_completeness"] + 
                gemini_evaluation["legal_relevance"] + 
                gemini_evaluation["structure_quality"]
            ) / 3
            
            results["summary"]["chatgpt"]["total_score"] += chatgpt_score
            results["summary"]["gemini"]["total_score"] += gemini_score
            results["summary"]["chatgpt"]["documents_tested"] += 1
            results["summary"]["gemini"]["documents_tested"] += 1
            
            logger.info(f"✅ {doc_name}: ChatGPT={chatgpt_score:.3f}, Gemini={gemini_score:.3f}")
            
            # Пауза между запросами
            time.sleep(2)
        
        # Вычисляем средние оценки
        if results["summary"]["chatgpt"]["documents_tested"] > 0:
            results["summary"]["chatgpt"]["avg_score"] = (
                results["summary"]["chatgpt"]["total_score"] / 
                results["summary"]["chatgpt"]["documents_tested"]
            )
        
        if results["summary"]["gemini"]["documents_tested"] > 0:
            results["summary"]["gemini"]["avg_score"] = (
                results["summary"]["gemini"]["total_score"] / 
                results["summary"]["gemini"]["documents_tested"]
            )
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Сохраняет результаты сравнения"""
        try:
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в {self.results_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Выводит сводку результатов"""
        logger.info("📊 СВОДКА РЕЗУЛЬТАТОВ СРАВНЕНИЯ")
        logger.info("=" * 50)
        
        chatgpt_avg = results["summary"]["chatgpt"].get("avg_score", 0)
        gemini_avg = results["summary"]["gemini"].get("avg_score", 0)
        
        logger.info(f"🤖 ChatGPT средняя оценка: {chatgpt_avg:.3f}")
        logger.info(f"🧠 Gemini средняя оценка: {gemini_avg:.3f}")
        
        if chatgpt_avg > gemini_avg:
            logger.info("🏆 Победитель: ChatGPT")
            logger.info(f"📈 Преимущество: {((chatgpt_avg - gemini_avg) / gemini_avg * 100):.1f}%")
        elif gemini_avg > chatgpt_avg:
            logger.info("🏆 Победитель: Gemini")
            logger.info(f"📈 Преимущество: {((gemini_avg - chatgpt_avg) / chatgpt_avg * 100):.1f}%")
        else:
            logger.info("🤝 Результаты равны")
        
        logger.info("=" * 50)
        
        # Детальная статистика по документам
        for doc_name, doc_results in results["documents"].items():
            logger.info(f"📄 {doc_name}:")
            chatgpt_eval = doc_results["chatgpt"]["evaluation"]
            gemini_eval = doc_results["gemini"]["evaluation"]
            
            logger.info(f"  ChatGPT: {chatgpt_eval['total_chunks']} чанков, "
                       f"завершенность={chatgpt_eval['semantic_completeness']:.3f}, "
                       f"релевантность={chatgpt_eval['legal_relevance']:.3f}")
            
            logger.info(f"  Gemini: {gemini_eval['total_chunks']} чанков, "
                       f"завершенность={gemini_eval['semantic_completeness']:.3f}, "
                       f"релевантность={gemini_eval['legal_relevance']:.3f}")

def main():
    """Основная функция"""
    logger.info("🚀 Запуск сравнения моделей чанкования")
    
    # Проверяем наличие API ключей
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY не установлен")
        return
    
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY не установлен")
        return
    
    try:
        comparison = ChunkingComparison()
        results = comparison.compare_models()
        comparison.save_results(results)
        comparison.print_summary(results)
        
        logger.info("🎉 Сравнение завершено успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при сравнении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
