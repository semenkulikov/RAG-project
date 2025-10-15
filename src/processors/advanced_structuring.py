"""
Продвинутая структуризация судебных актов через Gemini.
Формирует глубокий JSON с legal_reasoning_blocks (ratio decidendi, цитаты, ссылки и вес).
"""

from typing import Dict, Any
import os
import json
from loguru import logger
import google.generativeai as genai

from ..utils.config import GEMINI_API_KEY


ADVANCED_STRUCTURING_PROMPT = (
    """
Проанализируй следующий текст судебного акта Верховного Суда РФ как юрист-аналитик высшей квалификации. Извлеки из него ключевую информацию и представь ее СТРОГО в формате JSON.

### ЗАДАЧА ###
1.  case_info: Извлеки основную информацию о деле.
2.  legal_category: Определи отрасль права и суть спора.
3.  summary_facts: Изложи краткую фабулу дела.
4.  legal_reasoning_blocks: САМОЕ ВАЖНОЕ. Найди ключевые правовые позиции. Для каждой позиции извлеки:
    * principle: Кратко сформулированный правовой принцип.
    * law_reference: Упомянутые статьи закона.
    * ratio_decidendi: Суть правовой позиции (дух закона) в 1-2 предложениях.
    * core_quote: Прямая, самая сильная цитата (буква закона).
    * importance_score: Оцени вес акта по шкале 1-5.
    * case_outcome: Результат рассмотрения (например, "Отменено, направлено на новое рассмотрение").

### СТРУКТУРА JSON ###
{
  "case_info": { "case_number": "...", "date": "...", "court": "...", "type": "..." },
  "legal_category": "...",
  "summary_facts": "...",
  "legal_reasoning_blocks": [
    {
      "id": "ID-block-1",
      "principle": "...",
      "law_reference": "...",
      "ratio_decidendi": "...",
      "core_quote": "...",
      "importance_score": 3,
      "case_outcome": "..."
    }
  ]
}

### ТЕКСТ СУДЕБНОГО АКТА ###
{text_content}
"""
    .strip()
)


def _configure_gemini():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY не установлен")
    genai.configure(api_key=GEMINI_API_KEY)


def structure_court_decision_advanced(text: str) -> Dict[str, Any]:
    """
    Возвращает глубокий JSON анализа судебного акта по продвинутому промпту.
    """
    _configure_gemini()
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = ADVANCED_STRUCTURING_PROMPT.format(text_content=text)
    response = model.generate_content(prompt)

    content = response.text or ""

    # Попытка безопасного извлечения JSON
    json_text = _extract_json(content)
    if not json_text:
        logger.error("Не удалось извлечь JSON из ответа Gemini при структуризации")
        raise ValueError("Gemini вернул не-JSON ответ при структуризации")

    data = json.loads(json_text)
    return data


def _extract_json(response_text: str) -> str:
    try:
        import re

        patterns = [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{[\s\S]*\}"]
        for p in patterns:
            m = re.findall(p, response_text, re.DOTALL)
            if m:
                candidate = m[0].strip()
                json.loads(candidate)
                return candidate

        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            candidate = response_text[start:end]
            json.loads(candidate)
            return candidate
        return ""
    except Exception:
        return ""


