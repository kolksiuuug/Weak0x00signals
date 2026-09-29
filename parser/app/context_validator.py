import json
import logging

from parser.app.llm import complete_grounded

logger = logging.getLogger("parser.context_validator")


async def validate_context(query: str, documents):
    """Финальная проверка соответствия документов исходному запросу через LLM."""
    if not documents:
        return []

    items = [
        {
            "id": doc.id,
            "title": (doc.title or "")[:500],
            "abstract": (getattr(doc, "abstract", "") or "")[:1200],
        }
        for doc in documents
    ]

    prompt = """Ты выполняешь финальную проверку релевантности результатов научного поиска.

ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Для каждого документа определи, действительно ли он относится к предметной
области исходного запроса.

Проверяй СМЫСЛ, а не отдельные совпадающие слова.

Документ считается VALID только если его основная тема действительно
относится к запросу.

Если совпадает только слово "ИИ", "технологии", "промышленный",
"данные" или другое отдельное слово, но предметная область другая —
это INVALID.

Примеры:
- запрос про промышленный ИИ → методы ИИ для производства = VALID
- запрос про промышленный ИИ → ИИ для промышленного оборудования = VALID
- запрос про промышленный ИИ → маркетинг = INVALID
- запрос про промышленный ИИ → лекарства = INVALID
- запрос про промышленный ИИ → сельское хозяйство без связи с промышленным ИИ = INVALID

Верни результат ДЛЯ КАЖДОГО документа.

Формат ТОЛЬКО JSON:
[
  {{
    "id": "ID",
    "valid": true,
    "score": 95,
    "reason": "Кратко объясни соответствие"
  }}
]

score:
100 = прямое соответствие
80 = очень близкое соответствие
60 = умеренное соответствие
40 = слабая связь
20 = практически не относится
0 = не относится

valid=true только если документ действительно подходит
по предметной области запроса.

Не придумывай факты.
Не меняй запрос.
Не используй внешние знания.
""".format(
        query=query,
    )

    prompt += "\n\nДОКУМЕНТЫ:\n" + json.dumps(
        items,
        ensure_ascii=False,
    )

    try:
        raw, _, mode = await complete_grounded(prompt)

        if mode != "llm" or not raw:
            logger.warning(
                "Context validation unavailable; keeping reranked documents"
            )
            return documents

        cleaned = raw.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "", 1)
            cleaned = cleaned.replace("```", "", 1).strip()

        left = cleaned.find("[")
        right = cleaned.rfind("]")

        if left < 0 or right <= left:
            logger.warning(
                "Context validation returned invalid JSON: %r",
                raw[:500],
            )
            return documents

        data = json.loads(cleaned[left:right + 1])

        if not isinstance(data, list):
            return documents

        valid_ids = {doc.id for doc in documents}
        decisions = {}

        for item in data:
            if not isinstance(item, dict):
                continue

            doc_id = item.get("id")

            if doc_id not in valid_ids:
                continue

            valid = bool(item.get("valid", False))

            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0

            reason = str(item.get("reason", "")).strip()

            decisions[doc_id] = {
                "valid": valid,
                "score": max(0.0, min(100.0, score)),
                "reason": reason,
            }

        if not decisions:
            logger.warning("Context validation returned no valid decisions")
            return documents

        accepted = []
        rejected = 0

        for doc in documents:
            decision = decisions.get(doc.id)

            if not decision:
                continue

            if not decision["valid"]:
                rejected += 1
                logger.info(
                    "context reject: score=%.1f title=%r reason=%s",
                    decision["score"],
                    doc.title[:120],
                    decision["reason"],
                )
                continue

            accepted.append(
                doc.model_copy(
                    update={
                        "retrieval_score": round(
                            decision["score"] / 100.0,
                            3,
                        ),
                    }
                )
            )

        logger.info(
            "Context validation: candidates=%d checked=%d accepted=%d rejected=%d",
            len(documents),
            len(decisions),
            len(accepted),
            rejected,
        )

        return accepted

    except Exception as exc:
        logger.warning(
            "Context validation failed: %s: %s; keeping reranked documents",
            type(exc).__name__,
            exc,
        )
        return documents
