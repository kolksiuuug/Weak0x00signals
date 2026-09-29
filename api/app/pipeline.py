"""Оркестрация одного поиска: parser → trust-слой → ml → ранжирование → ТОП-15.

Выполняется фоновой задачей FastAPI, результат кладётся в Redis под job_id.
Позже этот же модуль станет точкой распараллеливания и кэширования тяжёлых вызовов.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from api.app import clients, jobs
from api.app.clients import ServiceUnavailable
from api.app.config import settings
from api.app.trust import check_document
from shared.contracts import JobStatus, SearchRequest, SearchStats
from shared.schema import SignalDoc

logger = logging.getLogger("api.pipeline")


async def run_search(job_id: str, req: SearchRequest) -> None:
    """Полный прогон. Все ошибки перехватываются и превращаются в статус «ошибка»."""
    job = await jobs.get_job(job_id)
    if job is None:
        logger.warning("Задача %s исчезла из Redis до начала обработки", job_id)
        return

    job.status = JobStatus.RUNNING
    await jobs.save_job(job)

    try:
        candidate_pool = max(req.limit * 5, 75)
        collected = await clients.collect(req.query, area=req.area, limit=candidate_pool)
        logger.info("job=%s: собрано документов=%d, режим=%s, коннекторы=%s", job_id, len(collected.documents), collected.retrieval_mode, collected.connector_status)

        # 1) trust-слой: дедуп источников, пересчёт доверенности, отсев по §2.8
        checked: List[SignalDoc] = []
        rejected: List[SignalDoc] = []
        for doc in collected.documents:
            doc, reason = check_document(doc)
            if reason:
                rejected.append(doc.model_copy(update={"is_weak_signal": False, "rejected_reason": reason}))
            else:
                checked.append(doc)

        # 2) Канонический ML scoring + maturity filter + TOP-15 ranking.
        # Важно: ranking.py содержит единственную формулу ранжирования из методологии
        # (0.50 confidence + 0.30 priority + 0.20 evidence). Не дублируем её здесь.
        ranked = await clients.rank_documents(checked, req.query, req.limit) if checked else {
            "results": [], "rejected": [], "total_candidates": 0
        }
        top = [SignalDoc.model_validate(item) for item in ranked.get("results", [])]
        rejected.extend(SignalDoc.model_validate(item) for item in ranked.get("rejected", []))

        # LLM не вызывается для всей выдачи. Раньше /enrich проходил по каждому
        # TOP-документу и создавал серию запросов к GigaChat, после чего начинались 429.
        # Теперь enrichment ленивый: только при открытии конкретного сигнала.
        llm_model = os.getenv("LLM_MODEL") or None

        job.results = top
        job.rejected = rejected
        scores = [float(d.score or 0.0) for d in top]
        if scores:
            logger.info(
                "job=%s: score distribution min=%.3f median=%.3f max=%.3f",
                job_id, min(scores), sorted(scores)[len(scores)//2], max(scores),
            )
        job.stats = SearchStats(
            sources_processed=collected.sources_processed,
            candidates_total=len(collected.documents),
            candidates_rejected=len(rejected),
            confident_signals=sum(1 for d in top if (d.score or 0) > settings.confidence_threshold),
            llm_model=llm_model,
            ml_mode=(top[0].model_mode if top else None),
            retrieval_mode=collected.retrieval_mode,
            connector_status=collected.connector_status,
        )
        job.status = JobStatus.DONE
        logger.info(
            "job=%s: готово. в выдаче=%d, отклонено=%d, модель=%s",
            job_id, len(top), len(rejected), llm_model,
        )
    except ServiceUnavailable as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        logger.error("job=%s: %s", job_id, exc)
    except Exception as exc:  # noqa: BLE001 — задача не должна ронять воркер
        job.status = JobStatus.FAILED
        job.error = "Внутренняя ошибка обработки запроса: {}".format(exc)
        logger.exception("job=%s: непредвиденная ошибка", job_id)

    job.finished_at = jobs.now_iso()
    await jobs.save_job(job)
