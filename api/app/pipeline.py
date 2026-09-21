"""Оркестрация одного поиска: parser → trust-слой → ml → ранжирование → ТОП-15.

Выполняется фоновой задачей FastAPI, результат кладётся в Redis под job_id.
Позже этот же модуль станет точкой распараллеливания и кэширования тяжёлых вызовов.
"""

from __future__ import annotations

import logging
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
        collected = await clients.collect(req.query, area=req.area, limit=max(req.limit * 2, 30))
        logger.info("job=%s: собрано документов=%d", job_id, len(collected.documents))

        # 1) trust-слой: дедуп источников, пересчёт доверенности, отсев по §2.8
        checked: List[SignalDoc] = []
        rejected: List[SignalDoc] = []
        for doc in collected.documents:
            doc, reason = check_document(doc)
            if reason:
                rejected.append(doc.model_copy(update={"is_weak_signal": False, "rejected_reason": reason}))
            else:
                checked.append(doc)

        # 2) скоринг и фильтр зрелости на стороне ml
        scores: Dict[str, object] = {}
        if checked:
            for item in await clients.score_batch(checked, req.query):
                scores[item.id] = item

        accepted: List[SignalDoc] = []
        for doc in checked:
            item = scores.get(doc.id)
            if item is None:
                rejected.append(doc.model_copy(update={
                    "is_weak_signal": False,
                    "rejected_reason": "Отклонено: ML-сервис не вернул оценку для кандидата.",
                }))
                continue
            enriched = doc.model_copy(update={
                "score": item.score,
                "why": item.why,
                "is_weak_signal": item.is_weak_signal,
                "rejected_reason": item.rejected_reason,
            })
            if item.rejected_reason or not item.is_weak_signal:
                reason = item.rejected_reason or ("Отклонено: уверенность модели {:.0f}% ниже порога слабого сигнала.".format(item.score * 100))
                rejected.append(enriched.model_copy(update={"is_weak_signal": False, "rejected_reason": reason}))
            else:
                accepted.append(enriched)

        # 3) ранжирование и ТОП-15
        accepted.sort(key=lambda d: ((d.score or 0.0) * 0.85 + (d.retrieval_score or 0.0) * 0.15, d.dataset_score or 0), reverse=True)
        top = accepted[: req.limit]

        try:
            model_info = await clients.llm_models()
            llm_model = model_info.get("selected")
        except ServiceUnavailable:
            llm_model = None

        # 4) RAG-enrichment только после ranking: описание/преимущество/кейс строятся из найденных источников.
        insight_mode = None
        if top:
            try:
                enriched = await clients.enrich(top, req.query)
                top = enriched.documents
                insight_mode = enriched.mode
            except ServiceUnavailable as exc:
                logger.warning("job=%s: enrichment недоступен: %s", job_id, exc)
                insight_mode = "grounded-template"

        job.results = top
        job.rejected = rejected
        job.stats = SearchStats(
            sources_processed=collected.sources_processed,
            candidates_total=len(collected.documents),
            candidates_rejected=len(rejected),
            confident_signals=sum(1 for d in top if (d.score or 0) > settings.confidence_threshold),
            llm_model=llm_model,
            ml_mode=(next(iter(scores.values())).model_mode if scores else None),
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
