"""Сохранение сырья/сигналов в PostgreSQL + pgvector."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List, Optional

from shared.schema import SignalDoc

logger = logging.getLogger("parser.database")


def _dsn() -> str:
    return os.getenv("DATABASE_URL", "")


def persist_documents(documents: Iterable[SignalDoc]) -> None:
    dsn = _dsn()
    if not dsn:
        return
    try:
        import psycopg
        from pgvector.psycopg import register_vector
        from parser.app.embeddings import embed

        docs: List[SignalDoc] = list(documents)
        if not docs:
            return
        with psycopg.connect(dsn.replace("postgresql+psycopg", "postgresql")) as conn:
            register_vector(conn)
            for doc in docs:
                vector = embed(doc.title + "\n" + doc.raw_text)
                conn.execute(
                    """
                    INSERT INTO documents (
                        id,title,area,companies,raw_text,stage,trend,dataset_score,score,why,
                        is_weak_signal,rejected_reason,description,advantage,case_example,
                        evidence_summary,retrieval_score,model_version,model_mode,embedding,updated_at
                    ) VALUES (
                        %(id)s,%(title)s,%(area)s,%(companies)s::jsonb,%(raw_text)s,%(stage)s,%(trend)s,
                        %(dataset_score)s,%(score)s,%(why)s,%(is_weak_signal)s,%(rejected_reason)s,
                        %(description)s,%(advantage)s,%(case_example)s,%(evidence_summary)s,
                        %(retrieval_score)s,%(model_version)s,%(model_mode)s,%(embedding)s,now()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        title=EXCLUDED.title, area=EXCLUDED.area, companies=EXCLUDED.companies,
                        raw_text=EXCLUDED.raw_text, stage=EXCLUDED.stage, trend=EXCLUDED.trend,
                        dataset_score=EXCLUDED.dataset_score, score=EXCLUDED.score, why=EXCLUDED.why,
                        is_weak_signal=EXCLUDED.is_weak_signal, rejected_reason=EXCLUDED.rejected_reason,
                        description=EXCLUDED.description, advantage=EXCLUDED.advantage,
                        case_example=EXCLUDED.case_example, evidence_summary=EXCLUDED.evidence_summary,
                        retrieval_score=EXCLUDED.retrieval_score, model_version=EXCLUDED.model_version,
                        model_mode=EXCLUDED.model_mode, embedding=EXCLUDED.embedding, updated_at=now()
                    """,
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "area": doc.area,
                        "companies": json.dumps(doc.companies, ensure_ascii=False),
                        "raw_text": doc.raw_text,
                        "stage": doc.stage,
                        "trend": doc.trend,
                        "dataset_score": doc.dataset_score,
                        "score": doc.score,
                        "why": doc.why,
                        "is_weak_signal": doc.is_weak_signal,
                        "rejected_reason": doc.rejected_reason,
                        "description": doc.description,
                        "advantage": doc.advantage,
                        "case_example": doc.case_example,
                        "evidence_summary": doc.evidence_summary,
                        "retrieval_score": doc.retrieval_score,
                        "model_version": doc.model_version,
                        "model_mode": doc.model_mode,
                        "embedding": vector,
                    },
                )
                if doc.score is not None:
                    conn.execute(
                        """INSERT INTO signals (document_id,model_version,model_mode,score,dataset_score,is_weak_signal,why,rejected_reason)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (doc.id, doc.model_version or "unknown", doc.model_mode or "unknown", doc.score, doc.dataset_score, bool(doc.is_weak_signal), doc.why, doc.rejected_reason),
                    )
                conn.execute("DELETE FROM sources WHERE document_id=%s", (doc.id,))
                for source in doc.sources:
                    conn.execute(
                        """
                        INSERT INTO sources (
                            document_id,title,url,published_at,source_type,language,trust_level,translated
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (document_id,url) DO UPDATE SET
                            title=EXCLUDED.title, published_at=EXCLUDED.published_at,
                            source_type=EXCLUDED.source_type, language=EXCLUDED.language,
                            trust_level=EXCLUDED.trust_level, translated=EXCLUDED.translated
                        """,
                        (
                            doc.id, source.title, str(source.url), source.date, source.source_type.value,
                            source.language, source.trust_level.value, source.translated,
                        ),
                    )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось сохранить документы в PostgreSQL: %s", exc)


def list_sources(limit: int = 100) -> List[dict]:
    dsn = _dsn()
    if not dsn:
        return []
    try:
        import psycopg
        with psycopg.connect(dsn.replace("postgresql+psycopg", "postgresql")) as conn:
            rows = conn.execute(
                "SELECT title,url,published_at,source_type,language,trust_level,translated FROM sources ORDER BY id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [dict(title=r[0], url=r[1], date=r[2], source_type=r[3], language=r[4], trust_level=r[5], translated=r[6]) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось прочитать sources из PostgreSQL: %s", exc)
        return []


def get_document(document_id: str) -> Optional[dict]:
    dsn = _dsn()
    if not dsn:
        return None
    try:
        import psycopg
        from shared.schema import SourceType, TrustLevel
        with psycopg.connect(dsn.replace("postgresql+psycopg", "postgresql")) as conn:
            row = conn.execute(
                "SELECT id,title,area,companies,raw_text,stage,trend,dataset_score,score,why,is_weak_signal,rejected_reason,description,advantage,case_example,evidence_summary,retrieval_score,model_version,model_mode FROM documents WHERE id=%s",
                (document_id,),
            ).fetchone()
            if not row:
                return None
            source_rows = conn.execute(
                "SELECT title,url,published_at,source_type,language,trust_level,translated FROM sources WHERE document_id=%s ORDER BY id",
                (document_id,),
            ).fetchall()
        data = {
            "id": row[0], "title": row[1], "area": row[2], "companies": row[3] or [], "raw_text": row[4],
            "stage": row[5], "trend": row[6], "dataset_score": row[7], "score": row[8], "why": row[9],
            "is_weak_signal": row[10], "rejected_reason": row[11], "description": row[12], "advantage": row[13],
            "case_example": row[14], "evidence_summary": row[15], "retrieval_score": row[16],
            "model_version": row[17], "model_mode": row[18],
            "sources": [{"title": r[0], "url": r[1], "date": r[2], "source_type": SourceType(r[3]), "language": r[4], "trust_level": TrustLevel(r[5]), "translated": r[6]} for r in source_rows],
        }
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось прочитать документ %s: %s", document_id, exc)
        return None


def semantic_scores(query: str, document_ids: List[str]) -> dict:
    dsn = _dsn()
    if not dsn or not document_ids:
        return {}
    try:
        import psycopg
        from pgvector.psycopg import register_vector
        from parser.app.embeddings import embed
        qvec = embed("query: " + query)
        with psycopg.connect(dsn.replace("postgresql+psycopg", "postgresql")) as conn:
            register_vector(conn)
            rows = conn.execute(
                "SELECT id, 1 - (embedding <=> %s) AS similarity FROM documents WHERE id = ANY(%s) AND embedding IS NOT NULL",
                (qvec, document_ids),
            ).fetchall()
        return {row[0]: float(row[1]) for row in rows}
    except Exception as exc:  # noqa: BLE001
        logger.warning("pgvector semantic rerank недоступен: %s", exc)
        return {}
