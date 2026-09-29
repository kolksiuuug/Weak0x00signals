-- 002_indexes.sql — индексы под реальные запросы сервиса.
--
-- Индексы вынесены из 001 намеренно: схема и стратегия доступа меняются с разной
-- частотой, и переиндексация не должна требовать правки файла со структурой таблиц.

-- --- documents --------------------------------------------------------------

-- Выборка обучающей части: WHERE origin IN ('dataset','control') AND label IS NOT NULL.
CREATE INDEX IF NOT EXISTS documents_origin_label_idx
    ON documents (origin, label);

-- Фильтр выдачи по области (одна из 6 областей или свободная тема).
CREATE INDEX IF NOT EXISTS documents_area_idx
    ON documents (area)
    WHERE area IS NOT NULL;

-- Поиск дубликатов по названию: нормализуем так же, как ml/app/ranking.py — без регистра.
CREATE INDEX IF NOT EXISTS documents_title_lower_idx
    ON documents (lower(title));

-- Векторный поиск по косинусной близости (pgvector). HNSW, а не IVFFlat:
-- IVFFlat требует обучения списков на уже загруженных данных, а база наполняется
-- инкрементально по ходу работы парсера.
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

-- --- sources ----------------------------------------------------------------

CREATE INDEX IF NOT EXISTS sources_document_id_idx
    ON sources (document_id);

-- Аудит доверенности: «покажи все источники пониженной доверенности» (CLAUDE.md §2.8).
CREATE INDEX IF NOT EXISTS sources_trust_level_idx
    ON sources (trust_level);

-- Дедупликация по URL между разными документами.
CREATE INDEX IF NOT EXISTS sources_url_idx
    ON sources (url);

CREATE INDEX IF NOT EXISTS sources_domain_idx
    ON sources (domain)
    WHERE domain IS NOT NULL;

-- --- signals ----------------------------------------------------------------

-- Последняя оценка документа: ORDER BY created_at DESC LIMIT 1.
CREATE INDEX IF NOT EXISTS signals_document_created_idx
    ON signals (document_id, created_at DESC);

-- Статистика для экрана результатов: число сигналов с уверенностью выше порога.
CREATE INDEX IF NOT EXISTS signals_score_idx
    ON signals (score DESC)
    WHERE is_weak_signal;

-- Разбор отклонений: «покажи, что и почему отсеял фильтр зрелости» (§2.10).
CREATE INDEX IF NOT EXISTS signals_rejected_idx
    ON signals (created_at DESC)
    WHERE rejected_reason IS NOT NULL;
