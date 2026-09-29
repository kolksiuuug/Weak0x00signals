-- 003_card_fields.sql — поля итоговой карточки-инсайта (CLAUDE.md §8).
--
-- Зачем отдельной миграцией, а не правкой 001. Файл 001 уже применён (или будет
-- применён) с известной контрольной суммой; раннер (ml/db/migrate.py) сознательно
-- падает, если применённая миграция изменилась. Новые поля добавляются только новым
-- файлом — историю схемы не переписываем.
--
-- Откуда взялся состав колонок. Это ровно те поля, которые появились в SignalDoc
-- (shared/schema.py) и требуются по §8: «по каждому сигналу: описание, преимущество,
-- кейс-пример, источники, скоринг, объяснение». Заполняет их НЕ модель этапа 1:
-- description / advantage / case_example / evidence_summary генерирует LLM-сервис
-- Участника 3 строго по найденным источникам (§2.3), retrieval_score приходит из
-- поиска Участника 1. Со стороны /ml это хранилище, а не вычисление.
--
-- Идемпотентность: ADD COLUMN IF NOT EXISTS, повторный запуск безопасен.

-- --- текст карточки ---------------------------------------------------------
-- Всё на русском (§2.1). Для зарубежных источников текст обязан быть русским
-- резюме с пометкой автоперевода — сама пометка живёт в sources.translated (001).

ALTER TABLE documents ADD COLUMN IF NOT EXISTS description      TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS advantage        TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS case_example     TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS evidence_summary TEXT;

COMMENT ON COLUMN documents.description IS
    'Описание технологии на русском, сгенерировано по найденным источникам (§2.3, §8)';
COMMENT ON COLUMN documents.advantage IS
    'В чём преимущество технологии — по источникам, не из знаний LLM (§2.3, §8)';
COMMENT ON COLUMN documents.case_example IS
    'Кейс-пример внедрения или применения со ссылкой на источник (§8)';
COMMENT ON COLUMN documents.evidence_summary IS
    'Сводка подтверждающей фактуры: на чём основан вывод (§8)';

-- --- скоринг и происхождение оценки -----------------------------------------
-- dataset_score — «Балл» заказчика (стадия+тренд, 3..7). В documents уже есть points
-- из 001 с тем же смыслом; dataset_score добавляется как имя из контракта §3, чтобы
-- слой API не мапил поля вручную. Ограничение то же, что у points.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS dataset_score   SMALLINT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS retrieval_score DOUBLE PRECISION;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS model_version   TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS model_mode      TEXT;

COMMENT ON COLUMN documents.dataset_score IS
    'Балл заказчика (стадия+тренд, 3..7) под именем из контракта §3; дублирует points из 001';
COMMENT ON COLUMN documents.retrieval_score IS
    'Релевантность кандидата запросу из поиска (зона Участника 1), не оценка модели этапа 1';
COMMENT ON COLUMN documents.model_version IS
    'Версия модели, выдавшей последнюю оценку — обязательна для логирования выбора модели (§2.4)';
COMMENT ON COLUMN documents.model_mode IS
    'Режим оценки: обученный артефакт или прозрачные правила запасного режима';

-- Диапазон балла — тот же, что в 001 у points (§1). Ограничение ставим отдельным
-- шагом: у ADD CONSTRAINT нет IF NOT EXISTS, поэтому оборачиваем в DO-блок.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'documents_dataset_score_range'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_dataset_score_range
            CHECK (dataset_score IS NULL OR dataset_score BETWEEN 3 AND 7);
    END IF;
END
$$;

-- Оценка релевантности — доля, а не произвольное число: иначе ранжирование по ней
-- несопоставимо между прогонами.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'documents_retrieval_score_range'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_retrieval_score_range
            CHECK (retrieval_score IS NULL OR (retrieval_score >= 0 AND retrieval_score <= 1));
    END IF;
END
$$;

-- --- выдача карточек --------------------------------------------------------
-- Запрос витрины: показать готовые карточки с самым свежим текстом.
CREATE INDEX IF NOT EXISTS documents_card_ready_idx
    ON documents (updated_at DESC)
    WHERE description IS NOT NULL;
