-- 001_core_schema.sql — ядро схемы: documents / sources / signals (CLAUDE.md §6, Участник 2).
--
-- Соответствие контракту §3: таблицы повторяют поля SignalDoc и Source один в один,
-- включая типы-перечисления. Значения ENUM обязаны совпадать со строками в
-- shared/schema.py посимвольно — иначе pydantic и БД разойдутся молча.
--
-- Миграция идемпотентна: повторный запуск ничего не ломает.

CREATE EXTENSION IF NOT EXISTS vector;

-- --- перечисления -----------------------------------------------------------
-- CREATE TYPE не поддерживает IF NOT EXISTS, поэтому оборачиваем в DO-блок.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'source_type') THEN
        CREATE TYPE source_type AS ENUM (
            'научная_статья',
            'патент',
            'новость',
            'аналитический_отчёт',
            'гос_реестр',
            'блог',
            'соцсеть',
            'пресс_релиз'
        );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'trust_level') THEN
        CREATE TYPE trust_level AS ENUM ('высокий', 'средний', 'пониженный');
    END IF;
END
$$;

-- --- documents --------------------------------------------------------------
-- Технология-кандидат. Сюда попадают и строки датасета организаторов (origin='dataset'),
-- и контрольная выборка (origin='control'), и живые документы парсера (origin='parser').

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,                       -- SignalDoc.id
    title         TEXT NOT NULL,
    area          TEXT,
    companies     TEXT[] NOT NULL DEFAULT '{}',
    raw_text      TEXT NOT NULL DEFAULT '',
    stage         SMALLINT CHECK (stage BETWEEN 1 AND 4), -- шкала CLAUDE.md §1
    trend         SMALLINT CHECK (trend BETWEEN 1 AND 3),
    points        SMALLINT CHECK (points BETWEEN 3 AND 7),-- «Балл» из датасета, если есть
    -- Имена ограничений заданы явно, а не отданы автогенерации: блок совместимости
    -- ниже ищет их по имени через pg_constraint, иначе он добавил бы дубликат.
    origin        TEXT NOT NULL DEFAULT 'parser'
                  CONSTRAINT documents_origin_known
                  CHECK (origin IN ('dataset', 'control', 'parser')),
    label         SMALLINT CONSTRAINT documents_label_binary
                  CHECK (label IS NULL OR label IN (0, 1)),  -- метка; NULL для живых данных
    -- 384 — размерность intfloat/multilingual-e5-small, которую хардкодит
    -- parser/app/embeddings.py (EMBED_DIM). Эмбеддинги считает и пишет парсер (У3),
    -- /ml их не вычисляет и не читает, поэтому цифру диктует парсер, а не эта схема.
    -- Менять только вместе с EMBED_DIM: вектор другой длины база не примет.
    embedding     vector(384),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON COLUMN documents.origin IS
    'dataset — датасет организаторов; control — контрольная выборка отрицательного класса; parser — живой документ из открытых источников';
COMMENT ON COLUMN documents.label IS
    'Обучающая метка этапа 1: 1 — слабый сигнал, 0 — зрелая технология или хайп. NULL — размётки нет';

-- --- sources ----------------------------------------------------------------
-- Полный набор метаданных обязателен (CLAUDE.md §2.7). NOT NULL здесь — это не
-- педантизм: источник без типа, языка или доверенности нельзя показывать в выдаче.

CREATE TABLE IF NOT EXISTS sources (
    id            BIGSERIAL PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    published_at  TEXT,                                   -- ISO 8601 или NULL (даты может не быть)
    source_type   source_type NOT NULL,
    language      TEXT NOT NULL,                          -- 'ru', 'en', ...
    trust_level   trust_level NOT NULL,
    translated    BOOLEAN NOT NULL DEFAULT FALSE,         -- пометка автоперевода (§2.9)
    domain        TEXT,                                   -- денормализовано для аудита доверенности
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT sources_unique_url_per_document UNIQUE (document_id, url)
);

COMMENT ON COLUMN sources.translated IS
    'TRUE — использован автоперевод или генеративное резюме, требуется пометка в выдаче (CLAUDE.md §2.9)';

-- --- signals ----------------------------------------------------------------
-- Результат инференса. Отдельная таблица, а не колонки в documents: одна и та же
-- технология переоценивается при каждом прогоне и при каждой новой версии модели,
-- и историю оценок нужно сохранять — иначе нечем объяснить, почему вчера было иначе.

CREATE TABLE IF NOT EXISTS signals (
    id               BIGSERIAL PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    score            DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    is_weak_signal   BOOLEAN NOT NULL,
    why              TEXT NOT NULL,                       -- объяснение на русском (§2.1, §2.2)
    rejected_reason  TEXT,                                -- причина отклонения (§2.10)
    predictors       JSONB NOT NULL DEFAULT '[]'::jsonb,  -- вклады признаков
    model_kind       TEXT NOT NULL DEFAULT 'unknown',
    model_trained_at TEXT,
    threshold        DOUBLE PRECISION,
    query            TEXT,                                -- пользовательский запрос, если прогон из /search
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE signals IS
    'История оценок. Каждый прогон модели добавляет строку, старые не перезаписываются';

-- --- совместимость с db/init.sql ---------------------------------------------
-- Зачем это здесь. Те же три таблицы создаёт db/init.sql, смонтированный в
-- /docker-entrypoint-initdb.d/ (docker-compose.yml). На инициализации контейнера он
-- выполняется ПЕРВЫМ, поэтому CREATE TABLE IF NOT EXISTS выше становятся no-op,
-- и колонок, которых нет в init.sql, в таблицах не окажется. Без блока ниже на такой
-- базе падает уже 002_indexes.sql — на CREATE INDEX по несуществующей колонке.
--
-- ALTER ... ADD COLUMN IF NOT EXISTS делает 001 безразличным к тому, кто создал
-- таблицы: на чистой базе это no-op (колонки уже созданы выше), на базе init.sql —
-- добавляет ровно то, чего не хватает /ml. Ни одну чужую колонку мы не меняем и не
-- удаляем, поэтому живые вставки парсера продолжают работать без правок.
--
-- Состав блока — это полный список того, что /ml требует сверх init.sql:
-- documents.origin / label / points (обучающая выборка, ml/db/seed.py) и
-- sources.domain (аудит доверенности по домену, §2.8). Таблицу signals /ml не читает
-- и не пишет — она целиком в зоне парсера, поэтому её расхождения здесь не трогаем.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'parser';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS label  SMALLINT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS points SMALLINT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE sources   ADD COLUMN IF NOT EXISTS domain TEXT;

-- CHECK-ограничения ставим отдельно: у ADD CONSTRAINT нет IF NOT EXISTS. На чистой
-- базе они уже пришли из CREATE TABLE выше — тогда DO-блок ничего не делает.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_origin_known') THEN
        ALTER TABLE documents ADD CONSTRAINT documents_origin_known
            CHECK (origin IN ('dataset', 'control', 'parser'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_label_binary') THEN
        ALTER TABLE documents ADD CONSTRAINT documents_label_binary
            CHECK (label IS NULL OR label IN (0, 1));
    END IF;
END
$$;

-- --- обновление updated_at --------------------------------------------------

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_touch_updated_at ON documents;
CREATE TRIGGER documents_touch_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
