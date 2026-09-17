-- Инициализация БД. Выполняется один раз при первом создании тома postgres_data.
-- Полную схему (documents / signals / sources) проектирует Участник 2 в /ml.

CREATE EXTENSION IF NOT EXISTS vector;

-- Минимальная проверочная таблица, чтобы скелет мог убедиться, что расширение доступно.
CREATE TABLE IF NOT EXISTS healthcheck (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    note        TEXT NOT NULL DEFAULT 'скелет: pgvector подключён'
);

INSERT INTO healthcheck (note) VALUES ('скелет: pgvector подключён')
ON CONFLICT DO NOTHING;
