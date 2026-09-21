CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    area TEXT,
    companies JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_text TEXT NOT NULL,
    stage SMALLINT,
    trend SMALLINT,
    dataset_score SMALLINT,
    score DOUBLE PRECISION,
    why TEXT,
    is_weak_signal BOOLEAN,
    rejected_reason TEXT,
    description TEXT,
    advantage TEXT,
    case_example TEXT,
    evidence_summary TEXT,
    retrieval_score DOUBLE PRECISION,
    model_version TEXT,
    model_mode TEXT,
    embedding vector(384),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sources (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT,
    source_type TEXT NOT NULL,
    language TEXT NOT NULL,
    trust_level TEXT NOT NULL,
    translated BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(document_id, url)
);

CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    model_mode TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    dataset_score SMALLINT,
    is_weak_signal BOOLEAN NOT NULL,
    why TEXT,
    rejected_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_area ON documents(area);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);

CREATE TABLE IF NOT EXISTS model_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT NOT NULL,
    mode TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    positive_samples INTEGER NOT NULL,
    negative_samples INTEGER NOT NULL,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    report JSONB NOT NULL DEFAULT '{}'::jsonb
);
