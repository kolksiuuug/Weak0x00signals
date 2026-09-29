-- Балл = стадия + тренд.
-- stage: 1..4, trend: 1..3, поэтому корректный диапазон: 2..7.

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_dataset_score_range;

ALTER TABLE documents
    ADD CONSTRAINT documents_dataset_score_range
    CHECK (dataset_score IS NULL OR dataset_score BETWEEN 2 AND 7);

COMMENT ON COLUMN documents.dataset_score IS
    'Балл заказчика (стадия+тренд, 2..7); дублирует points из 001';
