-- Совместимость существующей БД с parser/app/database.py.
-- Старый postgres volume мог быть создан до появления новых полей.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS why TEXT,
    ADD COLUMN IF NOT EXISTS is_weak_signal BOOLEAN,
    ADD COLUMN IF NOT EXISTS rejected_reason TEXT;

-- В старой схеме companies = TEXT[], а parser передаёт JSON.
-- Сначала убираем старый default, затем меняем тип.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'documents'
          AND column_name = 'companies'
          AND data_type = 'ARRAY'
    ) THEN
        ALTER TABLE documents
            ALTER COLUMN companies DROP DEFAULT;

        ALTER TABLE documents
            ALTER COLUMN companies TYPE JSONB
            USING to_jsonb(companies);

        ALTER TABLE documents
            ALTER COLUMN companies SET DEFAULT '[]'::jsonb;
    END IF;
END $$;
