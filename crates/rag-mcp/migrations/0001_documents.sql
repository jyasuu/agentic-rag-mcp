-- Schema assumption for the corpus this server reads from. Per SPEC.md
-- ("Out of Scope: Ingestion pipeline"), content and embeddings are written
-- here by an existing/separate ingestion process -- this migration exists
-- so the tsvector/pg_trgm/pgvector backend tickets have a concrete,
-- testable schema to implement against, not because this server owns
-- ingestion.
--
-- `simple` config (not `english`) is used for `search_vector` deliberately:
-- the corpus mixes Chinese content with English/code identifiers and error
-- codes (SPEC.md), and English stemming would mangle identifier-style
-- tokens (e.g. stemming "connecting" -> "connect" is undesirable when
-- matching a function name verbatim).
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    language        TEXT,
    content         TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector   tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_search_vector_idx ON documents USING GIN (search_vector);
