-- Schema for the pg_trgm fallback pre-filter and the pgvector ANN stage.
-- See SPEC.md: pg_trgm is the fallback when ES is unavailable or unsynced,
-- and pgvector backs the cosine/L2 ANN stage. Like 0001_documents.sql, this
-- is owned by the external ingestion/schema process, not by this server.
-- These statements exist so the backend tickets have a concrete, testable
-- schema to implement against.
--
-- NOTE: keep SQL comments free of semicolons. The integration-test helper
-- that applies these migrations splits statements on a semicolon, and
-- comment text is not parsed.
--
-- Note: `CREATE EXTENSION` requires a superuser or a user granted the
-- extension's role. Run this migration with the schema-owner/ingestion role,
-- not the runtime `RAG_MCP_DATABASE_URL` user.
--
-- BGE-M3 produces 1024-dim dense embeddings (SPEC.md: "BGE-M3 via `ort`"), so
-- the ANN table uses vector(1024) with a cosine-distance HNSW index. The
-- `documents.content` GIN trigram index backs the fallback's
-- `word_similarity` scan. It is a soft dependency: the strategy's query
-- works without it on small corpora, the index is what keeps it fast at scale.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id          TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    embedding   vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_idx
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_content_trgm_idx
    ON documents USING GIN (content gin_trgm_ops);
