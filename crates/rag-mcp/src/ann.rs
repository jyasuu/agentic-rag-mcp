//! pgvector `AnnClient` — the ANN (approximate nearest neighbour) stage of the
//! funnel (SPEC.md implementation decision 2: "pgvector cosine/L2 search").
//! Given a pre-computed query embedding, queries the `chunk_embeddings` table
//! (see `migrations/0002_pg_trgm_pgvector.sql`) with pgvector's `<=>` cosine
//! distance operator, backed by the HNSW index on `embedding`, and converts
//! distance back into similarity (`similarity = 1 - distance`, i.e. cosine
//! similarity in [-1, 1]).

use async_trait::async_trait;
use rag_core::{AnnClient, AnnHit, RagError, RagResult};
use sqlx::PgPool;
use sqlx::Row;

pub struct PgvectorAnnClient {
    pool: PgPool,
}

impl PgvectorAnnClient {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl AnnClient for PgvectorAnnClient {
    async fn search(&self, embedding: &[f32], limit: usize) -> RagResult<Vec<AnnHit>> {
        // The query embedding is bound as a pgvector literal text (sqlx has
        // no native vector type) and cast with `::vector`.
        let literal = vector_literal(embedding);
        let rows = sqlx::query(
            r#"
            SELECT
                ce.id,
                d.source,
                d.content,
                (1 - (ce.embedding <=> $1::vector))::float4 AS similarity
            FROM chunk_embeddings ce
            JOIN documents d ON d.id = ce.id
            ORDER BY ce.embedding <=> $1::vector
            LIMIT $2
            "#,
        )
        .bind(&literal)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| RagError::Ann(format!("pgvector query failed: {e}")))?;

        Ok(rows
            .into_iter()
            .map(|row| AnnHit {
                id: row.get::<String, _>("id"),
                source: row.get::<String, _>("source"),
                similarity: row.get::<f32, _>("similarity"),
                content_preview: row.get::<String, _>("content"),
            })
            .collect())
    }
}

/// Renders `embedding` as a pgvector literal, e.g. `[0.1,0.2,0.3]`.
fn vector_literal(v: &[f32]) -> String {
    let body: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!("[{}]", body.join(","))
}

// Real-Postgres integration tests, following the trigram.rs pattern: run
// against `RAG_MCP_DATABASE_URL`, applying the schema idempotently and
// skipping (not failing) when the env var is unset. Fixtures use
// pseudo-random embeddings so concurrent runs of this file -- which search
// across all `chunk_embeddings` rows -- never match each other's fixtures.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{apply_schema, cleanup_documents, insert_document, test_pool, unique_token};

    /// A deterministic pseudo-random vector seeded by `seed`, so two fixtures
    /// built with different seeds are far apart in cosine space and the
    /// query's own vector lands closest to its fixture.
    fn vec(seed: u32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| ((seed as f64) + (i as f64) * 1.7).sin() as f32).collect()
    }

    async fn insert_embedding(pool: &PgPool, id: &str, embedding: &[f32]) {
        let literal = vector_literal(embedding);
        sqlx::query("INSERT INTO chunk_embeddings (id, embedding) VALUES ($1, $2::vector)")
            .bind(id)
            .bind(&literal)
            .execute(pool)
            .await
            .unwrap();
    }

    async fn test_db() -> Option<PgPool> {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return None;
        };
        apply_schema(&pool).await;
        Some(pool)
    }

    async fn test_search(embedding: &[f32], limit: usize) -> (Option<PgPool>, Vec<AnnHit>) {
        let Some(pool) = test_db().await else {
            return (None, Vec::new());
        };
        let hits = PgvectorAnnClient::new(pool.clone())
            .search(embedding, limit)
            .await
            .expect("search should succeed");
        (Some(pool), hits)
    }

    #[tokio::test]
    async fn returns_ranked_hits_by_cosine_similarity() {
        // `unique_token` already embeds pid + counter, so it doubles as the
        // cleanable id prefix (stale rows from other runs never match it).
        let prefix = unique_token("ann");
        let id_a = format!("{prefix}-a");
        let id_b = format!("{prefix}-b");
        let Some(pool) = test_db().await else {
            return;
        };

        let va = vec(1, 1024);
        let vb = vec(2, 1024);
        insert_document(&pool, &id_a, "wiki/api.md", &format!("alpha {prefix}")).await;
        insert_document(&pool, &id_b, "wiki/errors.md", &format!("beta {prefix}")).await;
        insert_embedding(&pool, &id_a, &va).await;
        insert_embedding(&pool, &id_b, &vb).await;

        // Query close to `va` (same seed), so `id_a` must rank first. The
        // shared table may hold unrelated rows from other runs, so ordering
        // is asserted only among this test's own prefix-matched fixtures.
        let hits = PgvectorAnnClient::new(pool.clone())
            .search(&vec(1, 1024), 10)
            .await
            .unwrap();

        let ids: Vec<&str> = hits
            .iter()
            .filter(|h| h.id.starts_with(&prefix))
            .map(|h| h.id.as_str())
            .collect();
        assert_eq!(ids, vec![id_a.as_str(), id_b.as_str()], "closest fixture should rank first");
        assert!(
            hits.iter().any(|h| h.id == id_a && h.similarity > 0.99),
            "near-identical vector should score near 1.0"
        );

        cleanup_documents(&pool, &prefix).await;
    }

    #[tokio::test]
    async fn no_fixtures_does_not_error() {
        let token = unique_token("ann-none");
        // The shared table may contain rows from other tests (search ranks,
        // it never filters), so "empty" is not assertable -- the guarantees
        // here are: the call succeeds, and none of the hits are this test's
        // fixtures (it inserted nothing).
        let (pool, hits) = test_search(&vec(7, 1024), 10).await;
        if pool.is_none() {
            return;
        }
        assert!(hits.iter().all(|h| !h.id.contains(&token)));
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let prefix = unique_token("ann");
        let Some(pool) = test_db().await else {
            return;
        };

        for i in 0..3 {
            let id = format!("{prefix}-{i}");
            insert_document(&pool, &id, "wiki/test.md", &format!("doc {i} {prefix}")).await;
            insert_embedding(&pool, &id, &vec(10 + i, 1024)).await;
        }

        // All three fixtures are far from the query but the limit must still cap.
        let hits = PgvectorAnnClient::new(pool.clone()).search(&vec(99, 1024), 2).await.unwrap();
        assert_eq!(hits.len(), 2, "limit=2 should cap results");

        cleanup_documents(&pool, &prefix).await;
    }
}
