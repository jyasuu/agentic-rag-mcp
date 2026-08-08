//! Postgres `tsvector` `PreFilterStrategy` — the pre-filter path for
//! English/code content (identifiers, error codes) per SPEC.md. Queries the
//! `documents` table's generated `search_vector` column (see
//! `migrations/0001_documents.sql`), ranking with `ts_rank` and producing a
//! query-aware snippet via `ts_headline` so results carry a highlighted
//! match in context (SPEC.md user story 7), matching the shape ES-backed
//! results provide.
//!
//! `simple` config is used throughout (not `english`) to avoid stemming
//! mangling identifier-style tokens — see the migration file for the full
//! rationale.

use async_trait::async_trait;
use rag_core::{PreFilterHit, PreFilterStrategy, PreFilterStrategyKind, RagError, RagResult};
use sqlx::PgPool;
use sqlx::Row;

pub struct TsvectorPreFilter {
    pool: PgPool,
}

impl TsvectorPreFilter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl PreFilterStrategy for TsvectorPreFilter {
    async fn search(&self, query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>> {
        // `websearch_to_tsquery` tolerates arbitrary user input (quotes,
        // punctuation) without raising a syntax error, unlike
        // `to_tsquery`/`plainto_tsquery`'s stricter operator parsing --
        // appropriate here since the query text comes directly from an
        // agent, not a controlled search-syntax input.
        let rows = sqlx::query(
            r#"
            SELECT
                id,
                source,
                ts_rank(search_vector, websearch_to_tsquery('simple', $1)) AS rank,
                ts_headline('simple', content, websearch_to_tsquery('simple', $1),
                            'StartSel=<<, StopSel=>>, MaxWords=35, MinWords=15') AS snippet
            FROM documents
            WHERE search_vector @@ websearch_to_tsquery('simple', $1)
            ORDER BY rank DESC
            LIMIT $2
            "#,
        )
        .bind(query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| RagError::PreFilter(format!("tsvector query failed: {e}")))?;

        Ok(rows
            .into_iter()
            .map(|row| PreFilterHit {
                id: row.get::<String, _>("id"),
                source: row.get::<String, _>("source"),
                raw_score: row.get::<f32, _>("rank"),
                highlighted_snippet: Some(row.get::<String, _>("snippet")),
                which_strategy: PreFilterStrategyKind::Tsvector,
            })
            .collect())
    }
}

// Real-Postgres integration tests, per SPEC.md's testing decisions ("each
// [strategy] against a minimal fixture dataset, to isolate strategy-
// specific bugs from funnel orchestration bugs"). SPEC.md's stated
// preference is Testcontainers; this sandbox has no Docker available, so
// these run against `RAG_MCP_DATABASE_URL` directly instead (point it at a
// disposable local/CI Postgres with the `documents` table from
// migrations/0001_documents.sql applied). Skipped automatically -- not
// failed -- when that env var isn't set, so `cargo test` stays green
// without Postgres for anyone just working on other crates.
//
// Each test embeds a unique token (test name + pid + counter) into both its
// fixture content and its search query, rather than relying on exact hit
// counts against a table that may contain other rows -- `cargo test` runs
// tests in this file concurrently by default, and this table may also hold
// unrelated data (e.g. from manual exploration against a shared dev
// database), so uniqueness has to come from the content itself.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_token(test_name: &str) -> String {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("{test_name}{}{n}", std::process::id())
    }

    async fn test_pool() -> Option<PgPool> {
        let url = std::env::var("RAG_MCP_DATABASE_URL").ok()?;
        Some(
            PgPool::connect(&url)
                .await
                .expect("RAG_MCP_DATABASE_URL set but Postgres unreachable"),
        )
    }

    async fn insert(pool: &PgPool, id: &str, source: &str, content: &str) {
        sqlx::query("INSERT INTO documents (id, source, content) VALUES ($1, $2, $3)")
            .bind(id)
            .bind(source)
            .bind(content)
            .execute(pool)
            .await
            .unwrap();
    }

    async fn cleanup(pool: &PgPool, id_prefix: &str) {
        sqlx::query("DELETE FROM documents WHERE id LIKE $1")
            .bind(format!("{id_prefix}%"))
            .execute(pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn exact_error_code_returns_matching_document() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        let token = unique_token("errcode");
        let id = format!("tsv-{token}");
        insert(
            &pool,
            &id,
            "wiki/errors.md",
            &format!("{token} is raised when the client fails to establish a connection."),
        )
        .await;

        let strategy = TsvectorPreFilter::new(pool.clone());
        let hits = strategy.search(&token, 10).await.unwrap();

        assert_eq!(hits.len(), 1, "expected exactly one match for a unique token");
        assert_eq!(hits[0].id, id);
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Tsvector);
        assert!(hits[0].highlighted_snippet.is_some());

        cleanup(&pool, "tsv-errcode").await;
    }

    #[tokio::test]
    async fn exact_function_name_returns_matching_document() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        let token = unique_token("fnname");
        // Function-name-shaped: underscore-joined identifier, like the
        // corpus's real English/code content (SPEC.md).
        let fn_name = format!("process_{token}_request");
        let id = format!("tsv-{token}");
        insert(
            &pool,
            &id,
            "wiki/api.md",
            &format!("The function {fn_name} validates the payload and charges the card."),
        )
        .await;

        let strategy = TsvectorPreFilter::new(pool.clone());
        let hits = strategy.search(&fn_name, 10).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);

        cleanup(&pool, "tsv-fnname").await;
    }

    #[tokio::test]
    async fn no_match_returns_empty_not_error() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        let token = unique_token("nomatch");

        let strategy = TsvectorPreFilter::new(pool.clone());
        // Never inserted anywhere -- guaranteed no match.
        let hits = strategy.search(&token, 10).await.unwrap();

        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        let token = unique_token("limtest");
        let id_prefix = format!("tsv-{token}");

        for i in 0..3 {
            insert(
                &pool,
                &format!("{id_prefix}-{i}"),
                "wiki/test.md",
                &format!("Document number {i} mentions {token} for testing."),
            )
            .await;
        }

        let strategy = TsvectorPreFilter::new(pool.clone());
        let hits = strategy.search(&token, 2).await.unwrap();

        assert_eq!(hits.len(), 2, "limit=2 should cap results even though 3 rows match");

        cleanup(&pool, &id_prefix).await;
    }
}
