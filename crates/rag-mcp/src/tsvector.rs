//! Postgres `tsvector` keyword fallback `RetrievalBackend` — the exact-term
//! path for English/code content (identifiers, error codes) when Elasticsearch
//! errors mid-flight or hasn't caught up. Queries the `documents` table's
//! generated `search_vector` column (see `migrations/0001_documents.sql`),
//! ranking with `ts_rank` and producing a query-aware snippet via
//! `ts_headline` so results carry a highlighted match in context, matching the
//! shape ES-backed results provide.
//!
//! Keyword-only: semantic and hybrid modes return a clear error, because kNN
//! cannot fall back to tsvector — the fallback wrapper only ever routes the
//! keyword facet here.
//!
//! `simple` config is used throughout (not `english`) to avoid stemming
//! mangling identifier-style tokens — see the migration file for the full
//! rationale.

use async_trait::async_trait;
use rag_core::{
    PreFilterStrategyKind, RagError, RagResult, RankedHit, RetrievalBackend, RetrievalMode,
};
use sqlx::PgPool;
use sqlx::Row;

pub struct TsvectorRetrievalBackend {
    pool: PgPool,
}

impl TsvectorRetrievalBackend {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl RetrievalBackend for TsvectorRetrievalBackend {
    async fn search(
        &self,
        mode: RetrievalMode,
        keyword: Option<&str>,
        _query_vector: Option<&[f32]>,
        limit: usize,
    ) -> RagResult<Vec<RankedHit>> {
        if !matches!(mode, RetrievalMode::Keyword) {
            return Err(RagError::PreFilter(
                "tsvector fallback supports keyword mode only — semantic and hybrid queries need Elasticsearch".into(),
            ));
        }
        let query = keyword.ok_or_else(|| {
            RagError::PreFilter("keyword mode requires a keyword query".into())
        })?;
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
            .map(|row| RankedHit {
                id: row.get::<String, _>("id"),
                source: row.get::<String, _>("source"),
                score: row.get::<f32, _>("rank"),
                snippet: row.get::<String, _>("snippet"),
                which_strategy: PreFilterStrategyKind::Tsvector,
                matched_ann: false,
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

        let backend = TsvectorRetrievalBackend::new(pool.clone());
        let hits = backend
            .search(RetrievalMode::Keyword, Some(&token), None, 10)
            .await
            .unwrap();

        assert_eq!(hits.len(), 1, "expected exactly one match for a unique token");
        assert_eq!(hits[0].id, id);
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Tsvector);
        assert!(!hits[0].matched_ann);
        assert!(!hits[0].snippet.is_empty());

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

        let backend = TsvectorRetrievalBackend::new(pool.clone());
        let hits = backend
            .search(RetrievalMode::Keyword, Some(&fn_name), None, 10)
            .await
            .unwrap();

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

        let backend = TsvectorRetrievalBackend::new(pool.clone());
        // Never inserted anywhere -- guaranteed no match.
        let hits = backend
            .search(RetrievalMode::Keyword, Some(&token), None, 10)
            .await
            .unwrap();

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

        let backend = TsvectorRetrievalBackend::new(pool.clone());
        let hits = backend
            .search(RetrievalMode::Keyword, Some(&token), None, 2)
            .await
            .unwrap();

        assert_eq!(hits.len(), 2, "limit=2 should cap results even though 3 rows match");

        cleanup(&pool, &id_prefix).await;
    }

    #[tokio::test]
    async fn semantic_and_hybrid_modes_return_clear_error() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        let backend = TsvectorRetrievalBackend::new(pool);
        let err = backend
            .search(RetrievalMode::Hybrid, Some("连接失败"), Some(&[0.1f32]), 10)
            .await
            .expect_err("tsvector must refuse non-keyword modes");
        match err {
            RagError::PreFilter(msg) => {
                assert!(msg.contains("keyword mode only"), "got: {msg}");
            }
            other => panic!("expected PreFilter error, got {other:?}"),
        }
    }
}
