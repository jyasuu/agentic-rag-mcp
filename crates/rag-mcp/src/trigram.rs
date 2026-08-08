//! Postgres `pg_trgm` fallback `PreFilterStrategy` (SPEC.md user story 13):
//! keyword search that still functions when Elasticsearch is unavailable or
//! unsynced.
//!
//! Uses `word_similarity` -- the maximum trigram similarity between the query
//! string and any single word in a document's content -- rather than whole-
//! string `similarity`, because the corpus documents are large: a distinctive
//! term (function name, error code, CJK token) sits inside one word, and
//! whole-string trigram comparison would drown it in the rest of the
//! document's trigrams. See the migration (0002) for the backing GIN trigram
//! index; the query works without it, the index just keeps it fast at scale.
//!
//! Deliberately fuzzy: results are ranked by similarity, and the funnel's
//! short-circuit heuristic decides whether they're confident enough to skip
//! the ANN stage. No query-aware snippet is produced (`highlighted_snippet:
//! None`), so the funnel falls back to fixed truncation -- matching the
//! SPEC.md design (ES results highlight, ANN/trigram results truncate).

use async_trait::async_trait;
use rag_core::{PreFilterHit, PreFilterStrategy, PreFilterStrategyKind, RagError, RagResult};
use sqlx::PgPool;
use sqlx::Row;

const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.3;

pub struct TrigramPreFilter {
    pool: PgPool,
    threshold: f32,
}

impl TrigramPreFilter {
    pub fn new(pool: PgPool) -> Self {
        Self::with_threshold(pool, DEFAULT_SIMILARITY_THRESHOLD)
    }

    /// Test/calibration hook: the similarity threshold is a rough constant
    /// (see SPEC.md opportunity list -- pg_trgm activation triggers are
    /// deferred), so the ability to override it is kept behind this
    /// constructor rather than env config.
    pub fn with_threshold(pool: PgPool, threshold: f32) -> Self {
        Self { pool, threshold }
    }
}

#[async_trait]
impl PreFilterStrategy for TrigramPreFilter {
    async fn search(&self, query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>> {
        let rows = sqlx::query(
            r#"
            SELECT id, source, word_similarity($1, content) AS sim
            FROM documents
            WHERE word_similarity($1, content) >= $3
            ORDER BY sim DESC
            LIMIT $2
            "#,
        )
        .bind(query)
        .bind(limit as i64)
        .bind(self.threshold)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| {
            RagError::PreFilter(format!(
                "pg_trgm query failed (is the pg_trgm extension installed?): {e}"
            ))
        })?;

        Ok(rows
            .into_iter()
            .map(|row| PreFilterHit {
                id: row.get("id"),
                source: row.get("source"),
                raw_score: row.get::<f32, _>("sim"),
                // No query-aware highlight from trigram matching -- the
                // funnel truncates instead.
                highlighted_snippet: None,
                which_strategy: PreFilterStrategyKind::Trigram,
            })
            .collect())
    }
}

// Real-Postgres integration tests following the tsvector.rs pattern: run
// against `RAG_MCP_DATABASE_URL`, skipped (not failed) when unset. Each test
// applies the schema idempotently (via `testutil`) and embeds a unique token
// into both fixture content and query so concurrent test runs -- and unrelated
// rows in a shared dev database -- never interfere.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        apply_schema, cleanup_documents, insert_document, test_pool, unique_term, unique_token,
    };

    #[tokio::test]
    async fn exact_term_returns_matching_document() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        apply_schema(&pool).await;
        let token = unique_token("trgm-exact");
        let id = format!("trgm-{token}");
        // The distinctive term is a single random word with no scaffolding
        // around it: `word_similarity` compares the query against each whole
        // content word, so shared scaffolding words (e.g. `handle`,
        // `request`) fuzzy-match other tests' rows and stale rows left behind
        // by previous runs.
        let term = unique_term();
        insert_document(
            &pool,
            &id,
            "wiki/api.md",
            &format!("The function {term} validates the payload and charges the card."),
        )
        .await;

        let strategy = TrigramPreFilter::new(pool.clone());
        let hits = strategy.search(&term, 10).await.unwrap();

        assert_eq!(hits.len(), 1, "exact term should match");
        assert_eq!(hits[0].id, id);
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Trigram);
        assert!(hits[0].raw_score > 0.8, "verbatim term should score near 1.0");

        cleanup_documents(&pool, "trgm-exact").await;
    }

    #[tokio::test]
    async fn fuzzy_partial_term_still_matches() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        apply_schema(&pool).await;
        let token = unique_token("trgm-fuzzy");
        let id = format!("trgm-{token}");
        // A long distinctive word; the query is a truncated prefix of it --
        // the case where a naive exact matcher returns nothing but trigram
        // recovery succeeds. Using the current run's own random term (rather
        // than a fixed prefix like `supercalifragilisti`) keeps the fixture
        // isolated from other tests' and earlier runs' rows.
        let full = unique_term();
        let prefix = &full[..full.len() / 2];
        insert_document(
            &pool,
            &id,
            "wiki/errors.md",
            &format!("{full} is raised when the connection is dropped."),
        )
        .await;

        let strategy = TrigramPreFilter::new(pool.clone());
        let hits = strategy.search(prefix, 10).await.unwrap();

        assert!(
            hits.iter().any(|h| h.id == id),
            "fuzzy prefix should still recover the document: {hits:?}"
        );

        cleanup_documents(&pool, "trgm-fuzzy").await;
    }

    #[tokio::test]
    async fn no_match_returns_empty_not_error() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        apply_schema(&pool).await;
        let token = unique_term();

        let strategy = TrigramPreFilter::new(pool.clone());
        // Never inserted anywhere -- no trigrams can line up.
        let hits = strategy.search(&token, 10).await.unwrap();

        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        apply_schema(&pool).await;
        let token = unique_term();
        let id_prefix = format!("trgm-{}", unique_token("lim"));
        for i in 0..3 {
            insert_document(
                &pool,
                &format!("{id_prefix}-{i}"),
                "wiki/test.md",
                &format!("Document number {i} mentions the term {token}."),
            )
            .await;
        }

        let strategy = TrigramPreFilter::new(pool.clone());
        let hits = strategy.search(&token, 2).await.unwrap();

        assert_eq!(hits.len(), 2, "limit=2 should cap results even though 3 rows match");

        cleanup_documents(&pool, &id_prefix).await;
    }
}
