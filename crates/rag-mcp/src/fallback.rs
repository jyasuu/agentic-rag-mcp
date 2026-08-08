//! `FallbackPreFilter` — implements the SPEC.md fallback semantics: "pg_trgm
//! — fallback when ES is unavailable/unsynced." Wraps a primary strategy and
//! a fallback; `search` runs the primary, and only when it errors or returns
//! zero hits (an unavailable *or* unsynced cluster both produce that shape)
//! does the fallback run. The fallback's own `which_strategy` provenance is
//! preserved on its hits, so the funnel can still tell trigram results apart.

use async_trait::async_trait;
use rag_core::{PreFilterHit, PreFilterStrategy, RagResult};
use tracing::debug;

pub struct FallbackPreFilter {
    primary: Box<dyn PreFilterStrategy>,
    fallback: Box<dyn PreFilterStrategy>,
}

impl FallbackPreFilter {
    pub fn new(
        primary: Box<dyn PreFilterStrategy>,
        fallback: Box<dyn PreFilterStrategy>,
    ) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl PreFilterStrategy for FallbackPreFilter {
    async fn search(&self, query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>> {
        match self.primary.search(query, limit).await {
            Ok(hits) if !hits.is_empty() => Ok(hits),
            Ok(_) => {
                debug!(query, "primary pre-filter returned no hits; using fallback");
                self.fallback.search(query, limit).await
            }
            Err(e) => {
                debug!(query, error = %e, "primary pre-filter failed; using fallback");
                self.fallback.search(query, limit).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::{PreFilterHit, PreFilterStrategyKind, RagError};

    /// A scriptable strategy for exercising the fallback branches without
    /// any real backend. `RagError` isn't `Clone`, so errors are stored as
    /// messages and wrapped on each call.
    struct Stub {
        result: Result<Vec<PreFilterHit>, String>,
    }
    #[async_trait]
    impl PreFilterStrategy for Stub {
        async fn search(&self, _query: &str, _limit: usize) -> RagResult<Vec<PreFilterHit>> {
            match &self.result {
                Ok(hits) => Ok(hits.clone()),
                Err(msg) => Err(RagError::PreFilter(msg.clone())),
            }
        }
    }

    fn hit(id: &str, kind: PreFilterStrategyKind) -> PreFilterHit {
        PreFilterHit {
            id: id.into(),
            source: "wiki/test.md".into(),
            raw_score: 1.0,
            highlighted_snippet: None,
            which_strategy: kind,
        }
    }

    #[tokio::test]
    async fn uses_primary_when_it_succeeds() {
        let fb = FallbackPreFilter::new(
            Box::new(Stub { result: Ok(vec![hit("p1", PreFilterStrategyKind::Elasticsearch)]) }),
            Box::new(Stub { result: Ok(vec![hit("f1", PreFilterStrategyKind::Trigram)]) }),
        );
        let hits = fb.search("query", 10).await.unwrap();
        assert_eq!(hits[0].id, "p1");
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Elasticsearch);
    }

    #[tokio::test]
    async fn falls_back_when_primary_errors() {
        let fb = FallbackPreFilter::new(
            Box::new(Stub {
                result: Err("cluster down".to_string()),
            }),
            Box::new(Stub { result: Ok(vec![hit("f1", PreFilterStrategyKind::Trigram)]) }),
        );
        let hits = fb.search("query", 10).await.unwrap();
        assert_eq!(hits[0].id, "f1");
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Trigram);
    }

    #[tokio::test]
    async fn falls_back_when_primary_returns_nothing() {
        // The "unsynced" case: cluster is reachable but has no matching
        // rows, so the fallback must still return results.
        let fb = FallbackPreFilter::new(
            Box::new(Stub { result: Ok(vec![]) }),
            Box::new(Stub { result: Ok(vec![hit("f1", PreFilterStrategyKind::Trigram)]) }),
        );
        let hits = fb.search("query", 10).await.unwrap();
        assert_eq!(hits[0].id, "f1");
    }

    #[tokio::test]
    async fn propagates_fallback_error_when_both_fail() {
        let fb = FallbackPreFilter::new(
            Box::new(Stub {
                result: Err("cluster down".to_string()),
            }),
            Box::new(Stub {
                result: Err("postgres down".to_string()),
            }),
        );
        let err = fb.search("query", 10).await.unwrap_err();
        assert!(err.to_string().contains("postgres down"));
    }
}
