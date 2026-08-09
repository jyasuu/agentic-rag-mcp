//! `FallbackRetrievalBackend` — wraps the Elasticsearch primary and the
//! Postgres tsvector keyword fallback. Keyword-mode searches run the primary,
//! and only when it errors or returns zero hits (an unavailable *or* unsynced
//! cluster both produce that shape) does the fallback run. The fallback's own
//! `which_strategy` provenance is preserved on its hits, so callers can still
//! tell tsvector results apart.
//!
//! Semantic and hybrid searches never fall back: kNN cannot be served by
//! tsvector, so the ES error surfaces as-is (the clear error class the spec
//! requires) rather than silently degrading to keyword-only results.

use async_trait::async_trait;
use rag_core::{RagResult, RankedHit, RetrievalBackend, RetrievalMode};
use tracing::debug;

pub struct FallbackRetrievalBackend {
    primary: Box<dyn RetrievalBackend>,
    keyword_fallback: Box<dyn RetrievalBackend>,
}

impl FallbackRetrievalBackend {
    pub fn new(
        primary: Box<dyn RetrievalBackend>,
        keyword_fallback: Box<dyn RetrievalBackend>,
    ) -> Self {
        Self {
            primary,
            keyword_fallback,
        }
    }
}

#[async_trait]
impl RetrievalBackend for FallbackRetrievalBackend {
    async fn search(
        &self,
        mode: RetrievalMode,
        keyword: Option<&str>,
        query_vector: Option<&[f32]>,
        limit: usize,
    ) -> RagResult<Vec<RankedHit>> {
        if !matches!(mode, RetrievalMode::Keyword) {
            return self
                .primary
                .search(mode, keyword, query_vector, limit)
                .await;
        }

        match self.primary.search(mode, keyword, query_vector, limit).await {
            Ok(hits) if !hits.is_empty() => Ok(hits),
            Ok(_) => {
                debug!(query = ?keyword, "primary keyword search returned no hits; using tsvector fallback");
                self.keyword_fallback
                    .search(mode, keyword, query_vector, limit)
                    .await
            }
            Err(e) => {
                debug!(query = ?keyword, error = %e, "primary keyword search failed; using tsvector fallback");
                self.keyword_fallback
                    .search(mode, keyword, query_vector, limit)
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::{PreFilterStrategyKind, RagError};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A scriptable backend for exercising the fallback branches without any
    /// real engine. `RagError` isn't `Clone`, so errors are stored as messages
    /// and wrapped on each call; the call count lets tests assert that
    /// semantic/hybrid modes never touch the fallback.
    struct Stub {
        calls: Arc<AtomicUsize>,
        result: Result<Vec<RankedHit>, String>,
    }
    #[async_trait]
    impl RetrievalBackend for Stub {
        async fn search(
            &self,
            _mode: RetrievalMode,
            _keyword: Option<&str>,
            _query_vector: Option<&[f32]>,
            _limit: usize,
        ) -> RagResult<Vec<RankedHit>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.result {
                Ok(hits) => Ok(hits.clone()),
                Err(msg) => Err(RagError::PreFilter(msg.clone())),
            }
        }
    }

    fn stub(calls: Arc<AtomicUsize>, result: Result<Vec<RankedHit>, String>) -> Stub {
        Stub { calls, result }
    }

    fn hit(id: &str, kind: PreFilterStrategyKind) -> RankedHit {
        RankedHit {
            id: id.into(),
            source: "wiki/test.md".into(),
            score: 1.0,
            snippet: "snippet".into(),
            which_strategy: kind,
            matched_ann: false,
        }
    }

    #[tokio::test]
    async fn keyword_uses_primary_when_it_succeeds() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fb = FallbackRetrievalBackend::new(
            Box::new(stub(calls.clone(), Ok(vec![hit("p1", PreFilterStrategyKind::Elasticsearch)]))),
            Box::new(stub(calls.clone(), Ok(vec![hit("f1", PreFilterStrategyKind::Tsvector)]))),
        );

        let hits = fb.search(RetrievalMode::Keyword, Some("term"), None, 10).await.unwrap();

        assert_eq!(hits[0].id, "p1");
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Elasticsearch);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "fallback must not run");
    }

    #[tokio::test]
    async fn keyword_falls_back_when_primary_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fb = FallbackRetrievalBackend::new(
            Box::new(stub(calls.clone(), Err("cluster down".to_string()))),
            Box::new(stub(calls.clone(), Ok(vec![hit("f1", PreFilterStrategyKind::Tsvector)]))),
        );

        let hits = fb.search(RetrievalMode::Keyword, Some("term"), None, 10).await.unwrap();

        assert_eq!(hits[0].id, "f1");
        assert_eq!(hits[0].which_strategy, PreFilterStrategyKind::Tsvector);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn keyword_falls_back_when_primary_returns_nothing() {
        // The "unsynced" case: cluster is reachable but has no matching
        // rows, so the fallback must still return results.
        let calls = Arc::new(AtomicUsize::new(0));
        let fb = FallbackRetrievalBackend::new(
            Box::new(stub(calls.clone(), Ok(vec![]))),
            Box::new(stub(calls.clone(), Ok(vec![hit("f1", PreFilterStrategyKind::Tsvector)]))),
        );

        let hits = fb.search(RetrievalMode::Keyword, Some("term"), None, 10).await.unwrap();

        assert_eq!(hits[0].id, "f1");
    }

    #[tokio::test]
    async fn keyword_propagates_fallback_error_when_both_fail() {
        let fb = FallbackRetrievalBackend::new(
            Box::new(stub(Arc::new(AtomicUsize::new(0)), Err("cluster down".to_string()))),
            Box::new(stub(Arc::new(AtomicUsize::new(0)), Err("postgres down".to_string()))),
        );

        let err = fb.search(RetrievalMode::Keyword, Some("term"), None, 10).await.unwrap_err();
        assert!(err.to_string().contains("postgres down"));
    }

    #[tokio::test]
    async fn semantic_mode_never_falls_back() {
        let fallback_calls = Arc::new(AtomicUsize::new(0));
        let fb = FallbackRetrievalBackend::new(
            Box::new(stub(Arc::new(AtomicUsize::new(0)), Err("cluster down".to_string()))),
            Box::new(stub(fallback_calls.clone(), Ok(vec![hit("f1", PreFilterStrategyKind::Tsvector)]))),
        );

        let err = fb
            .search(RetrievalMode::Semantic, None, Some(&[0.1f32]), 10)
            .await
            .expect_err("semantic mode must surface the ES error, not fall back");

        assert!(err.to_string().contains("cluster down"));
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 0, "tsvector must not run for kNN");
    }

    #[tokio::test]
    async fn hybrid_mode_never_falls_back() {
        let fallback_calls = Arc::new(AtomicUsize::new(0));
        let fb = FallbackRetrievalBackend::new(
            Box::new(stub(Arc::new(AtomicUsize::new(0)), Err("cluster down".to_string()))),
            Box::new(stub(fallback_calls.clone(), Ok(vec![hit("f1", PreFilterStrategyKind::Tsvector)]))),
        );

        let err = fb
            .search(RetrievalMode::Hybrid, Some("连接失败"), Some(&[0.1f32]), 10)
            .await
            .expect_err("hybrid mode must surface the ES error, not degrade to keyword-only");

        assert!(err.to_string().contains("cluster down"));
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 0);
    }
}
