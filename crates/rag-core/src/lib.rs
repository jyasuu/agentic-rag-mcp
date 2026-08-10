pub mod funnel;
pub mod scoring;
pub mod traits;
pub mod types;

pub use funnel::RetrievalFunnel;
pub use scoring::{
    FusionWeights, HybridFusion, HybridFusionConfig, RrfConfig, ScoreNormalization,
};
pub use traits::{ContentStore, Embedder, RetrievalBackend};
pub use types::{
    Document, PreFilterStrategyKind, RagError, RagResult, RankedHit, RetrievalMode, ScoredResult,
    SearchFilters, SearchMode,
};

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct FakeEmbedder;
    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, _text: &str) -> RagResult<Vec<f32>> {
            Ok(vec![0.0; 4])
        }
    }

    struct FakeContentStore;
    #[async_trait]
    impl ContentStore for FakeContentStore {
        async fn fetch(&self, id: &str) -> RagResult<Document> {
            Ok(Document {
                id: id.to_string(),
                source: "fake".into(),
                content: "full content".into(),
                metadata: serde_json::json!({}),
            })
        }
    }

    /// Records the dispatch the funnel requested (mode + which optional
    /// keyword/vector) and returns canned hits — the fake that keeps the mode
    /// mapping testable without any real Postgres/ES/ort dependency.
    #[derive(Clone)]
    struct RecordingBackend {
        calls: Arc<std::sync::Mutex<Vec<(RetrievalMode, Option<String>, Option<usize>)>>>,
        hits: Vec<RankedHit>,
    }
    #[async_trait]
    impl RetrievalBackend for RecordingBackend {
        async fn search(
            &self,
            mode: RetrievalMode,
            keyword: Option<&str>,
            query_vector: Option<&[f32]>,
            limit: usize,
        ) -> RagResult<Vec<RankedHit>> {
            self.calls.lock().unwrap().push((
                mode,
                keyword.map(|k| k.to_string()),
                query_vector.map(|v| v.len()),
            ));
            Ok(self.hits.iter().take(limit).cloned().collect())
        }
    }

    fn hit(id: &str) -> RankedHit {
        RankedHit {
            id: id.to_string(),
            source: "fake".into(),
            score: 0.5,
            snippet: "…snippet…".into(),
            which_strategy: PreFilterStrategyKind::Elasticsearch,
            matched_ann: false,
        }
    }

    fn funnel(backend: Arc<RecordingBackend>) -> RetrievalFunnel {
        RetrievalFunnel::new(
            Box::new((*backend).clone()),
            Box::new(FakeEmbedder),
            Box::new(FakeContentStore),
        )
    }

    #[tokio::test]
    async fn keyword_mode_sends_bm25_request_without_vector() {
        let backend = Arc::new(RecordingBackend {
            calls: Default::default(),
            hits: vec![hit("doc-1")],
        });
        let funnel = funnel(backend.clone());

        let results = funnel
            .search("exact-code-123", SearchMode::Keyword, SearchFilters::default(), None)
            .await
            .unwrap();

        let calls = backend.calls.lock().unwrap();
        assert_eq!(
            calls.as_slice(),
            &[(RetrievalMode::Keyword, Some("exact-code-123".to_string()), None)]
        );
        assert_eq!(results.len(), 1);
        assert!(!results[0].matched_ann, "keyword hits never matched via ANN");
    }

    #[tokio::test]
    async fn semantic_mode_embeds_and_sends_knn_request_without_keyword() {
        let backend = Arc::new(RecordingBackend {
            calls: Default::default(),
            hits: vec![RankedHit { matched_ann: true, ..hit("doc-semantic") }],
        });
        let funnel = funnel(backend.clone());

        let results = funnel
            .search("vague intent query", SearchMode::Semantic, SearchFilters::default(), None)
            .await
            .unwrap();

        let calls = backend.calls.lock().unwrap();
        assert_eq!(
            calls.as_slice(),
            &[(RetrievalMode::Semantic, None, Some(4))],
            "semantic mode must pass the query vector and no keyword"
        );
        assert!(results[0].matched_ann, "kNN request means the hit came via ANN");
    }

    #[tokio::test]
    async fn hybrid_mode_embeds_and_sends_keyword_and_vector() {
        let backend = Arc::new(RecordingBackend {
            calls: Default::default(),
            hits: vec![hit("doc-hybrid")],
        });
        let funnel = funnel(backend.clone());

        let results = funnel
            .search("连接失败", SearchMode::Hybrid, SearchFilters::default(), None)
            .await
            .unwrap();

        let calls = backend.calls.lock().unwrap();
        assert_eq!(
            calls.as_slice(),
            &[(RetrievalMode::Hybrid, Some("连接失败".to_string()), Some(4))],
            "hybrid mode must pass both the keyword and the query vector"
        );
        assert_eq!(results[0].score, 0.5, "engine score must pass through untouched");
        assert_eq!(
            results[0].matched_via,
            vec![PreFilterStrategyKind::Elasticsearch],
            "hybrid hits report the ES strategy"
        );
    }

    #[tokio::test]
    async fn keyword_search_helper_maps_to_keyword_mode() {
        let backend = Arc::new(RecordingBackend {
            calls: Default::default(),
            hits: vec![hit("doc-1")],
        });
        let funnel = funnel(backend.clone());

        let _ = funnel.keyword_search("exact-code-123", None).await.unwrap();
        let calls = backend.calls.lock().unwrap();
        assert_eq!(
            calls.as_slice(),
            &[(RetrievalMode::Keyword, Some("exact-code-123".to_string()), None)]
        );
    }

    #[tokio::test]
    async fn vector_search_helper_maps_to_semantic_mode() {
        let backend = Arc::new(RecordingBackend {
            calls: Default::default(),
            hits: vec![hit("doc-semantic")],
        });
        let funnel = funnel(backend.clone());

        let _ = funnel.vector_search("vague intent query", None).await.unwrap();
        let calls = backend.calls.lock().unwrap();
        assert_eq!(
            calls.as_slice(),
            &[(RetrievalMode::Semantic, None, Some(4))]
        );
    }

    struct FailingBackend;
    #[async_trait]
    impl RetrievalBackend for FailingBackend {
        async fn search(
            &self,
            _mode: RetrievalMode,
            _keyword: Option<&str>,
            _query_vector: Option<&[f32]>,
            _limit: usize,
        ) -> RagResult<Vec<RankedHit>> {
            Err(RagError::Ann("backend unavailable".into()))
        }
    }

    #[tokio::test]
    async fn backend_error_surfaces_for_semantic_mode() {
        let funnel =
            RetrievalFunnel::new(Box::new(FailingBackend), Box::new(FakeEmbedder), Box::new(FakeContentStore));

        let err = funnel
            .search("vague intent query", SearchMode::Semantic, SearchFilters::default(), None)
            .await
            .expect_err("semantic mode with an unreachable backend must error, not return empty results");

        match err {
            RagError::Ann(msg) => assert!(msg.contains("backend unavailable")),
            other => panic!("expected Ann error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fetch_by_id_delegates_to_content_store() {
        let backend = Arc::new(RecordingBackend {
            calls: Default::default(),
            hits: vec![],
        });
        let funnel = funnel(backend);

        let doc = funnel.fetch_by_id("doc-1").await.unwrap();
        assert_eq!(doc.id, "doc-1");
        assert_eq!(doc.content, "full content");
    }
}
