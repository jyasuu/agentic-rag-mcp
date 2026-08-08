pub mod funnel;
pub mod scoring;
pub mod traits;
pub mod types;

pub use funnel::RetrievalFunnel;
pub use scoring::{ScoringConfig, ShortCircuitConfig};
pub use traits::{AnnClient, ContentStore, Embedder, PreFilterStrategy};
pub use types::{
    AnnHit, Document, PreFilterHit, PreFilterStrategyKind, RagError, RagResult, ScoredResult,
    SearchFilters, SearchMode,
};

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Minimal in-memory fakes proving `RetrievalFunnel` is testable without
    /// any real Postgres/ES/ort dependency — this is the point of the seam.
    struct FakePreFilter(Vec<PreFilterHit>);
    #[async_trait]
    impl PreFilterStrategy for FakePreFilter {
        async fn search(&self, _query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>> {
            Ok(self.0.iter().take(limit).cloned().collect())
        }
    }

    struct FakeAnn(Vec<AnnHit>);
    #[async_trait]
    impl AnnClient for FakeAnn {
        async fn search(&self, _embedding: &[f32], limit: usize) -> RagResult<Vec<AnnHit>> {
            Ok(self.0.iter().take(limit).cloned().collect())
        }
    }

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

    fn confident_prefilter_hit() -> PreFilterHit {
        PreFilterHit {
            id: "doc-1".into(),
            source: "fake".into(),
            raw_score: 0.9,
            highlighted_snippet: Some("...matched term...".into()),
            which_strategy: PreFilterStrategyKind::Elasticsearch,
        }
    }

    #[tokio::test]
    async fn hybrid_mode_short_circuits_ann_when_prefilter_is_confident() {
        // 3 confident hits should clear the default ShortCircuitConfig
        // (min_hit_count: 3, min_top_score: 0.5), so ANN must not run.
        let hits = vec![
            confident_prefilter_hit(),
            PreFilterHit { id: "doc-2".into(), ..confident_prefilter_hit() },
            PreFilterHit { id: "doc-3".into(), ..confident_prefilter_hit() },
        ];
        let funnel = RetrievalFunnel::new(
            vec![Box::new(FakePreFilter(hits))],
            Box::new(FakeAnn(vec![AnnHit {
                id: "should-not-appear".into(),
                source: "fake".into(),
                similarity: 0.99,
                content_preview: "irrelevant".into(),
            }])),
            Box::new(FakeEmbedder),
            Box::new(FakeContentStore),
        );

        let results = funnel
            .search("exact term", SearchMode::Hybrid, SearchFilters::default(), None)
            .await
            .unwrap();

        assert!(results.iter().all(|r| !r.matched_ann));
        assert!(results.iter().any(|r| r.id == "doc-1"));
    }

    #[tokio::test]
    async fn hybrid_mode_falls_through_to_ann_when_prefilter_is_weak() {
        let funnel = RetrievalFunnel::new(
            vec![Box::new(FakePreFilter(vec![]))],
            Box::new(FakeAnn(vec![AnnHit {
                id: "doc-semantic".into(),
                source: "fake".into(),
                similarity: 0.8,
                content_preview: "a long piece of semantically related content".into(),
            }])),
            Box::new(FakeEmbedder),
            Box::new(FakeContentStore),
        );

        let results = funnel
            .search("vague intent query", SearchMode::Hybrid, SearchFilters::default(), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].matched_ann);
        assert_eq!(results[0].id, "doc-semantic");
    }

    #[tokio::test]
    async fn keyword_mode_never_runs_ann_even_if_prefilter_is_weak() {
        let funnel = RetrievalFunnel::new(
            vec![Box::new(FakePreFilter(vec![]))],
            Box::new(FakeAnn(vec![AnnHit {
                id: "should-not-appear".into(),
                source: "fake".into(),
                similarity: 0.99,
                content_preview: "irrelevant".into(),
            }])),
            Box::new(FakeEmbedder),
            Box::new(FakeContentStore),
        );

        let results = funnel
            .search("exact-code-123", SearchMode::Keyword, SearchFilters::default(), None)
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn fetch_by_id_delegates_to_content_store() {
        let funnel = RetrievalFunnel::new(
            vec![Box::new(FakePreFilter(vec![]))],
            Box::new(FakeAnn(vec![])),
            Box::new(FakeEmbedder),
            Box::new(FakeContentStore),
        );

        let doc = funnel.fetch_by_id("doc-1").await.unwrap();
        assert_eq!(doc.id, "doc-1");
        assert_eq!(doc.content, "full content");
    }
}
