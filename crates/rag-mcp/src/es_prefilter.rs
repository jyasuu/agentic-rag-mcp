//! Elasticsearch + `ik_analyzer` `PreFilterStrategy` (SPEC.md user story 11):
//! Chinese-language keyword search with proper word segmentation, returning
//! ES-highlight-based snippets so matched terms are visible in context
//! (user story 7). This is the primary pre-filter for the corpus's majority
//! Chinese content; the funnel runs it alongside the Postgres `tsvector`
//! strategy (English/code identifiers) and merges hits.
//!
//! The index this strategy searches is created by the CDC sync with the
//! `ik_max_word` analyzer on `content` (see `es.rs::ensure_index`); the
//! search-time analysis in the match query uses the same analyzer so
//! segmentation is consistent at index and query time.

use async_trait::async_trait;
use rag_core::{PreFilterHit, PreFilterStrategy, PreFilterStrategyKind, RagError, RagResult};

use crate::es::{EsClient, EsSearchHit, IK_ANALYZER};

pub struct EsPreFilter {
    client: EsClient,
    index: String,
}

impl EsPreFilter {
    pub fn new(client: EsClient, index: impl Into<String>) -> Self {
        Self {
            client,
            index: index.into(),
        }
    }
}

/// Maps raw ES hits into `PreFilterHit`s: BM25 `_score` as the raw score and
/// the first query-aware highlight fragment as the snippet. Pure so it can be
/// unit-tested without a live cluster.
fn map_hits(hits: Vec<EsSearchHit>) -> Vec<PreFilterHit> {
    hits.into_iter()
        .map(|h| PreFilterHit {
            id: h.id,
            source: h.source,
            raw_score: h.score.unwrap_or(0.0),
            highlighted_snippet: h.highlight.into_iter().next(),
            which_strategy: PreFilterStrategyKind::Elasticsearch,
        })
        .collect()
}

#[async_trait]
impl PreFilterStrategy for EsPreFilter {
    async fn search(&self, query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>> {
        let hits = self
            .client
            .search(&self.index, query, limit, IK_ANALYZER)
            .await
            .map_err(|e| RagError::PreFilter(format!("Elasticsearch search failed: {e}")))?;
        Ok(map_hits(hits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::es::IK_ANALYZER;

    #[test]
    fn map_hits_uses_bm25_score_and_first_highlight_fragment() {
        let hits = vec![
            EsSearchHit {
                id: "doc-1".into(),
                score: Some(2.5),
                source: "wiki/zh.md".into(),
                highlight: vec![
                    "系统发生<em>连接</em>".into(),
                    "<em>失败</em>错误码".into(),
                ],
            },
            EsSearchHit {
                id: "doc-2".into(),
                score: None,
                source: "wiki/errors.md".into(),
                highlight: vec![],
            },
        ];

        let mapped = map_hits(hits);
        assert_eq!(mapped[0].id, "doc-1");
        assert_eq!(mapped[0].raw_score, 2.5);
        assert_eq!(mapped[0].highlighted_snippet.as_deref(), Some("系统发生<em>连接</em>"));
        assert_eq!(mapped[0].which_strategy, PreFilterStrategyKind::Elasticsearch);
        assert_eq!(mapped[1].raw_score, 0.0, "missing score defaults to 0");
        assert_eq!(mapped[1].highlighted_snippet, None);
    }

    // Real-Elasticsearch integration tests, mirroring the Postgres test
    // pattern: run against `RAG_MCP_ELASTICSEARCH_URL`, skipped (not failed)
    // when unset. Each test uses its own uniquely-named index so concurrent
    // test runs never interfere, and polls for searchability because ES is
    // near-real-time, not write-through.
    const TEST_INDEX_PREFIX: &str = "es-";

    async fn test_client() -> Option<EsClient> {
        let url = std::env::var("RAG_MCP_ELASTICSEARCH_URL").ok()?;
        // Generous timeout: the whole test suite runs against the same
        // single-node cluster concurrently, so a request can take several
        // seconds under load.
        Some(
            EsClient::new(url, Duration::from_secs(30))
                .expect("RAG_MCP_ELASTICSEARCH_URL set but client failed to build"),
        )
    }

    async fn unique_index(client: &EsClient) -> String {
        let index = format!("{}{}", TEST_INDEX_PREFIX, crate::testutil::unique_token("idx"));
        client
            .ensure_index(&index, IK_ANALYZER)
            .await
            .expect("ensure_index should succeed");
        index
    }

    async fn cleanup_index(client: &EsClient, index: &str) {
        let url = format!("{}/{}", client.base_url().trim_end_matches('/'), index);
        let resp = client
            .http()
            .delete(&url)
            .send()
            .await
            .expect("index delete should be reachable");
        assert!(resp.status().is_success(), "index cleanup should succeed");
    }

    /// ES is near-real-time: poll the strategy until the expected id is
    /// visible (or give up after a few seconds).
    async fn search_until_visible(
        strategy: &EsPreFilter,
        query: &str,
        expected_id: &str,
    ) -> Vec<PreFilterHit> {
        for _ in 0..20 {
            let hits = strategy.search(query, 10).await.expect("search should succeed");
            if hits.iter().any(|h| h.id == expected_id) {
                return hits;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        strategy.search(query, 10).await.expect("search should succeed")
    }

    #[tokio::test]
    async fn chinese_phrase_query_returns_segmented_match_with_highlight() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        client
            .index_document(&index, "zh-1", "wiki/zh.md", "系统发生连接失败错误码需要重试")
            .await
            .expect("indexing should succeed");
        client
            .index_document(&index, "zh-2", "wiki/zh.md", "今天天气晴朗适合外出")
            .await
            .expect("indexing should succeed");

        let strategy = EsPreFilter::new(client.clone(), index.clone());
        let hits = search_until_visible(&strategy, "连接失败", "zh-1").await;

        let first = hits
            .iter()
            .find(|h| h.id == "zh-1")
            .expect("zh-1 should be in results");
        assert_eq!(first.which_strategy, PreFilterStrategyKind::Elasticsearch);
        assert!(
            first.raw_score > 0.0,
            "BM25 score should be positive for a real match"
        );
        let snippet = first.highlighted_snippet.as_deref().expect("ES hit should have a highlight");
        assert!(
            snippet.contains("<em>连接</em>") && snippet.contains("<em>失败</em>"),
            "highlight should surface the segmented match terms in context, got: {snippet}"
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn exact_code_token_returns_precise_match() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        let token = crate::testutil::unique_token("es-code");
        client
            .index_document(&index, "code-1", "wiki/errors.md", &format!("ERROR_{token}: connection refused"))
            .await
            .expect("indexing should succeed");

        let strategy = EsPreFilter::new(client.clone(), index.clone());
        let hits = search_until_visible(&strategy, &token, "code-1").await;

        assert!(
            hits.iter().any(|h| h.id == "code-1"),
            "exact code token should match its document"
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn no_match_returns_empty_not_error() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        client
            .index_document(&index, "n-1", "wiki/zh.md", "系统发生连接失败")
            .await
            .expect("indexing should succeed");

        let strategy = EsPreFilter::new(client.clone(), index.clone());
        // A token never indexed anywhere.
        let token = crate::testutil::unique_token("es-absent");
        let hits = strategy.search(&token, 10).await.expect("search should succeed");

        assert!(hits.is_empty());

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        for i in 0..3 {
            client
                .index_document(&index, &format!("lim-{i}"), "wiki/zh.md", &format!("支付接口调用失败需要处理案例{i}"))
                .await
                .expect("indexing should succeed");
        }

        let strategy = EsPreFilter::new(client.clone(), index.clone());
        // Poll until the docs are searchable (all three contain "失败"), then
        // assert a limit-2 search caps the count.
        for _ in 0..20 {
            if !strategy.search("失败", 10).await.expect("search should succeed").is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let hits = strategy.search("失败", 2).await.expect("search should succeed");

        assert!(!hits.is_empty(), "docs should be searchable by now");
        assert!(
            hits.len() <= 2,
            "limit=2 should cap results even though 3 rows match, got {}",
            hits.len()
        );

        cleanup_index(&client, &index).await;
    }
}
