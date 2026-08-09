use tracing::{debug, instrument};

use crate::traits::{ContentStore, Embedder, RetrievalBackend};
use crate::types::{
    Document, RagResult, RankedHit, RetrievalMode, ScoredResult, SearchFilters, SearchMode,
};

/// Transport-independent retrieval service. This is the seam: MCP tool
/// handlers are thin wrappers around this struct's methods, so tests exercise
/// `RetrievalFunnel` directly (e.g. against real Elasticsearch + an embedder)
/// without touching the `rmcp`/axum layer at all.
///
/// Composes a single retrieval backend, the embedder, and the content store.
/// Ranking is the backend's job (BM25 / kNN / RRF all happen inside
/// Elasticsearch), so the funnel is a thin dispatch layer: it maps the
/// caller's `SearchMode` onto the backend's request shape and converts hits to
/// the stable `ScoredResult` response.
pub struct RetrievalFunnel {
    backend: Box<dyn RetrievalBackend>,
    embedder: Box<dyn Embedder>,
    content_store: Box<dyn ContentStore>,
    default_limit: usize,
}

impl RetrievalFunnel {
    pub fn new(
        backend: Box<dyn RetrievalBackend>,
        embedder: Box<dyn Embedder>,
        content_store: Box<dyn ContentStore>,
    ) -> Self {
        Self {
            backend,
            embedder,
            content_store,
            default_limit: 10,
        }
    }

    /// Primary entry point, backing the `search` MCP tool. Keyword mode is a
    /// single BM25 request; semantic and hybrid modes embed the query and hand
    /// the vector to the backend (kNN-only, or fused with RRF).
    #[instrument(skip(self))]
    pub async fn search(
        &self,
        query: &str,
        mode: SearchMode,
        _filters: SearchFilters,
        limit: Option<usize>,
    ) -> RagResult<Vec<ScoredResult>> {
        let limit = limit.unwrap_or(self.default_limit);
        let (keyword, query_vector) = match mode {
            SearchMode::Keyword => (Some(query), None),
            SearchMode::Semantic => (None, Some(self.embed_query(query).await?)),
            SearchMode::Hybrid => (Some(query), Some(self.embed_query(query).await?)),
        };
        let hits = self
            .backend
            .search(mode.into(), keyword, query_vector.as_deref(), limit)
            .await?;
        Ok(hits.into_iter().map(to_scored_result).collect())
    }

    /// Backs the `keyword_search` MCP tool: BM25-only, no embedding cost.
    pub async fn keyword_search(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> RagResult<Vec<ScoredResult>> {
        let limit = limit.unwrap_or(self.default_limit);
        let hits = self
            .backend
            .search(RetrievalMode::Keyword, Some(query), None, limit)
            .await?;
        Ok(hits.into_iter().map(to_scored_result).collect())
    }

    /// Backs the `vector_search` MCP tool: exactly one embed call plus one
    /// kNN-only request.
    pub async fn vector_search(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> RagResult<Vec<ScoredResult>> {
        let limit = limit.unwrap_or(self.default_limit);
        let embedding = self.embed_query(query).await?;
        let hits = self
            .backend
            .search(RetrievalMode::Semantic, None, Some(&embedding), limit)
            .await?;
        Ok(hits.into_iter().map(to_scored_result).collect())
    }

    /// Backs the `fetch_by_id` MCP tool.
    pub async fn fetch_by_id(&self, id: &str) -> RagResult<Document> {
        self.content_store.fetch(id).await
    }

    async fn embed_query(&self, query: &str) -> RagResult<Vec<f32>> {
        debug!(query, "embedding query for semantic/hybrid stage");
        self.embedder.embed(query).await
    }
}

/// Maps a backend hit onto the stable response shape. The engine's own score
/// passes through untouched (BM25 and RRF ranking is Elasticsearch's job);
/// `matched_via` is request-level provenance and `matched_ann` reflects
/// whether the request carried a kNN clause.
fn to_scored_result(hit: RankedHit) -> ScoredResult {
    ScoredResult {
        id: hit.id,
        source: hit.source,
        score: hit.score,
        snippet: hit.snippet,
        matched_via: vec![hit.which_strategy],
        matched_ann: hit.matched_ann,
    }
}
