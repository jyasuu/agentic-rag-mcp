use serde::{Deserialize, Serialize};

/// How the caller wants `search` to run the funnel.
///
/// `Hybrid` is the default: pre-filter always runs; ANN runs only if the
/// pre-filter stage doesn't return confident results (see
/// `ShortCircuitConfig`). `Keyword` and `Semantic` let an agent bypass the
/// funnel's own judgment when it already knows the query shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Keyword,
    Semantic,
    #[default]
    Hybrid,
}

/// Optional narrowing filters a caller can pass to `search` / `keyword_search`
/// / `vector_search`. Kept intentionally small for v1 — extend as real query
/// patterns emerge rather than speculatively.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub source: Option<String>,
    pub language: Option<String>,
}

/// A hit returned by a pre-filter strategy (tsvector / pg_trgm /
/// Elasticsearch), before ANN or scoring has run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFilterHit {
    pub id: String,
    pub source: String,
    /// Strategy-native relevance score (e.g. ES BM25 score, ts_rank,
    /// trigram similarity). Not comparable across strategies without
    /// normalization — `ScoringConfig` is responsible for that.
    pub raw_score: f32,
    /// Pre-rendered snippet, e.g. from ES highlight. `None` for strategies
    /// that don't produce one (pg_trgm), in which case the funnel falls
    /// back to truncation.
    pub highlighted_snippet: Option<String>,
    pub which_strategy: PreFilterStrategyKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreFilterStrategyKind {
    Elasticsearch,
    Tsvector,
    Trigram,
}

/// A hit returned by the ANN (pgvector) stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnHit {
    pub id: String,
    pub source: String,
    /// Cosine/L2 similarity, strategy-defined range.
    pub similarity: f32,
    /// Raw content used to produce a fallback truncated snippet, since ANN
    /// hits have no query-aware highlight (see opportunity list).
    pub content_preview: String,
}

/// Final, scored, snippet-level result returned to the MCP caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    pub id: String,
    pub source: String,
    pub score: f32,
    pub snippet: String,
    pub matched_via: Vec<PreFilterStrategyKind>,
    pub matched_ann: bool,
}

/// Full document/chunk content, returned by `fetch_by_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub source: String,
    pub content: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("pre-filter backend error: {0}")]
    PreFilter(String),
    #[error("ANN backend error: {0}")]
    Ann(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("content store error: {0}")]
    ContentStore(String),
    #[error("document not found: {0}")]
    NotFound(String),
}

pub type RagResult<T> = Result<T, RagError>;
