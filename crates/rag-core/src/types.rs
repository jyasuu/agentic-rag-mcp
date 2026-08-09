use serde::{Deserialize, Serialize};

/// How the caller wants `search` to run the funnel.
///
/// `Hybrid` is the default: Elasticsearch fuses the keyword (BM25) and
/// semantic (kNN) clauses natively with reciprocal rank fusion (RRF), so
/// relevance is decided by one engine's ranking rather than a hand-rolled
/// score merge. `Keyword` and `Semantic` let an agent bypass the fuse when it
/// already knows the query shape. Maps 1:1 onto `RetrievalMode`, which the
/// funnel hands to the retrieval backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Keyword,
    Semantic,
    #[default]
    Hybrid,
}

/// The retrieval dispatch mode passed to `RetrievalBackend::search`. Kept
/// separate from the caller-facing `SearchMode` so the backend contract
/// (which request shape to build) doesn't inherit the MCP-facing serde type's
/// `Default`/serialization semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    Keyword,
    Semantic,
    Hybrid,
}

impl From<SearchMode> for RetrievalMode {
    fn from(mode: SearchMode) -> Self {
        match mode {
            SearchMode::Keyword => RetrievalMode::Keyword,
            SearchMode::Semantic => RetrievalMode::Semantic,
            SearchMode::Hybrid => RetrievalMode::Hybrid,
        }
    }
}

/// Optional narrowing filters a caller can pass to `search` / `keyword_search`
/// / `vector_search`. Kept intentionally small for v1 — extend as real query
/// patterns emerge rather than speculatively.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub source: Option<String>,
    pub language: Option<String>,
}

/// A single hit returned by a `RetrievalBackend`, unifying keyword (BM25),
/// semantic (kNN), and hybrid (RRF) results so the funnel can produce
/// `ScoredResult`s directly. Replaces the `PreFilterHit` / `AnnHit` split.
#[derive(Debug, Clone)]
pub struct RankedHit {
    pub id: String,
    pub source: String,
    /// Engine-native score: BM25 for keyword, cosine similarity for semantic,
    /// and the RRF score for hybrid. Ranges are engine-specific and NOT
    /// normalized — Elasticsearch owns the ranking.
    pub score: f32,
    /// Rendered snippet: `<em>`-highlighted for BM25 matches, truncated for
    /// ANN-only hits (there is no query clause to highlight).
    pub snippet: String,
    /// Which strategy produced the hit. Request-level: for an ES hybrid
    /// request every hit reports Elasticsearch, since RRF responses don't
    /// expose which clause matched.
    pub which_strategy: PreFilterStrategyKind,
    /// Whether the request carried a kNN clause (request-level, not per-hit).
    pub matched_ann: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreFilterStrategyKind {
    Elasticsearch,
    Tsvector,
}

/// Final, ranked result returned to the MCP caller. Shape is stable so
/// existing tool-use patterns keep working.
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
