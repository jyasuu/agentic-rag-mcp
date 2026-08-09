use async_trait::async_trait;

use crate::types::{Document, RagResult, RankedHit, RetrievalMode};

/// The single retrieval backend — Elasticsearch in production, wrapped by a
/// Postgres tsvector keyword fallback. Replaces the `PreFilterStrategy` list +
/// `AnnClient` split: one engine owns ranking (BM25 / kNN / RRF), so the
/// funnel has no cross-engine score merge to calibrate.
///
/// Decision-rich signature: the mode plus the optional keyword and query
/// vector tell the backend exactly which request shape to build.
#[async_trait]
pub trait RetrievalBackend: Send + Sync {
    /// Keyword: keyword = Some(query), query_vector = None → BM25-only.
    /// Semantic: keyword = None, query_vector = Some(embedding) → kNN-only.
    /// Hybrid: keyword = Some(query), query_vector = Some(embedding) →
    /// fused BM25 + kNN request with reciprocal rank fusion.
    async fn search(
        &self,
        mode: RetrievalMode,
        keyword: Option<&str>,
        query_vector: Option<&[f32]>,
        limit: usize,
    ) -> RagResult<Vec<RankedHit>>;
}

/// Local embedding model (BGE-M3 via `ort` in production). Kept as a trait
/// so the funnel — and its tests — never depend on ONNX Runtime directly.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> RagResult<Vec<f32>>;
}

/// Fetches full document/chunk content by id, for `fetch_by_id`.
#[async_trait]
pub trait ContentStore: Send + Sync {
    async fn fetch(&self, id: &str) -> RagResult<Document>;
}
