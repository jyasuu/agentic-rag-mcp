use async_trait::async_trait;

use crate::types::{AnnHit, Document, PreFilterHit, RagResult};

/// One `FieldRule`-style pre-filter strategy (tsvector, pg_trgm, or
/// Elasticsearch). Each content type in the corpus maps to one of these;
/// `RetrievalFunnel` may query more than one and merge results.
#[async_trait]
pub trait PreFilterStrategy: Send + Sync {
    async fn search(&self, query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>>;
}

/// pgvector ANN search, given a pre-computed query embedding.
#[async_trait]
pub trait AnnClient: Send + Sync {
    async fn search(&self, embedding: &[f32], limit: usize) -> RagResult<Vec<AnnHit>>;
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
