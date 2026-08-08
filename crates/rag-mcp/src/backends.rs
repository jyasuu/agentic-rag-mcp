//! Placeholder backend implementations.
//!
//! Wiring real Postgres (tsvector/pg_trgm via `sqlx`), Elasticsearch
//! (`ik_analyzer`), pgvector ANN search, and a local BGE-M3 embedder (via
//! `ort`) is genuine next-step implementation work — each is its own
//! integration surface with connection config, error handling, and
//! (for embeddings) model loading. Stubbing them here keeps the MCP tool
//! wiring and funnel logic (the part validated by rag-core's tests)
//! separate from that work, per the seam in SPEC.md.
//!
//! Swap these for real implementations of `PreFilterStrategy`, `AnnClient`,
//! `Embedder`, and `ContentStore` from `rag-core::traits` without touching
//! `server.rs` or the funnel logic at all.

use async_trait::async_trait;
use rag_core::{AnnClient, AnnHit, ContentStore, Document, Embedder, PreFilterHit, PreFilterStrategy, RagError, RagResult};

/// Placeholder for pre-filter strategies not yet implemented (pg_trgm,
/// Elasticsearch/ik_analyzer) -- superseded for English/code content by
/// `TsvectorPreFilter` (see `tsvector.rs`), but kept here as the pattern
/// for the remaining strategy tickets and as a manual testing fixture.
#[allow(dead_code)]
pub struct NotImplementedPreFilter;
#[async_trait]
impl PreFilterStrategy for NotImplementedPreFilter {
    async fn search(&self, _query: &str, _limit: usize) -> RagResult<Vec<PreFilterHit>> {
        Err(RagError::PreFilter(
            "no pre-filter backend wired yet -- implement Elasticsearch/tsvector/pg_trgm in backends.rs".into(),
        ))
    }
}

pub struct NotImplementedAnn;
#[async_trait]
impl AnnClient for NotImplementedAnn {
    async fn search(&self, _embedding: &[f32], _limit: usize) -> RagResult<Vec<AnnHit>> {
        Err(RagError::Ann(
            "pgvector ANN backend not wired yet -- implement in backends.rs".into(),
        ))
    }
}

pub struct NotImplementedEmbedder;
#[async_trait]
impl Embedder for NotImplementedEmbedder {
    async fn embed(&self, _text: &str) -> RagResult<Vec<f32>> {
        Err(RagError::Embedding(
            "BGE-M3 embedder not wired yet -- implement via `ort` in backends.rs".into(),
        ))
    }
}

pub struct NotImplementedContentStore;
#[async_trait]
impl ContentStore for NotImplementedContentStore {
    async fn fetch(&self, _id: &str) -> RagResult<Document> {
        Err(RagError::ContentStore(
            "content store backend not wired yet -- implement Postgres fetch in backends.rs".into(),
        ))
    }
}
