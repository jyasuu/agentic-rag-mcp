//! The real-backend funnel wiring. Kept in its own module (rather than inline
//! in `main`) so the end-to-end integration tests in `integration.rs` build
//! the funnel through exactly the same code path the server runs.

use std::sync::Arc;

use rag_core::{Embedder, PreFilterStrategy, RetrievalFunnel};

use crate::ann::PgvectorAnnClient;
use crate::config::Config;
use crate::embedder::{BgeM3Embedder, UnavailableEmbedder};
use crate::es_prefilter::EsPreFilter;
use crate::fallback::FallbackPreFilter;
use crate::state::AppState;
use crate::store::PostgresContentStore;
use crate::trigram::TrigramPreFilter;
use crate::tsvector::TsvectorPreFilter;

/// Builds the production funnel: tsvector pre-filter (English/code) plus the
/// ES primary (Chinese/ik) with a pg_trgm fallback for an unavailable or
/// unsynced cluster, pgvector ANN, the BGE-M3 embedder (a clear-error stub
/// when `RAG_MCP_EMBEDDING_MODEL_DIR` is unset), and the Postgres content
/// store.
pub fn build_funnel(config: &Config, app_state: AppState) -> anyhow::Result<Arc<RetrievalFunnel>> {
    let prefilter: Vec<Box<dyn PreFilterStrategy>> = vec![
        Box::new(TsvectorPreFilter::new(app_state.pg_pool.clone())),
        Box::new(FallbackPreFilter::new(
            Box::new(EsPreFilter::new(app_state.es.clone(), config.es_index.clone())),
            Box::new(TrigramPreFilter::new(app_state.pg_pool.clone())),
        )),
    ];

    let embedder: Box<dyn Embedder> = match &config.embedding_model_dir {
        Some(dir) => Box::new(BgeM3Embedder::load(dir)?),
        None => {
            tracing::warn!(
                "RAG_MCP_EMBEDDING_MODEL_DIR not set; vector_search and semantic hybrid queries \
                 will fail until it points at the BGE-M3 model directory"
            );
            Box::new(UnavailableEmbedder)
        }
    };

    Ok(Arc::new(RetrievalFunnel::new(
        prefilter,
        Box::new(PgvectorAnnClient::new(app_state.pg_pool.clone())),
        embedder,
        Box::new(PostgresContentStore::new(app_state.pg_pool.clone())),
    )))
}
