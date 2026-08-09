//! The real-backend funnel wiring. Kept in its own module (rather than inline
//! in `main`) so the end-to-end integration tests in `integration.rs` build
//! the funnel through exactly the same code path the server runs.

use std::sync::Arc;

use rag_core::{Embedder, RetrievalBackend, RetrievalFunnel};

use crate::config::Config;
use crate::embedder::{BgeM3Embedder, OllamaEmbedder, UnavailableEmbedder};
use crate::es_prefilter::EsRetrievalBackend;
use crate::fallback::FallbackRetrievalBackend;
use crate::state::AppState;
use crate::store::PostgresContentStore;
use crate::tsvector::TsvectorRetrievalBackend;

/// Builds the production funnel: one Elasticsearch retrieval backend (BM25 /
/// kNN / RRF) with the Postgres tsvector keyword fallback for an unavailable
/// or unsynced cluster, the BGE-M3 embedder, and the Postgres content store.
///
/// Embedder selection: `RAG_MCP_OLLAMA_URL` (remote Ollama) takes priority
/// over the local ONNX `RAG_MCP_EMBEDDING_MODEL_DIR`; when neither is set, a
/// clear-error stub is used so keyword-only deployments still start.
pub fn build_funnel(config: &Config, app_state: AppState) -> anyhow::Result<Arc<RetrievalFunnel>> {
    let es_backend = EsRetrievalBackend::new(
        app_state.es.clone(),
        config.es_index.clone(),
        config.rrf,
    );

    let backend: Box<dyn RetrievalBackend> = Box::new(FallbackRetrievalBackend::new(
        Box::new(es_backend),
        Box::new(TsvectorRetrievalBackend::new(app_state.pg_pool.clone())),
    ));

    let embedder: Box<dyn Embedder> = match &config.ollama_url {
        Some(url) => {
            tracing::info!(
                ollama_url = %url, model = %config.ollama_model,
                "using remote Ollama for query embeddings"
            );
            Box::new(OllamaEmbedder::new(url, &config.ollama_model)?)
        }
        None => match &config.embedding_model_dir {
            Some(dir) => Box::new(BgeM3Embedder::load(dir)?),
            None => {
                tracing::warn!(
                    "no embedding model configured; vector_search and semantic hybrid queries \
                     will fail until RAG_MCP_OLLAMA_URL or RAG_MCP_EMBEDDING_MODEL_DIR is set"
                );
                Box::new(UnavailableEmbedder)
            }
        },
    };

    Ok(Arc::new(RetrievalFunnel::new(
        backend,
        embedder,
        Box::new(PostgresContentStore::new(app_state.pg_pool.clone())),
    )))
}
