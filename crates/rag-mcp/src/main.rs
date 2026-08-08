mod auth;
mod backends;
mod config;
mod es;
mod server;
mod state;
mod tsvector;

use std::sync::Arc;

use axum::middleware;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

use rag_core::RetrievalFunnel;

use crate::auth::{require_bearer_token, BearerToken};
use crate::backends::{NotImplementedAnn, NotImplementedContentStore, NotImplementedEmbedder};
use crate::config::Config;
use crate::server::RagMcpServer;
use crate::state::AppState;
use crate::tsvector::TsvectorPreFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::from_env()?;

    // Connects to Postgres and Elasticsearch and health-checks both, so a
    // misconfigured or unreachable backend fails the process immediately at
    // startup with a clear error rather than surfacing on the first tool
    // call.
    let app_state = AppState::connect(&config).await?;

    // TODO: replace the remaining `NotImplemented*` backends with real
    // pg_trgm/Elasticsearch (additional PreFilterStrategy entries),
    // pgvector (AnnClient), and BGE-M3 (Embedder) implementations, built
    // from `app_state` -- see backends.rs and tsvector.rs.
    let funnel = Arc::new(RetrievalFunnel::new(
        vec![Box::new(TsvectorPreFilter::new(app_state.pg_pool.clone()))],
        Box::new(NotImplementedAnn),
        Box::new(NotImplementedEmbedder),
        Box::new(NotImplementedContentStore),
    ));

    let rag_server = RagMcpServer::new(funnel.clone());
    let mcp_service = StreamableHttpService::new(
        move || Ok(rag_server.clone()),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let bearer = BearerToken(config.auth_token.clone());
    let router = axum::Router::new()
        .nest_service("/mcp", mcp_service)
        .layer(middleware::from_fn_with_state(bearer.clone(), require_bearer_token))
        .with_state(bearer);

    tracing::info!(bind_addr = %config.bind_addr, "starting agentic-rag MCP server");
    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.expect("failed to listen for ctrl-c");
        })
        .await?;

    Ok(())
}
