mod auth;
mod config;
mod embedder;
mod es;
mod es_prefilter;
mod fallback;
#[cfg(test)]
mod integration;
mod server;
mod state;
mod store;
#[cfg(test)]
mod testutil;
mod tsvector;
mod wiring;

use axum::middleware;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

use crate::auth::{require_bearer_token, BearerToken};
use crate::config::Config;
use crate::server::RagMcpServer;
use crate::state::AppState;
use crate::wiring::build_funnel;

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

    // The real backends: Elasticsearch retrieval (BM25 / kNN / RRF) with the
    // Postgres tsvector keyword fallback, the BGE-M3 embedder, and the
    // Postgres content store — see `wiring.rs` for the funnel construction
    // (shared with integration tests).
    let funnel = build_funnel(&config, app_state)?;

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
