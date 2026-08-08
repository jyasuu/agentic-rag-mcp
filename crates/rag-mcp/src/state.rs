use anyhow::Context;
use sqlx::postgres::PgPoolOptions;

use crate::config::Config;
use crate::es::EsClient;

/// Shared backend connections, held for the lifetime of the process and
/// cloned into anything that needs them (future `PreFilterStrategy` /
/// `AnnClient` / `ContentStore` implementations). Constructed once at
/// startup, with health checks so a misconfigured Postgres/Elasticsearch
/// URL fails the process immediately with a clear error instead of
/// surfacing as an opaque error on the first tool call.
// Fields are unused until the PreFilterStrategy/AnnClient/ContentStore
// tickets wire real backend implementations through this connection state
// (see main.rs) -- for now, connecting and health-checking is the entire
// scope of this scaffold.
#[allow(dead_code)]
#[derive(Clone)]
pub struct AppState {
    pub pg_pool: sqlx::PgPool,
    pub es: EsClient,
}

impl AppState {
    pub async fn connect(config: &Config) -> anyhow::Result<Self> {
        let pg_pool = PgPoolOptions::new()
            .acquire_timeout(config.connect_timeout)
            .connect(&config.database_url)
            .await
            .context(
                "failed to connect to Postgres -- check RAG_MCP_DATABASE_URL and that the \
                 database is reachable",
            )?;
        sqlx::query("SELECT 1")
            .execute(&pg_pool)
            .await
            .context("Postgres connection established but health check query failed")?;
        tracing::info!("Postgres connection established");

        let es = EsClient::new(&config.elasticsearch_url, config.connect_timeout)?;
        es.health_check().await.context(
            "failed to reach Elasticsearch -- check RAG_MCP_ELASTICSEARCH_URL and that the \
             cluster is reachable",
        )?;
        tracing::info!(url = %es.base_url(), "Elasticsearch connection established");

        Ok(Self { pg_pool, es })
    }
}
