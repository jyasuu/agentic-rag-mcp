use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context;

/// Server configuration, sourced entirely from environment variables. Kept
/// as a plain struct (rather than threading `std::env::var` calls through
/// `main`) so startup failures produce one clear error per missing/invalid
/// value instead of a partial panic mid-setup.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub auth_token: String,
    pub database_url: String,
    pub elasticsearch_url: String,
    pub connect_timeout: Duration,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let bind_addr = std::env::var("RAG_MCP_BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .parse()
            .context("RAG_MCP_BIND_ADDR must be a valid socket address (e.g. 127.0.0.1:8080)")?;

        let auth_token = std::env::var("RAG_MCP_AUTH_TOKEN")
            .context("RAG_MCP_AUTH_TOKEN must be set (bearer token for MCP endpoint auth)")?;

        let database_url = std::env::var("RAG_MCP_DATABASE_URL").context(
            "RAG_MCP_DATABASE_URL must be set (Postgres connection string, \
             e.g. postgres://user:pass@host:5432/dbname)",
        )?;

        let elasticsearch_url = std::env::var("RAG_MCP_ELASTICSEARCH_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:9200".into());

        let connect_timeout_secs: u64 = std::env::var("RAG_MCP_CONNECT_TIMEOUT_SECS")
            .ok()
            .map(|s| s.parse())
            .transpose()
            .context("RAG_MCP_CONNECT_TIMEOUT_SECS must be a valid integer")?
            .unwrap_or(5);

        Ok(Self {
            bind_addr,
            auth_token,
            database_url,
            elasticsearch_url,
            connect_timeout: Duration::from_secs(connect_timeout_secs),
        })
    }
}
