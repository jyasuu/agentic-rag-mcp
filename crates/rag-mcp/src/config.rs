use std::net::SocketAddr;
use std::path::PathBuf;
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
    /// Elasticsearch index the CDC sync writes and the ES pre-filter reads.
    pub es_index: String,
    /// Directory holding the BGE-M3 ONNX graph + tokenizer.json. Optional:
    /// keyword-only deployments can run without it; `vector_search` /
    /// semantic hybrid queries then fail with a clear error at call time.
    pub embedding_model_dir: Option<PathBuf>,
    /// Remote Ollama base URL (e.g. `https://…trycloudflare.com`). When set,
    /// embeddings are served by Ollama's `/api/embed` instead of the local
    /// ONNX session (takes priority over `embedding_model_dir`).
    pub ollama_url: Option<String>,
    /// Model name sent to Ollama's `/api/embed`. Must have
    /// `embedding_length = 1024` to match the `vector(1024)` column.
    pub ollama_model: String,
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

        let es_index = std::env::var("RAG_MCP_ES_INDEX").unwrap_or_else(|_| "documents".into());

        let embedding_model_dir = std::env::var_os("RAG_MCP_EMBEDDING_MODEL_DIR").map(Into::into);

        let ollama_url = std::env::var("RAG_MCP_OLLAMA_URL").ok();
        let ollama_model = std::env::var("RAG_MCP_OLLAMA_MODEL").unwrap_or_else(|_| "bge-m3".into());

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
            es_index,
            embedding_model_dir,
            ollama_url,
            ollama_model,
            connect_timeout: Duration::from_secs(connect_timeout_secs),
        })
    }
}
