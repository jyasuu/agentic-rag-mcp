use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use rag_core::{FusionWeights, HybridFusion, HybridFusionConfig, RrfConfig, ScoreNormalization};

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
    /// Reciprocal Rank Fusion parameters sent with every hybrid request
    /// (`RAG_MCP_RRF_WINDOW_SIZE` / `RAG_MCP_RRF_RANK_CONSTANT`). Optional:
    /// defaults match Elasticsearch's own RRF defaults.
    pub rrf: RrfConfig,
    /// Hybrid fusion strategy: how keyword and kNN results are combined in
    /// hybrid mode. Selected at startup (`RAG_MCP_HYBRID_FUSION`); the
    /// default `client-rrf` preserves current free-license behavior.
    pub fusion: HybridFusionConfig,
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

        let rrf_window_size: usize = parse_usize_env("RAG_MCP_RRF_WINDOW_SIZE", 100)?;
        let rrf_rank_constant: usize = parse_usize_env("RAG_MCP_RRF_RANK_CONSTANT", 60)?;

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
            rrf: RrfConfig {
                window_size: rrf_window_size,
                rank_constant: rrf_rank_constant,
            },
            fusion: parse_fusion_env(
                std::env::var("RAG_MCP_HYBRID_FUSION").ok().as_deref(),
                std::env::var("RAG_MCP_HYBRID_NORMALIZATION").ok().as_deref(),
                std::env::var("RAG_MCP_HYBRID_KEYWORD_WEIGHT").ok().as_deref(),
                std::env::var("RAG_MCP_HYBRID_VECTOR_WEIGHT").ok().as_deref(),
            )?,
        })
    }
}

/// Pure parser for the hybrid fusion env vars, so the config is testable
/// without mutating the process env. Returns the default config when every
/// var is unset; errors on any invalid value (matching the fail-fast startup
/// contract of the rest of `Config::from_env`).
fn parse_fusion_env(
    method: Option<&str>,
    normalization: Option<&str>,
    keyword_weight: Option<&str>,
    vector_weight: Option<&str>,
) -> anyhow::Result<HybridFusionConfig> {
    let method = match method {
        Some(s) => s
            .parse::<HybridFusion>()
            .map_err(|e| anyhow::anyhow!("RAG_MCP_HYBRID_FUSION: {e}"))?,
        None => HybridFusion::default(),
    };
    let normalization = match normalization {
        Some(s) => s
            .parse::<ScoreNormalization>()
            .map_err(|e| anyhow::anyhow!("RAG_MCP_HYBRID_NORMALIZATION: {e}"))?,
        None => ScoreNormalization::default(),
    };

    let parse_weight = |s: &str, name: &str| -> anyhow::Result<f32> {
        s.parse::<f32>()
            .with_context(|| format!("{name} must be a number between 0 and 1"))
    };
    let kw = keyword_weight.map(|s| parse_weight(s, "RAG_MCP_HYBRID_KEYWORD_WEIGHT")).transpose()?;
    let vec = vector_weight.map(|s| parse_weight(s, "RAG_MCP_HYBRID_VECTOR_WEIGHT")).transpose()?;

    let weights = match (kw, vec) {
        (Some(k), Some(v)) => {
            if (k + v - 1.0).abs() > 1e-6 {
                anyhow::bail!(
                    "RAG_MCP_HYBRID_KEYWORD_WEIGHT + RAG_MCP_HYBRID_VECTOR_WEIGHT must sum to 1, \
                     got {k} + {v}"
                );
            }
            FusionWeights { keyword: k, vector: v }
        }
        (Some(k), None) => FusionWeights { keyword: k, vector: 1.0 - k },
        (None, Some(v)) => FusionWeights { keyword: 1.0 - v, vector: v },
        (None, None) => FusionWeights::default(),
    };

    Ok(HybridFusionConfig {
        method,
        normalization,
        weights,
    })
}

/// Reads an optional `usize` env var, returning `default` when unset and
/// surfacing a clear parse error when set to a non-integer.
fn parse_usize_env(name: &str, default: usize) -> anyhow::Result<usize> {
    Ok(std::env::var(name)
        .ok()
        .map(|s| {
            s.parse::<usize>()
                .with_context(|| format!("{name} must be a valid positive integer"))
        })
        .transpose()?
        .unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::{HybridFusion, ScoreNormalization};

    fn parse(
        method: Option<&str>,
        norm: Option<&str>,
        kw: Option<&str>,
        vec: Option<&str>,
    ) -> anyhow::Result<HybridFusionConfig> {
        parse_fusion_env(method, norm, kw, vec)
    }

    #[test]
    fn fusion_defaults_to_client_rrf_equal_weights_min_max() {
        let cfg = parse(None, None, None, None).unwrap();
        assert_eq!(cfg.method, HybridFusion::ClientRrf);
        assert_eq!(cfg.normalization, ScoreNormalization::MinMax);
        assert_eq!(cfg.weights.keyword, 0.5);
        assert_eq!(cfg.weights.vector, 0.5);
    }

    #[test]
    fn fusion_method_and_normalization_parse() {
        let cfg = parse(Some("server-rrf"), Some("l2"), None, None).unwrap();
        assert_eq!(cfg.method, HybridFusion::ServerRrf);
        assert_eq!(cfg.normalization, ScoreNormalization::L2);
    }

    #[test]
    fn single_weight_fills_the_other_to_sum_one() {
        let cfg = parse(Some("normalized-mean"), None, Some("0.3"), None).unwrap();
        assert!((cfg.weights.keyword - 0.3).abs() < 1e-6);
        assert!((cfg.weights.vector - 0.7).abs() < 1e-6);

        let cfg = parse(Some("normalized-mean"), None, None, Some("0.8")).unwrap();
        assert!((cfg.weights.keyword - 0.2).abs() < 1e-6);
        assert!((cfg.weights.vector - 0.8).abs() < 1e-6);
    }

    #[test]
    fn explicit_pair_must_sum_to_one() {
        assert!(parse(Some("normalized-mean"), None, Some("0.5"), Some("0.5")).is_ok());
        assert!(
            parse(Some("normalized-mean"), None, Some("0.3"), Some("0.8")).is_err(),
            "weights summing away from 1 must be rejected"
        );
    }

    #[test]
    fn invalid_fusion_values_fail_startup() {
        assert!(parse(Some("rank-rrf"), None, None, None).is_err());
        assert!(parse(Some("normalized-mean"), Some("zscore"), None, None).is_err());
        assert!(parse(Some("normalized-mean"), None, Some("abc"), None).is_err());
    }
}
