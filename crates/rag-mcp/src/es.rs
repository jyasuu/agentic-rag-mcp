//! Minimal Elasticsearch client: a thin `reqwest` wrapper rather than a full
//! ES SDK. All this crate needs from ES today is a reachability check at
//! startup; the `ik_analyzer` pre-filter strategy ticket will extend this
//! with real search/index calls against the same `http`/`base_url`.

use std::time::Duration;

use anyhow::{Context, bail};

#[derive(Clone)]
pub struct EsClient {
    http: reqwest::Client,
    base_url: String,
}

impl EsClient {
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .context("failed to build Elasticsearch HTTP client")?;
        Ok(Self {
            http,
            base_url: base_url.into(),
        })
    }

    /// Hits `_cluster/health` to confirm Elasticsearch is reachable and
    /// responding. Used at startup so misconfiguration fails fast with a
    /// clear message instead of surfacing as an opaque error on the first
    /// real search call.
    pub async fn health_check(&self) -> anyhow::Result<()> {
        let url = format!("{}/_cluster/health", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to reach Elasticsearch at {url}"))?;

        if !resp.status().is_success() {
            bail!(
                "Elasticsearch health check at {url} returned status {}",
                resp.status()
            );
        }
        Ok(())
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Unused until the ik_analyzer `PreFilterStrategy` ticket makes real
    /// search/index calls through this client.
    #[allow(dead_code)]
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }
}
