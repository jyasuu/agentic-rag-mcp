//! Minimal Elasticsearch client: a thin `reqwest` wrapper rather than a full
//! ES SDK. All this crate needs from ES today is a reachability check at
//! startup, the `ik_analyzer` pre-filter search (`EsPreFilter` in
//! `es_prefilter.rs`), and the index mutations the CDC consumer applies
//! (`cdc.rs`), all against the same `http`/`base_url`.
//!
//! The index schema is owned by the CDC sync (index created here with a text
//! `content` field analyzed by `ik_max_word`), matching SPEC.md user story 11:
//! Chinese queries get proper word segmentation rather than naive CJK
//! tokenization.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::json;

/// The `ik_max_word` analyzer/tokenizer name. The analysis-ik plugin
/// registers it as both an analyzer and a tokenizer, so it can be used
/// directly as the `content` field analyzer in mappings and in match-query
/// search-time analysis.
pub const IK_ANALYZER: &str = "ik_max_word";

/// One parsed search hit. Kept ES-shape-agnostic (no `_`-prefixed fields) so
/// the pre-filter strategy and any other caller consume a plain type.
#[derive(Debug, Clone, Deserialize)]
pub struct EsSearchHit {
    pub id: String,
    pub score: Option<f32>,
    /// The `source` metadata field (document provenance, e.g. a wiki path).
    pub source: String,
    /// Query-aware highlight fragments, in ES highlight format
    /// (`<em>…</em>`), used to build snippets per SPEC.md user story 7.
    pub highlight: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    hits: SearchHits,
}

#[derive(Debug, Deserialize)]
struct SearchHits {
    hits: Vec<RawSearchHit>,
}

#[derive(Debug, Deserialize)]
struct RawSearchHit {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "_score")]
    score: Option<f32>,
    #[serde(rename = "_source")]
    source: Source,
    highlight: Option<HashMap<String, Vec<String>>>,
}

#[derive(Debug, Deserialize)]
struct Source {
    source: Option<String>,
}

/// Pure, unit-testable parser for a `_search` response body.
pub(crate) fn parse_search_response(body: &str) -> anyhow::Result<Vec<EsSearchHit>> {
    let resp: SearchResponse = serde_json::from_str(body)
        .with_context(|| "failed to parse Elasticsearch _search response")?;
    Ok(resp
        .hits
        .hits
        .into_iter()
        .map(|h| EsSearchHit {
            id: h.id,
            score: h.score,
            source: h.source.source.unwrap_or_default(),
            highlight: h.highlight.map_or_else(Vec::new, |f| {
                f.into_iter().flat_map(|(_, frags)| frags).collect()
            }),
        })
        .collect())
}

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

    /// `_search` against `index`, matching `content` with `analyzer` (the
    /// caller's choice of ik segmentation granularity). Returns hits with
    /// their query-aware highlights so the strategy can surface matched
    /// terms in context.
    pub async fn search(
        &self,
        index: &str,
        query: &str,
        limit: usize,
        analyzer: &str,
    ) -> anyhow::Result<Vec<EsSearchHit>> {
        let url = format!("{}/{}/_search", self.base_url.trim_end_matches('/'), index);
        let body = json!({
            "size": limit,
            "query": { "match": { "content": { "query": query, "analyzer": analyzer } } },
            "highlight": {
                "fields": { "content": {
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fragment_size": 150
                } },
                "number_of_fragments": 1
            }
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to reach Elasticsearch at {url}"))?;
        let text = resp
            .error_for_status()
            .with_context(|| format!("Elasticsearch search at {url} returned an error"))?
            .text()
            .await
            .context("failed to read Elasticsearch search response body")?;
        parse_search_response(&text)
    }

    /// Idempotently creates `index` with a text `content` field analyzed by
    /// `analyzer` (default `ik_max_word`) and a keyword `source` field, plus
    /// the custom analyzer definition backing `ik_max_word`. Returns `Ok`
    /// whether the index was created or already existed -- ES signals the
    /// latter with a `resource_already_exists_exception` on the create call.
    #[allow(dead_code)]
    pub async fn ensure_index(&self, index: &str, analyzer: &str) -> anyhow::Result<()> {
        let url = format!("{}/{}", self.base_url.trim_end_matches('/'), index);
        let body = json!({
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "ik": { "type": "custom", "tokenizer": "ik_max_word" }
                    }
                }
            },
            "mappings": { "properties": {
                "source": { "type": "keyword" },
                "content": { "type": "text", "analyzer": analyzer }
            } }
        });
        let resp = self
            .http
            .put(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to reach Elasticsearch at {url}"))?;

        match resp.status().as_u16() {
            200 => Ok(()),
            // The index already exists with (hopefully) the same mapping.
            400 => {
                let text = resp
                    .text()
                    .await
                    .context("failed to read Elasticsearch create-index error body")?;
                if text.contains("resource_already_exists_exception") {
                    Ok(())
                } else {
                    bail!("Elasticsearch create index at {url} failed: {text}");
                }
            }
            other => bail!("Elasticsearch create index at {url} returned status {other}"),
        }
    }

    /// Indexes (inserts or replaces) one document.
    #[allow(dead_code)]
    pub async fn index_document(
        &self,
        index: &str,
        id: &str,
        source: &str,
        content: &str,
    ) -> anyhow::Result<()> {
        let url = format!(
            "{}/{}/_doc/{}",
            self.base_url.trim_end_matches('/'),
            index,
            id
        );
        let body = json!({ "source": source, "content": content });
        let resp = self
            .http
            .put(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to reach Elasticsearch at {url}"))?;
        if !resp.status().is_success() {
            let text = resp
                .text()
                .await
                .context("failed to read Elasticsearch index-document error body")?;
            bail!("Elasticsearch index document at {url} failed: {text}");
        }
        Ok(())
    }

    /// Deletes one document by id. Deleting a non-existent document returns
    /// `200` with `result: "not_found"`, which is treated as success -- the
    /// end state (no such document) is what matters to the CDC consumer.
    #[allow(dead_code)]
    pub async fn delete_document(&self, index: &str, id: &str) -> anyhow::Result<()> {
        let url = format!(
            "{}/{}/_doc/{}",
            self.base_url.trim_end_matches('/'),
            index,
            id
        );
        let resp = self
            .http
            .delete(&url)
            .send()
            .await
            .with_context(|| format!("failed to reach Elasticsearch at {url}"))?;
        if !resp.status().is_success() {
            let text = resp
                .text()
                .await
                .context("failed to read Elasticsearch delete-document error body")?;
            bail!("Elasticsearch delete document at {url} failed: {text}");
        }
        Ok(())
    }

    /// Unused until the CDC ticket makes real index calls through this client.
    #[allow(dead_code)]
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_search_response_extracts_hits_and_highlights() {
        let body = r#"{
            "took": 3,
            "timed_out": false,
            "hits": {
                "total": { "value": 2, "relation": "eq" },
                "max_score": 1.2,
                "hits": [
                    {
                        "_index": "documents",
                        "_id": "doc-1",
                        "_score": 1.2,
                        "_source": { "source": "wiki/zh.md", "content": "系统发生连接失败错误码" },
                        "highlight": { "content": ["系统发生<em>连接</em><em>失败</em>错误码"] }
                    },
                    {
                        "_index": "documents",
                        "_id": "doc-2",
                        "_score": 0.5,
                        "_source": { "source": "wiki/errors.md" }
                    }
                ]
            }
        }"#;

        let hits = parse_search_response(body).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "doc-1");
        assert_eq!(hits[0].score, Some(1.2));
        assert_eq!(hits[0].source, "wiki/zh.md");
        assert_eq!(
            hits[0].highlight,
            vec!["系统发生<em>连接</em><em>失败</em>错误码"]
        );
        // No highlight block -> empty fragments, not an error.
        assert!(hits[1].highlight.is_empty());
        assert_eq!(hits[1].score, Some(0.5));
    }

    #[test]
    fn parse_search_response_handles_null_score_and_missing_source_field() {
        let body = r#"{
            "hits": {
                "total": { "value": 1, "relation": "eq" },
                "hits": [
                    {
                        "_id": "doc-x",
                        "_score": null,
                        "_source": { "source": null }
                    }
                ]
            }
        }"#;

        let hits = parse_search_response(body).unwrap();
        assert_eq!(hits[0].id, "doc-x");
        assert_eq!(hits[0].score, None);
        // `source: null` in _source deserializes to "" via Option coercion.
        assert_eq!(hits[0].source, "");
    }

    #[test]
    fn parse_search_response_errors_on_malformed_body() {
        assert!(parse_search_response("not json").is_err());
        assert!(parse_search_response(r#"{"hits": {"hits": []}}"#).is_ok());
    }
}
