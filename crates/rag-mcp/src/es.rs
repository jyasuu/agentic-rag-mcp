//! Minimal Elasticsearch client: a thin `reqwest` wrapper rather than a full
//! ES SDK. Everything this crate needs from ES today is a reachability check
//! at startup and the two search request shapes built by `EsRetrievalBackend`
//! (see `es_prefilter.rs`):
//!   - keyword  — BM25 `match` on the ik-analyzed `content`, query-only;
//!   - semantic — kNN on the indexed `embedding` `dense_vector`, knn-only.
//!
//! Hybrid mode issues both requests and fuses their ranks client-side with
//! reciprocal rank fusion (`es_prefilter::rrf_fuse`). ES's native `rank: { rrf }`
//! API is a paid-license feature, so the client-side fuse keeps hybrid
//! retrieval working on the free (basic) license the `es-ik` image ships with.
//!
//! The index schema is owned by the seed script / CDC mirror (created here via
//! `ensure_index`): a text `content` field analyzed by `ik_max_word`
//! (SPEC.md user story 11: Chinese queries get proper word segmentation rather
//! than naive CJK tokenization), a keyword `source` field, and an indexed
//! `embedding` `dense_vector(1024)` cosine field for ANN/hybrid retrieval.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::json;

use crate::embedder::EMBEDDING_DIM;

/// The `ik_max_word` analyzer/tokenizer name. The analysis-ik plugin
/// registers it as both an analyzer and a tokenizer, so it can be used
/// directly as the `content` field analyzer in mappings and in match-query
/// search-time analysis.
pub const IK_ANALYZER: &str = "ik_max_word";

/// The indexed `dense_vector` field holding each document's BGE-M3 embedding.
pub const EMBEDDING_FIELD: &str = "embedding";

/// One parsed search hit. Kept ES-shape-agnostic (no `_`-prefixed fields) so
/// the retrieval backend and any other caller consume a plain type.
#[derive(Debug, Clone, Deserialize)]
pub struct EsSearchHit {
    pub id: String,
    pub score: Option<f32>,
    /// The `source` metadata field (document provenance, e.g. a wiki path).
    pub source: String,
    /// Full content, used to build a truncated snippet when there is no
    /// query-aware highlight (ANN-only hits).
    pub content: Option<String>,
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
    content: Option<String>,
}

/// Pure, unit-testable parser for a `_search` response body. Handles all three
/// request shapes — RRF scores arrive in the same `_score` field as BM25/kNN.
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
            content: h.source.content,
            highlight: h.highlight.map_or_else(Vec::new, |f| {
                f.into_iter().flat_map(|(_, frags)| frags).collect()
            }),
        })
        .collect())
}

/// `k` and `num_candidates` for the kNN clause. ES requires
/// `num_candidates >= k`; both are derived from `limit` and kept small (the
/// demo corpus is tiny, so a 10x candidate pool is plenty).
fn knn_params(limit: usize) -> (usize, usize) {
    let k = limit.max(1);
    let num_candidates = k.saturating_mul(10).max(10);
    (k, num_candidates)
}

fn highlight_block() -> serde_json::Value {
    json!({
        "fields": { "content": {
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
            "fragment_size": 150
        } },
        "number_of_fragments": 1
    })
}

/// Pure request builder: BM25-only keyword search.
pub(crate) fn build_keyword_request(
    query: &str,
    limit: usize,
    analyzer: &str,
) -> serde_json::Value {
    json!({
        "size": limit,
        "query": { "match": { "content": { "query": query, "analyzer": analyzer } } },
        "highlight": highlight_block(),
    })
}

/// Pure request builder: kNN-only semantic search. No `query`, no `rank` — a
/// single embed call plus this request is the whole semantic path.
pub(crate) fn build_semantic_request(query_vector: &[f32], limit: usize) -> serde_json::Value {
    let (k, num_candidates) = knn_params(limit);
    json!({
        "size": limit,
        "knn": {
            "field": EMBEDDING_FIELD,
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates
        }
    })
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

    /// `_search` against `index`: BM25 `match` on `content` with `analyzer`
    /// (the caller's choice of ik segmentation granularity). Returns hits with
    /// their query-aware highlights so the backend can surface matched terms
    /// in context.
    pub async fn search_keyword(
        &self,
        index: &str,
        query: &str,
        limit: usize,
        analyzer: &str,
    ) -> anyhow::Result<Vec<EsSearchHit>> {
        let url = format!("{}/{}/_search", self.base_url.trim_end_matches('/'), index);
        self.post_search(&url, &build_keyword_request(query, limit, analyzer))
            .await
    }

    /// `_search` against `index`: kNN-only on `embedding`. The semantic path
    /// is exactly one embed call plus this request.
    pub async fn search_semantic(
        &self,
        index: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<EsSearchHit>> {
        let url = format!("{}/{}/_search", self.base_url.trim_end_matches('/'), index);
        self.post_search(&url, &build_semantic_request(query_vector, limit))
            .await
    }

    async fn post_search(&self, url: &str, body: &serde_json::Value) -> anyhow::Result<Vec<EsSearchHit>> {
        let resp = self
            .http
            .post(url)
            .json(body)
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
    /// `analyzer` (default `ik_max_word`), a keyword `source` field, and an
    /// indexed `embedding` `dense_vector(1024)` cosine field, plus the custom
    /// analyzer definition backing `ik_max_word`. Returns `Ok` whether the
    /// index was created or already existed -- ES signals the latter with a
    /// `resource_already_exists_exception` on the create call.
    ///
    /// Mapping changes require index recreation; the example lifecycle is
    /// create-or-bump (delete + recreate) driven by the seed script.
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
                "content": { "type": "text", "analyzer": analyzer },
                EMBEDDING_FIELD: {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": true,
                    "similarity": "cosine"
                }
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

    /// Indexes (inserts or replaces) one document, optionally carrying its
    /// BGE-M3 embedding so the index is self-sufficient for kNN/hybrid search.
    #[allow(dead_code)]
    pub async fn index_document(
        &self,
        index: &str,
        id: &str,
        source: &str,
        content: &str,
        embedding: Option<&[f32]>,
    ) -> anyhow::Result<()> {
        let url = format!(
            "{}/{}/_doc/{}",
            self.base_url.trim_end_matches('/'),
            index,
            id
        );
        let mut body = json!({ "source": source, "content": content });
        if let Some(embedding) = embedding {
            body[EMBEDDING_FIELD] = json!(embedding);
        }
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
    fn parse_search_response_extracts_hits_highlights_and_content() {
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
        assert_eq!(hits[0].content.as_deref(), Some("系统发生连接失败错误码"));
        assert_eq!(
            hits[0].highlight,
            vec!["系统发生<em>连接</em><em>失败</em>错误码"]
        );
        // No highlight block -> empty fragments, not an error.
        assert!(hits[1].highlight.is_empty());
        assert_eq!(hits[1].score, Some(0.5));
        assert_eq!(hits[1].content, None);
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

    #[test]
    fn keyword_request_is_query_only_with_ik_analyzer_and_highlight() {
        let req = build_keyword_request("连接失败", 5, IK_ANALYZER);
        assert_eq!(req["size"], 5);
        assert_eq!(
            req["query"]["match"]["content"]["query"],
            "连接失败",
            "keyword request must match content via BM25"
        );
        assert_eq!(req["query"]["match"]["content"]["analyzer"], IK_ANALYZER);
        assert!(req.get("knn").is_none(), "keyword request must not carry a kNN clause");
        assert!(req.get("rank").is_none(), "keyword request must not carry RRF");
        assert!(req["highlight"]["fields"]["content"]["pre_tags"][0] == "<em>");
    }

    #[test]
    fn semantic_request_is_knn_only() {
        let vector: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let req = build_semantic_request(&vector, 3);
        assert_eq!(req["size"], 3);
        assert!(req.get("query").is_none(), "semantic request must not carry a BM25 clause");
        assert!(req.get("rank").is_none(), "semantic request must not carry RRF");
        assert_eq!(req["knn"]["field"], EMBEDDING_FIELD);
        assert_eq!(req["knn"]["query_vector"], serde_json::json!([0.0, 1.0, 2.0, 3.0]));
        assert_eq!(req["knn"]["k"], 3);
        assert_eq!(req["knn"]["num_candidates"], 30, "num_candidates must be derived from limit and >= k");
    }

    #[test]
    fn semantic_request_keeps_num_candidates_at_least_k_for_small_limits() {
        let req = build_semantic_request(&[0.1, 0.2], 1);
        let k = req["knn"]["k"].as_u64().unwrap();
        let num_candidates = req["knn"]["num_candidates"].as_u64().unwrap();
        assert!(num_candidates >= k, "num_candidates {num_candidates} must be >= k {k}");
    }
}
