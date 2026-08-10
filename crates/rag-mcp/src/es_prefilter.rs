//! Elasticsearch `RetrievalBackend` — the single retrieval engine behind the
//! funnel. Keyword mode issues a BM25-only request, semantic mode a kNN-only
//! request, and hybrid mode both, fused client-side with reciprocal rank
//! fusion (`rrf_fuse`) — rank-based, so there are no score-normalization
//! coefficients to calibrate, and it runs on ES's free license (the server-
//! side `rank: { rrf }` API needs a paid license). `matched_ann` stays
//! per-hit accurate: a hit is an ANN match exactly when it appeared in the
//! kNN list.
//!
//! Errors are classed so callers know the failure mode: keyword-mode failures
//! surface as `RagError::PreFilter` (the tsvector fallback can catch those),
//! while semantic/hybrid failures surface as `RagError::Ann` because kNN
//! cannot fall back to tsvector.
//!
//! The index is created by `ensure_index` with the `ik_max_word` analyzer on
//! `content` and an indexed `embedding` `dense_vector` (see `es.rs`); the
//! search-time analysis in the match query uses the same analyzer so
//! segmentation is consistent at index and query time.

use std::collections::HashMap;

use async_trait::async_trait;
use rag_core::{
    HybridFusion, HybridFusionConfig, PreFilterStrategyKind, RagError, RagResult, RankedHit,
    RetrievalBackend, RetrievalMode, RrfConfig, ScoreNormalization,
};

use crate::es::{EsClient, EsSearchHit, IK_ANALYZER};

/// Max chars for ANN-only snippet fallback (no query clause to highlight).
const SNIPPET_FALLBACK_LEN: usize = 200;

pub struct EsRetrievalBackend {
    client: EsClient,
    index: String,
    rrf: RrfConfig,
    fusion: HybridFusionConfig,
}

impl EsRetrievalBackend {
    pub fn new(
        client: EsClient,
        index: impl Into<String>,
        rrf: RrfConfig,
        fusion: HybridFusionConfig,
    ) -> Self {
        Self {
            client,
            index: index.into(),
            rrf,
            fusion,
        }
    }
}

/// Maps raw ES hits into `RankedHit`s: `_score` as the score (BM25, kNN
/// cosine, or RRF — engine-owned either way), the query-aware highlight as the
/// snippet when present, and a truncated content fallback for ANN-only hits.
/// Pure so it can be unit-tested without a live cluster.
fn map_hits(hits: Vec<EsSearchHit>, matched_ann: bool) -> Vec<RankedHit> {
    hits.into_iter()
        .map(|h| RankedHit {
            id: h.id,
            source: h.source,
            score: h.score.unwrap_or(0.0),
            snippet: if !h.highlight.is_empty() {
                h.highlight.into_iter().next().unwrap_or_default()
            } else {
                h.content
                    .as_deref()
                    .map(|c| truncate(c, SNIPPET_FALLBACK_LEN))
                    .unwrap_or_default()
            },
            which_strategy: PreFilterStrategyKind::Elasticsearch,
            matched_ann,
        })
        .collect()
}

fn truncate(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", &s[..idx]),
        None => s.to_string(),
    }
}

/// Client-side reciprocal rank fusion of the BM25 and kNN result lists. Each
/// hit contributes `1 / (rank_constant + rank + 1)` per list it appears in
/// (capped at `window_size` entries per list), mirroring the RRF formula
/// without needing ES's paid-license `rank: { rrf }` API. A hit that appeared
/// in the kNN list is an ANN match (`matched_ann: true`), so provenance stays
/// per-hit accurate. Pure so the fusion math is unit-testable without a
/// cluster.
fn rrf_fuse(
    keyword: Vec<EsSearchHit>,
    semantic: Vec<EsSearchHit>,
    limit: usize,
    rrf: RrfConfig,
) -> Vec<RankedHit> {
    struct Fused {
        source: String,
        score: f32,
        snippet: String,
        content: Option<String>,
        matched_ann: bool,
    }

    let mut by_id: HashMap<String, Fused> = HashMap::new();
    for (hits, matched_ann) in [(keyword, false), (semantic, true)] {
        for (rank, h) in hits.into_iter().take(rrf.window_size).enumerate() {
            let entry = by_id.entry(h.id.clone()).or_insert_with(|| Fused {
                source: h.source.clone(),
                score: 0.0,
                snippet: String::new(),
                content: h.content.clone(),
                matched_ann,
            });
            entry.score += 1.0 / (rrf.rank_constant as f32 + rank as f32 + 1.0);
            entry.matched_ann = entry.matched_ann || matched_ann;
            if entry.snippet.is_empty() {
                entry.snippet = h.highlight.into_iter().next().unwrap_or_default();
            }
        }
    }

    let mut results: Vec<RankedHit> = by_id
        .into_iter()
        .map(|(id, mut f)| {
            if f.snippet.is_empty() {
                f.snippet = f
                    .content
                    .as_deref()
                    .map(|c| truncate(c, SNIPPET_FALLBACK_LEN))
                    .unwrap_or_default();
            }
            RankedHit {
                id,
                source: f.source,
                score: f.score,
                snippet: f.snippet,
                which_strategy: PreFilterStrategyKind::Elasticsearch,
                matched_ann: f.matched_ann,
            }
        })
        .collect();

    // Score desc, then id asc so ties (e.g. two docs each in one list) are
    // deterministic across runs.
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

/// Client-side score-based fusion of the BM25 and kNN result lists. Each
/// list's raw scores are normalized independently (min-max over the returned
/// list, or L2), then each hit's fused score is the weighted mean of its
/// normalized scores: `weights.keyword * norm_keyword + weights.vector *
/// norm_vector`. Unlike `rrf_fuse`, score *magnitude* matters (a much
/// stronger BM25 match can outrank a middling one) and the per-list weights
/// let the operator emphasize the signal they trust more. `matched_ann` stays
/// per-hit accurate and snippet behavior matches `rrf_fuse` (keyword
/// highlight wins, truncated fallback otherwise). Pure so the fusion math is
/// unit-testable without a cluster.
fn score_fuse(
    keyword: Vec<EsSearchHit>,
    semantic: Vec<EsSearchHit>,
    limit: usize,
    cfg: HybridFusionConfig,
) -> Vec<RankedHit> {
    struct Fused {
        source: String,
        score: f32,
        snippet: String,
        content: Option<String>,
        matched_ann: bool,
    }

    let mut by_id: HashMap<String, Fused> = HashMap::new();

    for (hits, matched_ann, weight) in [
        (&keyword, false, cfg.weights.keyword),
        (&semantic, true, cfg.weights.vector),
    ] {
        let norm = normalize_scores(hits, cfg.normalization);
        for (h, n) in hits.iter().zip(norm) {
            let entry = by_id.entry(h.id.clone()).or_insert_with(|| Fused {
                source: h.source.clone(),
                score: 0.0,
                snippet: String::new(),
                content: h.content.clone(),
                matched_ann,
            });
            entry.score += weight * n;
            entry.matched_ann = entry.matched_ann || matched_ann;
            if entry.snippet.is_empty() {
                entry.snippet = h.highlight.first().cloned().unwrap_or_default();
            }
        }
    }

    let mut results: Vec<RankedHit> = by_id
        .into_iter()
        .map(|(id, mut f)| {
            if f.snippet.is_empty() {
                f.snippet = f
                    .content
                    .as_deref()
                    .map(|c| truncate(c, SNIPPET_FALLBACK_LEN))
                    .unwrap_or_default();
            }
            RankedHit {
                id,
                source: f.source,
                score: f.score,
                snippet: f.snippet,
                which_strategy: PreFilterStrategyKind::Elasticsearch,
                matched_ann: f.matched_ann,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

/// Normalizes a list of raw scores per the chosen method. Empty input
/// produces an empty output; a zero-norm L2 list (all-zero scores) and a
/// min-max list with no range both avoid divide-by-zero by falling back to a
/// neutral value rather than NaN.
fn normalize_scores(hits: &[EsSearchHit], norm: ScoreNormalization) -> Vec<f32> {
    let scores: Vec<f32> = hits.iter().map(|h| h.score.unwrap_or(0.0)).collect();
    match norm {
        ScoreNormalization::MinMax => {
            let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range <= f32::EPSILON {
                vec![1.0; scores.len()]
            } else {
                scores.iter().map(|s| (s - min) / range).collect()
            }
        }
        ScoreNormalization::L2 => {
            let l2: f32 = scores.iter().map(|s| s * s).sum::<f32>().sqrt();
            if l2 <= f32::EPSILON {
                vec![0.0; scores.len()]
            } else {
                scores.iter().map(|s| s / l2).collect()
            }
        }
    }
}

#[async_trait]
impl RetrievalBackend for EsRetrievalBackend {
    async fn search(
        &self,
        mode: RetrievalMode,
        keyword: Option<&str>,
        query_vector: Option<&[f32]>,
        limit: usize,
    ) -> RagResult<Vec<RankedHit>> {
        match mode {
            RetrievalMode::Keyword => {
                let query = keyword.ok_or_else(|| {
                    RagError::PreFilter("keyword mode requires a keyword query".into())
                })?;
                let hits = self
                    .client
                    .search_keyword(&self.index, query, limit, IK_ANALYZER)
                    .await
                    .map_err(|e| RagError::PreFilter(format!("Elasticsearch search failed: {e}")))?;
                Ok(map_hits(hits, false))
            }
            RetrievalMode::Semantic => {
                let vector = query_vector.ok_or_else(|| {
                    RagError::Ann("semantic mode requires a query vector".into())
                })?;
                let hits = self
                    .client
                    .search_semantic(&self.index, vector, limit)
                    .await
                    .map_err(|e| RagError::Ann(format!("Elasticsearch kNN search failed: {e}")))?;
                Ok(map_hits(hits, true))
            }
            RetrievalMode::Hybrid => {
                let query = keyword.ok_or_else(|| {
                    RagError::Ann("hybrid mode requires a keyword query".into())
                })?;
                let vector = query_vector.ok_or_else(|| {
                    RagError::Ann("hybrid mode requires a query vector".into())
                })?;
                match self.fusion.method {
                    HybridFusion::ClientRrf => {
                        let keyword_hits = self
                            .client
                            .search_keyword(&self.index, query, limit, IK_ANALYZER)
                            .await
                            .map_err(|e| RagError::Ann(format!("Elasticsearch keyword search failed: {e}")))?;
                        let semantic_hits = self
                            .client
                            .search_semantic(&self.index, vector, limit)
                            .await
                            .map_err(|e| RagError::Ann(format!("Elasticsearch kNN search failed: {e}")))?;
                        Ok(rrf_fuse(keyword_hits, semantic_hits, limit, self.rrf))
                    }
                    HybridFusion::NormalizedMean => {
                        let keyword_hits = self
                            .client
                            .search_keyword(&self.index, query, limit, IK_ANALYZER)
                            .await
                            .map_err(|e| RagError::Ann(format!("Elasticsearch keyword search failed: {e}")))?;
                        let semantic_hits = self
                            .client
                            .search_semantic(&self.index, vector, limit)
                            .await
                            .map_err(|e| RagError::Ann(format!("Elasticsearch kNN search failed: {e}")))?;
                        Ok(score_fuse(keyword_hits, semantic_hits, limit, self.fusion))
                    }
                    HybridFusion::ServerRrf => {
                        let hits = self
                            .client
                            .search_hybrid_server(
                                &self.index,
                                query,
                                vector,
                                limit,
                                IK_ANALYZER,
                                self.rrf,
                            )
                            .await
                            .map_err(|e| RagError::Ann(format!("Elasticsearch hybrid search failed: {e}")))?;
                        // ES RRF responses don't expose per-hit clause matches,
                        // so `matched_ann` is request-level: the request carried
                        // a kNN clause.
                        Ok(map_hits(hits, true))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use rag_core::FusionWeights;

    use crate::es::IK_ANALYZER;

    fn es_hit(id: &str, content: &str, highlights: Vec<&str>) -> EsSearchHit {
        EsSearchHit {
            id: id.into(),
            score: Some(1.0),
            source: "wiki/zh.md".into(),
            content: Some(content.into()),
            highlight: highlights.into_iter().map(String::from).collect(),
        }
    }

    /// `es_hit` with an explicit raw score, for normalization tests.
    fn es_hit_scored(id: &str, score: f32, content: &str) -> EsSearchHit {
        EsSearchHit {
            id: id.into(),
            score: Some(score),
            source: "wiki/zh.md".into(),
            content: Some(content.into()),
            highlight: vec![],
        }
    }

    #[test]
    fn score_fuse_minmax_normalizes_and_combines_with_weights() {
        // keyword scores: a=100, b=10  -> minmax: a=1.0, b=0.0
        // vector  scores: a=50,  c=40  -> minmax: a=1.0, c=0.0
        // weights 0.5/0.5:
        //   a = 0.5*1.0 + 0.5*1.0 = 1.0
        //   b = 0.5*0.0 = 0.0
        //   c = 0.5*0.0 = 0.0
        // tie b/c broken by id asc -> b, c.
        let keyword = vec![
            es_hit_scored("a", 100.0, "alpha"),
            es_hit_scored("b", 10.0, "beta"),
        ];
        let semantic = vec![
            es_hit_scored("a", 50.0, "alpha"),
            es_hit_scored("c", 40.0, "gamma"),
        ];
        let cfg = HybridFusionConfig {
            method: HybridFusion::NormalizedMean,
            normalization: ScoreNormalization::MinMax,
            weights: FusionWeights {
                keyword: 0.5,
                vector: 0.5,
            },
        };

        let fused = score_fuse(keyword, semantic, 10, cfg);

        assert_eq!(
            fused.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!((fused[0].score - 1.0).abs() < 1e-6, "in both lists at max -> 1.0");
        assert!((fused[1].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn score_fuse_minmax_uses_keyword_highlight_and_flags_ann() {
        let keyword = vec![
            es_hit_scored("a", 100.0, "alpha"),
        ];
        let semantic = vec![
            es_hit_scored("a", 50.0, "alpha"),
            es_hit_scored("b", 40.0, "beta"),
        ];
        let cfg = HybridFusionConfig {
            method: HybridFusion::NormalizedMean,
            normalization: ScoreNormalization::MinMax,
            weights: FusionWeights { keyword: 0.7, vector: 0.3 },
        };
        // a has no keyword highlight, so it falls back to a truncated snippet;
        // b is in the kNN list -> matched_ann.
        let fused = score_fuse(keyword, semantic, 10, cfg);
        let by_id = |id: &str| fused.iter().find(|h| h.id == id).unwrap();
        assert!(by_id("b").matched_ann, "kNN-list presence is an ANN match");
        assert!(by_id("b").snippet.starts_with("beta"), "ANN-only snippet truncation fallback");
    }

    #[test]
    fn score_fuse_l2_normalizes_by_list_norm() {
        // keyword: a=3, b=4 -> l2 = 5 -> a=0.6, b=0.8
        // vector:  a=6, c=8 -> l2 = 10 -> a=0.6, c=0.8
        // weights 0.5/0.5:
        //   a = 0.5*0.6 + 0.5*0.6 = 0.6
        //   b = 0.5*0.8 = 0.4
        //   c = 0.5*0.8 = 0.4 -> tie broken by id -> b, c
        let keyword = vec![
            es_hit_scored("a", 3.0, "alpha"),
            es_hit_scored("b", 4.0, "beta"),
        ];
        let semantic = vec![
            es_hit_scored("a", 6.0, "alpha"),
            es_hit_scored("c", 8.0, "gamma"),
        ];
        let cfg = HybridFusionConfig {
            method: HybridFusion::NormalizedMean,
            normalization: ScoreNormalization::L2,
            weights: FusionWeights { keyword: 0.5, vector: 0.5 },
        };

        let fused = score_fuse(keyword, semantic, 10, cfg);
        assert_eq!(
            fused.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!((fused[0].score - 0.6).abs() < 1e-6);
        assert!((fused[1].score - 0.4).abs() < 1e-6);
    }

    #[test]
    fn score_fuse_handles_empty_and_zero_norm_lists() {
        let cfg = HybridFusionConfig {
            method: HybridFusion::NormalizedMean,
            normalization: ScoreNormalization::MinMax,
            weights: FusionWeights { keyword: 0.5, vector: 0.5 },
        };
        // Empty semantic list: keyword-only hits survive with weight-scaled
        // normalized scores.
        let keyword = vec![es_hit_scored("a", 42.0, "alpha")];
        let fused = score_fuse(keyword, vec![], 10, cfg);
        assert_eq!(fused.len(), 1);
        assert!((fused[0].score - 0.5).abs() < 1e-6, "single-hit list normalizes to 1.0, halved by weight");

        // Zero-norm list: constant score (max == min) is not a divide-by-zero;
        // both hits get the same normalized score and rank by id.
        let flat_keyword = vec![
            es_hit_scored("a", 1.0, "alpha"),
            es_hit_scored("b", 1.0, "beta"),
        ];
        let fused = score_fuse(flat_keyword, vec![], 10, cfg);
        assert_eq!(fused.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
    }

    #[test]
    fn score_fuse_respects_limit_and_keeps_positive_scores() {
        let cfg = HybridFusionConfig {
            method: HybridFusion::NormalizedMean,
            normalization: ScoreNormalization::MinMax,
            weights: FusionWeights { keyword: 0.5, vector: 0.5 },
        };
        let keyword: Vec<EsSearchHit> = (0..5).map(|i| es_hit_scored(&format!("k{i}"), i as f32 + 1.0, "k")).collect();
        let fused = score_fuse(keyword, vec![], 2, cfg);
        assert_eq!(fused.len(), 2, "limit truncates after fusion");
        assert!(fused.iter().all(|h| h.score >= 0.0));
    }

    #[test]
    fn rrf_fuse_combines_ranks_from_both_lists_and_breaks_ties_by_id() {
        // keyword: a@rank0, b@rank1; semantic: a@rank0, c@rank1.
        // a = 1/61 + 1/61; b and c = 1/62 each; tie broken by id asc -> b, c.
        let keyword = vec![
            es_hit("a", "alpha", vec!["<em>连接</em>失败"]),
            es_hit("b", "beta", vec![]),
        ];
        let semantic = vec![
            es_hit("a", "alpha", vec![]),
            es_hit("c", "gamma", vec![]),
        ];
        let rrf = RrfConfig::default();

        let fused = rrf_fuse(keyword, semantic, 10, rrf);

        assert_eq!(
            fused.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        let expected_a = 2.0 / 61.0;
        assert!(
            (fused[0].score - expected_a).abs() < 1e-6,
            "a in both lists scores 2/(k+1), got {}",
            fused[0].score
        );
        assert!(
            (fused[1].score - 1.0 / 62.0).abs() < 1e-6,
            "single-list docs score 1/(k+rank+1), got {}",
            fused[1].score
        );
    }

    #[test]
    fn rrf_fuse_flags_ann_matches_from_knn_presence() {
        let keyword = vec![es_hit("both", "x", vec![]), es_hit("kw", "y", vec![])];
        let semantic = vec![es_hit("both", "x", vec![]), es_hit("ann", "z", vec![])];

        let fused = rrf_fuse(keyword, semantic, 10, RrfConfig::default());
        let by_id = |id: &str| fused.iter().find(|h| h.id == id).unwrap();

        assert!(by_id("both").matched_ann, "present in the kNN list is an ANN match");
        assert!(!by_id("kw").matched_ann, "keyword-only hit is not an ANN match");
        assert!(by_id("ann").matched_ann, "kNN-only hit is an ANN match");
    }

    #[test]
    fn rrf_fuse_uses_keyword_highlight_else_truncated_snippet() {
        let long = "系统发生连接失败错误码需要重试。".repeat(50);
        let keyword = vec![es_hit("a", &long, vec!["<em>连接</em>失败错误码"])];
        let semantic = vec![es_hit("b", &long, vec![])];

        let fused = rrf_fuse(keyword, semantic, 10, RrfConfig::default());
        let by_id = |id: &str| fused.iter().find(|h| h.id == id).unwrap();

        assert_eq!(by_id("a").snippet, "<em>连接</em>失败错误码", "keyword highlight wins");
        assert!(
            by_id("b").snippet.starts_with("系统") && by_id("b").snippet.ends_with('…'),
            "ANN-only hits fall back to a truncated content snippet"
        );
    }

    #[test]
    fn rrf_fuse_respects_window_size_and_limit() {
        // window_size=1: only rank0 of each list contributes, so b (keyword
        // rank1) and c (semantic rank1) both drop out entirely.
        let keyword = vec![es_hit("a", "alpha", vec![]), es_hit("b", "beta", vec![])];
        let semantic = vec![es_hit("a", "alpha", vec![]), es_hit("c", "gamma", vec![])];
        let rrf = RrfConfig { window_size: 1, rank_constant: 60 };

        let fused = rrf_fuse(keyword.clone(), semantic.clone(), 10, rrf);
        assert_eq!(
            fused.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a"],
            "docs outside the per-list window are excluded"
        );

        let limited = rrf_fuse(keyword, semantic, 1, RrfConfig::default());
        assert_eq!(
            limited.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a"],
            "limit truncates after fusion"
        );
    }

    #[test]
    fn map_hits_uses_highlight_when_present_and_truncates_content_otherwise() {
        let long_content = "系统发生连接失败错误码需要重试。".repeat(50);
        let hits = vec![
            EsSearchHit {
                id: "doc-1".into(),
                score: Some(2.5),
                source: "wiki/zh.md".into(),
                content: Some(long_content.clone()),
                highlight: vec![
                    "系统发生<em>连接</em>".into(),
                    "<em>失败</em>错误码".into(),
                ],
            },
            EsSearchHit {
                id: "doc-2".into(),
                score: None,
                source: "wiki/errors.md".into(),
                content: Some(long_content.clone()),
                highlight: vec![],
            },
            EsSearchHit {
                id: "doc-3".into(),
                score: Some(0.4),
                source: "wiki/empty.md".into(),
                content: None,
                highlight: vec![],
            },
        ];

        let mapped = map_hits(hits, true);
        assert_eq!(mapped[0].id, "doc-1");
        assert_eq!(mapped[0].score, 2.5);
        assert_eq!(mapped[0].snippet, "系统发生<em>连接</em>");
        assert!(mapped[0].matched_ann);
        assert_eq!(mapped[0].which_strategy, PreFilterStrategyKind::Elasticsearch);

        // No highlight -> content truncation fallback.
        assert!(mapped[1].snippet.starts_with("系统发生连接失败"));
        assert!(
            mapped[1].snippet.chars().count() <= SNIPPET_FALLBACK_LEN + 1,
            "fallback snippet should be capped at the fallback length"
        );
        assert_eq!(mapped[1].score, 0.0, "missing score defaults to 0");

        // No content at all -> empty snippet, not an error.
        assert_eq!(mapped[2].snippet, "");
    }

    #[test]
    fn map_hits_keyword_mode_reports_no_ann() {
        let hits = vec![EsSearchHit {
            id: "doc-1".into(),
            score: Some(1.0),
            source: "wiki/api.md".into(),
            content: None,
            highlight: vec![],
        }];
        let mapped = map_hits(hits, false);
        assert!(!mapped[0].matched_ann, "keyword hits never matched via ANN");
    }

    // Real-Elasticsearch integration tests, mirroring the Postgres test
    // pattern: run against `RAG_MCP_ELASTICSEARCH_URL`, skipped (not failed)
    // when unset. Each test uses its own uniquely-named index so concurrent
    // test runs never interfere, and polls for searchability because ES is
    // near-real-time, not write-through.
    const TEST_INDEX_PREFIX: &str = "es-";

    async fn test_client() -> Option<EsClient> {
        let url = std::env::var("RAG_MCP_ELASTICSEARCH_URL").ok()?;
        // Generous timeout: the whole test suite runs against the same
        // single-node cluster concurrently, so a request can take several
        // seconds under load.
        Some(
            EsClient::new(url, Duration::from_secs(30))
                .expect("RAG_MCP_ELASTICSEARCH_URL set but client failed to build"),
        )
    }

    async fn unique_index(client: &EsClient) -> String {
        let index = format!("{}{}", TEST_INDEX_PREFIX, crate::testutil::unique_token("idx"));
        client
            .ensure_index(&index, IK_ANALYZER)
            .await
            .expect("ensure_index should succeed");
        index
    }

    async fn cleanup_index(client: &EsClient, index: &str) {
        let url = format!("{}/{}", client.base_url().trim_end_matches('/'), index);
        let resp = client
            .http()
            .delete(&url)
            .send()
            .await
            .expect("index delete should be reachable");
        assert!(resp.status().is_success(), "index cleanup should succeed");
    }

    /// A deterministic pseudo-random vector seeded by `seed`, so two fixtures
    /// built with different seeds are far apart in cosine space and a query
    /// vector lands closest to its own fixture (used in place of real BGE-M3
    /// embeddings so these tests don't need an embedder).
    fn vec(seed: u32, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed as f64) + (i as f64) * 1.7).sin() as f32)
            .collect()
    }

    fn backend(client: EsClient, index: String) -> EsRetrievalBackend {
        EsRetrievalBackend::new(client, index, RrfConfig::default(), HybridFusionConfig::default())
    }

    fn backend_with_fusion(
        client: EsClient,
        index: String,
        fusion: HybridFusionConfig,
    ) -> EsRetrievalBackend {
        EsRetrievalBackend::new(client, index, RrfConfig::default(), fusion)
    }

    /// ES is near-real-time: poll until the expected id is visible (or give
    /// up after a few seconds).
    async fn search_until_visible(
        backend: &EsRetrievalBackend,
        query: &str,
        expected_id: &str,
    ) -> Vec<RankedHit> {
        for _ in 0..20 {
            let hits = backend
                .search(RetrievalMode::Keyword, Some(query), None, 10)
                .await
                .expect("search should succeed");
            if hits.iter().any(|h| h.id == expected_id) {
                return hits;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        backend
            .search(RetrievalMode::Keyword, Some(query), None, 10)
            .await
            .expect("search should succeed")
    }

    #[tokio::test]
    async fn keyword_search_returns_segmented_match_with_highlight() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        client
            .index_document(&index, "zh-1", "wiki/zh.md", "系统发生连接失败错误码需要重试", None)
            .await
            .expect("indexing should succeed");
        client
            .index_document(&index, "zh-2", "wiki/zh.md", "今天天气晴朗适合外出", None)
            .await
            .expect("indexing should succeed");

        let backend = backend(client.clone(), index.clone());
        let hits = search_until_visible(&backend, "连接失败", "zh-1").await;

        let first = hits
            .iter()
            .find(|h| h.id == "zh-1")
            .expect("zh-1 should be in results");
        assert_eq!(first.which_strategy, PreFilterStrategyKind::Elasticsearch);
        assert!(!first.matched_ann, "keyword request carries no kNN clause");
        assert!(first.score > 0.0, "BM25 score should be positive for a real match");
        assert!(
            first.snippet.contains("<em>连接</em>") && first.snippet.contains("<em>失败</em>"),
            "highlight should surface the segmented match terms in context, got: {}",
            first.snippet
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn exact_code_token_returns_precise_match() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        let token = crate::testutil::unique_token("es-code");
        client
            .index_document(&index, "code-1", "wiki/errors.md", &format!("ERROR_{token}: connection refused"), None)
            .await
            .expect("indexing should succeed");

        let backend = backend(client.clone(), index.clone());
        let hits = search_until_visible(&backend, &token, "code-1").await;

        assert!(
            hits.iter().any(|h| h.id == "code-1"),
            "exact code token should match its document"
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn no_match_returns_empty_not_error() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        client
            .index_document(&index, "n-1", "wiki/zh.md", "系统发生连接失败", None)
            .await
            .expect("indexing should succeed");

        let backend = backend(client.clone(), index.clone());
        // A token never indexed anywhere.
        let token = crate::testutil::unique_token("es-absent");
        let hits = backend
            .search(RetrievalMode::Keyword, Some(&token), None, 10)
            .await
            .expect("search should succeed");

        assert!(hits.is_empty());

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn semantic_search_ranks_knn_hits_by_cosine_similarity() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        let prefix = crate::testutil::unique_token("sem");
        let id_a = format!("{prefix}-a");
        let id_b = format!("{prefix}-b");
        let va = vec(1, 1024);
        let vb = vec(2, 1024);
        client
            .index_document(&index, &id_a, "wiki/api.md", &format!("alpha {prefix}"), Some(&va))
            .await
            .expect("indexing should succeed");
        client
            .index_document(&index, &id_b, "wiki/errors.md", &format!("beta {prefix}"), Some(&vb))
            .await
            .expect("indexing should succeed");

        let backend = backend(client.clone(), index.clone());
        // Query close to `va` (same seed), so `id_a` must rank first. Poll
        // until the fixtures are searchable.
        let mut hits = Vec::new();
        for _ in 0..20 {
            hits = backend
                .search(RetrievalMode::Semantic, None, Some(&va), 10)
                .await
                .expect("kNN search should succeed");
            if hits.iter().any(|h| h.id.starts_with(&prefix)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        let ids: Vec<&str> = hits
            .iter()
            .filter(|h| h.id.starts_with(&prefix))
            .map(|h| h.id.as_str())
            .collect();
        assert_eq!(ids, vec![id_a.as_str(), id_b.as_str()], "closest fixture should rank first");
        let top = hits
            .iter()
            .find(|h| h.id == id_a)
            .expect("id_a should be present");
        assert!(top.matched_ann, "kNN request means hits came via ANN");
        assert!(top.score > 0.9, "near-identical vector should score near 1.0");
        assert!(
            top.snippet.starts_with("alpha"),
            "ANN-only hits fall back to a truncated content snippet"
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn hybrid_search_fuses_keyword_and_knn_clauses() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        let va = vec(1, 1024);
        let vb = vec(2, 1024);
        client
            .index_document(&index, "hy-1", "wiki/zh.md", "系统发生连接失败错误码需要重试", Some(&va))
            .await
            .expect("indexing should succeed");
        client
            .index_document(&index, "hy-2", "wiki/zh.md", "今天天气晴朗适合外出", Some(&vb))
            .await
            .expect("indexing should succeed");

        let backend = backend(client.clone(), index.clone());
        // hy-1 matches the keyword clause AND is kNN-near the query vector;
        // hy-2 matches only the kNN clause. Client-side RRF must return both,
        // rank hy-1 first (it appears in both ranked lists), and keep scores
        // positive.
        let mut hits = Vec::new();
        for _ in 0..20 {
            hits = backend
                .search(RetrievalMode::Hybrid, Some("连接失败"), Some(&va), 10)
                .await
                .expect("hybrid search should succeed");
            if hits.len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        assert_eq!(hits.len(), 2, "both the keyword match and the kNN match should surface");
        assert_eq!(hits[0].id, "hy-1", "doc matched by both clauses ranks first under RRF");
        let hy1 = &hits[0];
        assert!(hy1.matched_ann, "hy-1 appeared in the kNN list, so it is an ANN match");
        assert!(
            hy1.snippet.contains("<em>连接</em>"),
            "the keyword-clause highlight survives client-side fusion"
        );
        let hy2 = hits.iter().find(|h| h.id == "hy-2").expect("kNN-only doc should be present");
        assert!(hy2.matched_ann, "kNN-only doc is an ANN match");
        assert!(hy2.score > 0.0, "RRF scores should be positive");
        assert!(
            hy2.snippet.starts_with("今天"),
            "ANN-only hits fall back to a truncated content snippet"
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn hybrid_search_normalized_mean_weights_keyword_over_vector() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        let va = vec(1, 1024);
        let vb = vec(2, 1024);
        // Both docs appear in both lists (each is both a strong keyword and a
        // near kNN match); the one with the stronger BM25 score must rank
        // first under the weighted normalized mean.
        client
            .index_document(&index, "nm-1", "wiki/zh.md", "系统发生连接失败错误码需要重试 连接失败 连接失败", Some(&va))
            .await
            .expect("indexing should succeed");
        client
            .index_document(&index, "nm-2", "wiki/zh.md", "系统发生连接失败错误码 连接失败", Some(&vb))
            .await
            .expect("indexing should succeed");

        let fusion = HybridFusionConfig {
            method: HybridFusion::NormalizedMean,
            normalization: ScoreNormalization::MinMax,
            weights: FusionWeights {
                keyword: 0.7,
                vector: 0.3,
            },
        };
        let backend = backend_with_fusion(client.clone(), index.clone(), fusion);

        let mut hits = Vec::new();
        for _ in 0..20 {
            hits = backend
                .search(RetrievalMode::Hybrid, Some("连接失败"), Some(&va), 10)
                .await
                .expect("hybrid search should succeed");
            if hits.len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        assert_eq!(hits.len(), 2, "both docs should surface under weighted-mean fusion");
        assert_eq!(hits[0].id, "nm-1", "stronger keyword match wins when keyword weight dominates");
        assert!(
            hits[0].snippet.contains("<em>连接</em>"),
            "keyword highlight survives normalized-mean fusion"
        );
        assert!(
            hits.iter().all(|h| h.score >= 0.0),
            "normalized-mean fused scores must be non-negative"
        );

        cleanup_index(&client, &index).await;
    }

    #[tokio::test]
    async fn hybrid_search_server_rrf_returns_results_or_clean_license_error() {
        let Some(client) = test_client().await else {
            eprintln!("skipping: RAG_MCP_ELASTICSEARCH_URL not set");
            return;
        };
        let index = unique_index(&client).await;
        let va = vec(1, 1024);
        client
            .index_document(&index, "sr-1", "wiki/zh.md", "系统发生连接失败错误码需要重试", Some(&va))
            .await
            .expect("indexing should succeed");

        let fusion = HybridFusionConfig {
            method: HybridFusion::ServerRrf,
            ..HybridFusionConfig::default()
        };
        let backend = backend_with_fusion(client.clone(), index.clone(), fusion);

        // The license gate is engine-side: a licensed cluster returns fused
        // results; a basic-license cluster rejects the request, and that must
        // surface as a clear `Ann` error naming the failure rather than
        // silently degrading to empty results.
        match backend.search(RetrievalMode::Hybrid, Some("连接失败"), Some(&va), 10).await {
            Ok(hits) => {
                assert!(
                    hits.iter().any(|h| h.id == "sr-1"),
                    "server-side RRF should surface the matching doc, got {hits:?}"
                );
            }
            Err(RagError::Ann(msg)) => {
                assert!(
                    msg.contains("hybrid search failed"),
                    "license rejection must surface as a clear Ann error, got {msg:?}"
                );
            }
            Err(other) => panic!("server-rrf must surface Ok or RagError::Ann, got {other:?}"),
        }

        cleanup_index(&client, &index).await;
    }
}
