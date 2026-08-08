use std::collections::HashMap;

use tracing::{debug, instrument};

use crate::scoring::{ScoringConfig, ShortCircuitConfig};
use crate::traits::{AnnClient, ContentStore, Embedder, PreFilterStrategy};
use crate::types::{Document, PreFilterHit, RagResult, ScoredResult, SearchFilters, SearchMode};

const SNIPPET_FALLBACK_LEN: usize = 200;

/// Transport-independent retrieval service. This is the seam: MCP tool
/// handlers are thin wrappers around this struct's methods, so tests exercise
/// `RetrievalFunnel` directly (e.g. against a Testcontainers-backed Postgres
/// + Elasticsearch) without touching the `rmcp`/axum layer at all.
pub struct RetrievalFunnel {
    prefilter: Vec<Box<dyn PreFilterStrategy>>,
    ann: Box<dyn AnnClient>,
    embedder: Box<dyn Embedder>,
    content_store: Box<dyn ContentStore>,
    scoring: ScoringConfig,
    short_circuit: ShortCircuitConfig,
    default_limit: usize,
}

impl RetrievalFunnel {
    pub fn new(
        prefilter: Vec<Box<dyn PreFilterStrategy>>,
        ann: Box<dyn AnnClient>,
        embedder: Box<dyn Embedder>,
        content_store: Box<dyn ContentStore>,
    ) -> Self {
        Self {
            prefilter,
            ann,
            embedder,
            content_store,
            scoring: ScoringConfig::default(),
            short_circuit: ShortCircuitConfig::default(),
            default_limit: 10,
        }
    }

    pub fn with_scoring(mut self, scoring: ScoringConfig) -> Self {
        self.scoring = scoring;
        self
    }

    pub fn with_short_circuit(mut self, short_circuit: ShortCircuitConfig) -> Self {
        self.short_circuit = short_circuit;
        self
    }

    /// Primary entry point, backing the `search` MCP tool.
    #[instrument(skip(self))]
    pub async fn search(
        &self,
        query: &str,
        mode: SearchMode,
        _filters: SearchFilters,
        limit: Option<usize>,
    ) -> RagResult<Vec<ScoredResult>> {
        let limit = limit.unwrap_or(self.default_limit);

        let prefilter_hits = if matches!(mode, SearchMode::Keyword | SearchMode::Hybrid) {
            self.run_prefilter(query, limit).await?
        } else {
            Vec::new()
        };

        let should_run_ann = match mode {
            SearchMode::Semantic => true,
            SearchMode::Keyword => false,
            SearchMode::Hybrid => !self.is_confident(&prefilter_hits),
        };

        let ann_hits = if should_run_ann {
            debug!(query, "running ANN stage");
            let embedding = self.embedder.embed(query).await?;
            self.ann.search(&embedding, limit).await?
        } else {
            debug!(query, "skipping ANN stage (short-circuit or keyword-only mode)");
            Vec::new()
        };

        Ok(self.merge_and_score(prefilter_hits, ann_hits, limit))
    }

    /// Backs the `keyword_search` MCP tool: pre-filter stage only.
    pub async fn keyword_search(&self, query: &str, limit: Option<usize>) -> RagResult<Vec<ScoredResult>> {
        let limit = limit.unwrap_or(self.default_limit);
        let hits = self.run_prefilter(query, limit).await?;
        Ok(self.merge_and_score(hits, Vec::new(), limit))
    }

    /// Backs the `vector_search` MCP tool: ANN stage only.
    pub async fn vector_search(&self, query: &str, limit: Option<usize>) -> RagResult<Vec<ScoredResult>> {
        let limit = limit.unwrap_or(self.default_limit);
        let embedding = self.embedder.embed(query).await?;
        let hits = self.ann.search(&embedding, limit).await?;
        Ok(self.merge_and_score(Vec::new(), hits, limit))
    }

    /// Backs the `fetch_by_id` MCP tool.
    pub async fn fetch_by_id(&self, id: &str) -> RagResult<Document> {
        self.content_store.fetch(id).await
    }

    async fn run_prefilter(&self, query: &str, limit: usize) -> RagResult<Vec<PreFilterHit>> {
        // Query every configured strategy and merge. In production only one
        // strategy is typically wired per content type (ES for Chinese,
        // tsvector for English/code, pg_trgm as fallback), so this is
        // usually a single call, not a fan-out cost.
        let mut all_hits = Vec::new();
        for strategy in &self.prefilter {
            all_hits.extend(strategy.search(query, limit).await?);
        }
        Ok(all_hits)
    }

    /// Auto short-circuit heuristic (opportunity list: threshold tuning).
    fn is_confident(&self, hits: &[PreFilterHit]) -> bool {
        if hits.len() < self.short_circuit.min_hit_count {
            return false;
        }
        hits.iter()
            .map(|h| h.raw_score)
            .fold(f32::MIN, f32::max)
            >= self.short_circuit.min_top_score
    }

    fn merge_and_score(
        &self,
        prefilter_hits: Vec<PreFilterHit>,
        ann_hits: Vec<crate::types::AnnHit>,
        limit: usize,
    ) -> Vec<ScoredResult> {
        // Accumulate per-id signal components first, then combine once per
        // id at the end — avoids re-deriving partial scores across passes.
        struct Signals {
            source: String,
            exact: f32,
            ann: f32,
            snippet: String,
            matched_via: Vec<crate::types::PreFilterStrategyKind>,
            matched_ann: bool,
        }

        let mut by_id: HashMap<String, Signals> = HashMap::new();

        for hit in prefilter_hits {
            let exact_norm = normalize(hit.raw_score);
            let entry = by_id.entry(hit.id.clone()).or_insert_with(|| Signals {
                source: hit.source.clone(),
                exact: 0.0,
                ann: 0.0,
                snippet: String::new(),
                matched_via: Vec::new(),
                matched_ann: false,
            });
            entry.exact = entry.exact.max(exact_norm);
            entry.matched_via.push(hit.which_strategy);
            if entry.snippet.is_empty() {
                if let Some(snippet) = &hit.highlighted_snippet {
                    entry.snippet = snippet.clone();
                }
            }
        }

        for hit in ann_hits {
            let ann_norm = hit.similarity.clamp(0.0, 1.0);
            let entry = by_id.entry(hit.id.clone()).or_insert_with(|| Signals {
                source: hit.source.clone(),
                exact: 0.0,
                ann: 0.0,
                snippet: String::new(),
                matched_via: Vec::new(),
                matched_ann: false,
            });
            entry.ann = entry.ann.max(ann_norm);
            entry.matched_ann = true;
            if entry.snippet.is_empty() {
                entry.snippet = truncate(&hit.content_preview, SNIPPET_FALLBACK_LEN);
            }
        }

        let mut results: Vec<ScoredResult> = by_id
            .into_iter()
            .map(|(id, s)| ScoredResult {
                id,
                source: s.source,
                score: self.scoring.combine(s.exact, s.ann, 0.0),
                snippet: s.snippet,
                matched_via: s.matched_via,
                matched_ann: s.matched_ann,
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}

/// Normalizes a strategy-native raw score into a comparable [0.0, 1.0]
/// range. Placeholder linear clamp — real normalization is strategy-specific
/// (e.g. ES BM25 scores are unbounded) and belongs on the opportunity list
/// alongside short-circuit threshold tuning.
fn normalize(raw_score: f32) -> f32 {
    raw_score.clamp(0.0, 1.0)
}

fn truncate(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", &s[..idx]),
        None => s.to_string(),
    }
}
