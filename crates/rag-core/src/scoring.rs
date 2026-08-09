/// Reciprocal Rank Fusion (RRF) configuration, passed through to the
/// Elasticsearch hybrid request (`rank: { rrf: { window_size, rank_constant } }`).
/// Replaces the weighted `ScoringConfig` and the hybrid short-circuit
/// heuristic: RRF is rank-based, so there are no score-normalization
/// coefficients to calibrate and no "is the keyword stage confident enough"
/// decision to make — Elasticsearch owns the fused ranking.
#[derive(Debug, Clone, Copy)]
pub struct RrfConfig {
    /// RRF window size — the number of top hits each ranked list contributes
    /// to the fused score.
    pub window_size: usize,
    /// RRF rank constant (k in the standard `1 / (k + rank)` formula).
    pub rank_constant: usize,
}

impl Default for RrfConfig {
    fn default() -> Self {
        // Elasticsearch's documented defaults; exposed via env
        // (RAG_MCP_RRF_WINDOW_SIZE / RAG_MCP_RRF_RANK_CONSTANT) so fusion
        // behavior can be tuned without code changes.
        Self {
            window_size: 100,
            rank_constant: 60,
        }
    }
}
