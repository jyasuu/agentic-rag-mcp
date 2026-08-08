/// Fixed weighting coefficients for combining pre-filter and ANN scores.
///
/// Deliberately not per-query tunable in v1 (see opportunity list) — kept as
/// one config struct so tuning later is a one-place change, not a hunt
/// through funnel logic.
#[derive(Debug, Clone, Copy)]
pub struct ScoringConfig {
    pub w_exact: f32,
    pub w_ann: f32,
    pub w_metadata: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        // Starting coefficients — expected to need calibration once real
        // query traffic is observed (see opportunity list).
        Self {
            w_exact: 0.6,
            w_ann: 0.35,
            w_metadata: 0.05,
        }
    }
}

impl ScoringConfig {
    /// Combines a normalized exact-match score, ANN similarity, and an
    /// optional metadata score (0.0 if absent) into a single ranking score.
    /// Inputs are expected to already be normalized to a comparable
    /// [0.0, 1.0] range by the caller (`RetrievalFunnel`).
    pub fn combine(&self, exact: f32, ann: f32, metadata: f32) -> f32 {
        self.w_exact * exact + self.w_ann * ann + self.w_metadata * metadata
    }
}

/// Threshold config deciding when the pre-filter stage is "confident enough"
/// to short-circuit the ANN stage in `Hybrid` mode.
#[derive(Debug, Clone, Copy)]
pub struct ShortCircuitConfig {
    /// Minimum number of pre-filter hits required to consider skipping ANN.
    pub min_hit_count: usize,
    /// Minimum raw_score (strategy-native) a top hit must have.
    pub min_top_score: f32,
}

impl Default for ShortCircuitConfig {
    fn default() -> Self {
        // Rough starting point — see opportunity list: needs calibration
        // against real query data per pre-filter strategy.
        Self {
            min_hit_count: 3,
            min_top_score: 0.5,
        }
    }
}
