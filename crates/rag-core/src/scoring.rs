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

/// Which fusion strategy the hybrid retrieval path uses to combine the BM25
/// and kNN result lists. Selected at startup (not per query) so the operator
/// picks the method whose assumptions match their corpus and their trust in
/// BM25-vs-cosine score comparability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum HybridFusion {
    /// Two requests, fused client-side with reciprocal rank fusion
    /// (`rrf_fuse`). Rank-based, no score-magnitude sensitivity, free license.
    #[default]
    ClientRrf,
    /// Two requests, each list's raw scores normalized (min-max or L2) then
    /// combined by a weighted arithmetic mean. Score-magnitude-aware and
    /// per-list-weightable, still free license.
    NormalizedMean,
    /// One combined request carrying `rank: { rrf }`, fused natively by the
    /// engine. Requires Elasticsearch Platinum/Enterprise (license-gated).
    ServerRrf,
}

impl std::str::FromStr for HybridFusion {    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "client-rrf" => Ok(Self::ClientRrf),
            "normalized-mean" => Ok(Self::NormalizedMean),
            "server-rrf" => Ok(Self::ServerRrf),
            other => Err(format!(
                "unknown hybrid fusion method {other:?} (expected client-rrf, normalized-mean, or server-rrf)"
            )),
        }
    }
}

/// How each sub-query's raw scores are scaled before the weighted-mean
/// combination of `HybridFusion::NormalizedMean`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ScoreNormalization {
    /// `(s - min) / (max - min)` over the returned list. Outlier-sensitive:
    /// one unusually high score compresses the rest.
    #[default]
    MinMax,
    /// Divide the list by its L2 norm; keeps relative scale between the two
    /// lists without the extreme outlier pressure of min-max.
    L2,
}

impl std::str::FromStr for ScoreNormalization {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "min-max" => Ok(Self::MinMax),
            "l2" => Ok(Self::L2),
            other => Err(format!(
                "unknown score normalization {other:?} (expected min-max or l2)"
            )),
        }
    }
}

/// Per-list weights for the weighted-mean fusion, applying to the keyword
/// (BM25) and vector (kNN) lists respectively. Only meaningful under
/// `HybridFusion::NormalizedMean`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FusionWeights {
    pub keyword: f32,
    pub vector: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            keyword: 0.5,
            vector: 0.5,
        }
    }
}

/// The complete hybrid fusion configuration, selected at startup and passed
/// through to the ES retrieval backend alongside `RrfConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridFusionConfig {
    pub method: HybridFusion,
    pub normalization: ScoreNormalization,
    pub weights: FusionWeights,
}

impl Default for HybridFusionConfig {
    fn default() -> Self {
        Self {
            method: HybridFusion::ClientRrf,
            normalization: ScoreNormalization::MinMax,
            weights: FusionWeights::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_fusion_parses_env_style_names() {
        assert_eq!("client-rrf".parse::<HybridFusion>().unwrap(), HybridFusion::ClientRrf);
        assert_eq!("normalized-mean".parse::<HybridFusion>().unwrap(), HybridFusion::NormalizedMean);
        assert_eq!("server-rrf".parse::<HybridFusion>().unwrap(), HybridFusion::ServerRrf);
    }

    #[test]
    fn hybrid_fusion_rejects_unknown_names() {
        assert!("rank-rrf".parse::<HybridFusion>().is_err());
        assert!("".parse::<HybridFusion>().is_err());
    }

    #[test]
    fn score_normalization_parses_env_style_names() {
        assert_eq!("min-max".parse::<ScoreNormalization>().unwrap(), ScoreNormalization::MinMax);
        assert_eq!("l2".parse::<ScoreNormalization>().unwrap(), ScoreNormalization::L2);
        assert!("".parse::<ScoreNormalization>().is_err());
    }

    #[test]
    fn default_config_is_client_rrf_with_equal_weights() {
        let cfg = HybridFusionConfig::default();
        assert_eq!(cfg.method, HybridFusion::ClientRrf);
        assert_eq!(cfg.normalization, ScoreNormalization::MinMax);
        assert_eq!(cfg.weights.keyword, 0.5);
        assert_eq!(cfg.weights.vector, 0.5);
    }
}
