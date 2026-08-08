//! Local BGE-M3 query-embedding via `ort` (ONNX Runtime) (SPEC.md user
//! story 15): no external embedding API dependency, cost, or latency for the
//! ANN stage.
//!
//! The embedder loads an ONNX export of BGE-M3 (`input_ids` + `attention_mask`
//! int64 inputs, `last_hidden_state` float32 output, hidden dim 1024) plus a
//! HF `tokenizer.json`, then computes the dense query embedding with the
//! model's documented pooling: mean-pool `last_hidden_state` over non-padding
//! tokens and L2-normalize. Pooling/normalization are pure functions, kept
//! unit-testable; the session run itself is the only ORT-coupled part.
//!
//! Model directory layout (`RAG_MCP_EMBEDDING_MODEL_DIR`):
//!   - `model_int8.onnx` (or `model.onnx`) -- the ONNX graph
//!   - `tokenizer.json` -- HF tokenizer config

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use rag_core::{Embedder, RagError, RagResult};
use tokenizers::Tokenizer;

/// Query text is truncated to this many tokens before embedding; BGE-M3
/// supports far longer inputs, but agent queries are short and a generous cap
/// bounds per-call latency.
const MAX_TOKENS: usize = 512;
pub const EMBEDDING_DIM: usize = 1024;

pub struct BgeM3Embedder {
    /// ORT `Session::run` takes `&mut self`, so the session sits behind a
    /// mutex; `Arc` lets the blocking closure in `embed` move a clone in.
    session: Arc<std::sync::Mutex<Session>>,
    tokenizer: Tokenizer,
}

/// Used when `RAG_MCP_EMBEDDING_MODEL_DIR` is unset: keyword-only
/// deployments keep working, and semantic/hybrid queries fail with a clear,
/// actionable error at call time rather than at startup.
pub struct UnavailableEmbedder;

#[async_trait]
impl Embedder for UnavailableEmbedder {
    async fn embed(&self, _text: &str) -> RagResult<Vec<f32>> {
        Err(RagError::Embedding(
            "no embedding model configured -- set RAG_MCP_EMBEDDING_MODEL_DIR to a directory \
             containing the BGE-M3 ONNX graph and tokenizer.json"
                .into(),
        ))
    }
}

impl BgeM3Embedder {
    /// Loads the ONNX graph and tokenizer from `model_dir`.
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let session = Self::load_session(model_dir)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to load tokenizer from {} -- is the model directory complete? (error: {e})",
                tokenizer_path.display()
            )
        })?;
        Ok(Self {
            session: Arc::new(std::sync::Mutex::new(session)),
            tokenizer,
        })
    }

    fn load_session(model_dir: &Path) -> anyhow::Result<Session> {
        // Prefer the smaller int8 quantized graph; fall back to the fp32 one.
        let model_path = ["model_int8.onnx", "model.onnx"]
            .into_iter()
            .map(|name| model_dir.join(name))
            .find(|p| p.exists())
            .with_context(|| {
                format!(
                    "no model.onnx / model_int8.onnx in {} -- set RAG_MCP_EMBEDDING_MODEL_DIR \
                     to a directory containing the BGE-M3 ONNX export and tokenizer.json",
                    model_dir.display()
                )
            })?;

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create ONNX Runtime session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set ORT optimization level: {e}"))?
            .with_intra_threads(std::thread::available_parallelism().map_or(1, |n| n.get()))
            .map_err(|e| anyhow::anyhow!("failed to set ORT intra-op threads: {e}"))?;
        builder
            .commit_from_file(&model_path)
            .map_err(|e| {
                anyhow::anyhow!("failed to load ONNX model from {}: {e}", model_path.display())
            })
    }

    /// The pure embedding core, split out so the pooling/normalization math is
    /// unit-testable without ORT or the tokenizer.
    fn embed_tokens(input_ids: &[i64], attention_mask: &[i64], hidden: &[f32], dim: usize) -> Vec<f32> {
        let seq_len = input_ids.len();
        let pooled = mean_pool(hidden, seq_len, dim, attention_mask);
        l2_normalize(pooled)
    }
}

#[async_trait]
impl Embedder for BgeM3Embedder {
    async fn embed(&self, text: &str) -> RagResult<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| RagError::Embedding(format!("tokenization failed: {e}")))?;
        let ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .take(MAX_TOKENS)
            .map(|&t| t as i64)
            .collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .take(MAX_TOKENS)
            .map(|&m| m as i64)
            .collect();
        let seq_len = ids.len();
        if seq_len == 0 {
            return Err(RagError::Embedding(
                "text tokenized to zero tokens -- empty query?".into(),
            ));
        }

        // ORT `run` is blocking; offload to a blocking thread so the async
        // runtime isn't stalled by the forward pass. Clones of `ids`/`mask`
        // are moved into the closure; the originals are needed afterward for
        // `embed_tokens` (which receives the model's raw hidden state).
        let session = Arc::clone(&self.session);
        let ids_for_model = ids.clone();
        let mask_for_model = mask.clone();
        let hidden = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let mut session = session
                .lock()
                .map_err(|e| anyhow::anyhow!("embedding session mutex poisoned: {e}"))?;
            let input_ids = Value::from_array(([1usize, seq_len], ids_for_model))
                .map_err(|e| anyhow::anyhow!("failed to build input_ids tensor: {e}"))?;
            let attention_mask = Value::from_array(([1usize, seq_len], mask_for_model))
                .map_err(|e| anyhow::anyhow!("failed to build attention_mask tensor: {e}"))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask
                ])
                .map_err(|e| anyhow::anyhow!("ORT inference failed: {e}"))?;
            let (_, data) = outputs
                .get("last_hidden_state")
                .context("model output last_hidden_state missing")?
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("last_hidden_state is not a float32 tensor: {e}"))?;
            Ok(data.to_vec())
        })
        .await
        .map_err(|e| RagError::Embedding(format!("embedding task panicked: {e}")))?
        .map_err(|e| RagError::Embedding(e.to_string()))?;

        Ok(Self::embed_tokens(&ids, &mask, &hidden, EMBEDDING_DIM))
    }
}

/// Mean-pools the last hidden state over non-padding tokens (BGE-M3's
/// documented dense-embedding pooling). `hidden` is row-major [seq, dim].
fn mean_pool(hidden: &[f32], seq_len: usize, dim: usize, attention_mask: &[i64]) -> Vec<f32> {
    let mut pooled = vec![0.0f32; dim];
    let mut count = 0usize;
    for t in 0..seq_len {
        if attention_mask[t] == 0 {
            continue;
        }
        count += 1;
        let row = &hidden[t * dim..(t + 1) * dim];
        for d in 0..dim {
            pooled[d] += row[d];
        }
    }
    if count > 0 {
        for v in pooled.iter_mut() {
            *v /= count as f32;
        }
    }
    pooled
}

/// L2-normalizes a vector in place, returning `Err`-free zero vector for the
/// degenerate all-zero input (nothing to normalize).
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(ids: &[i64], mask: &[i64], hidden: &[f32], expected: &[f32], tolerance: f32) {
        let got = BgeM3Embedder::embed_tokens(ids, mask, hidden, hidden.len() / ids.len());
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected) {
            assert!(
                (a - b).abs() < tolerance,
                "expected {b} but got {a}"
            );
        }
    }

    #[test]
    fn mean_pool_ignores_padding_tokens() {
        // 2 real tokens (dim 2) + 2 padding tokens, so plain averaging would
        // dilute the signal; mean pooling over the mask must not.
        let ids = [1i64, 2, 3, 3];
        let mask = [1i64, 1, 0, 0];
        // token0=[1.0, 0.0], token1=[0.0, 2.0]
        let hidden = [1.0, 0.0, 0.0, 2.0, 99.0, 99.0, 99.0, 99.0];
        let pooled = mean_pool(&hidden, ids.len(), 2, &mask);
        assert_eq!(pooled, vec![0.5, 1.0]);
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let v = l2_normalize(vec![3.0, 4.0]);
        assert_eq!(v, vec![0.6, 0.8]);

        // All-zero input stays zero (no NaN).
        assert_eq!(l2_normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn embed_tokens_mean_pools_then_normalizes() {
        // Two real tokens, dim 2: pooled = [1.5, 2.0], norm = 2.5.
        let ids = [1i64, 2];
        let mask = [1i64, 1];
        let hidden = [1.0, 2.0, 2.0, 2.0];
        check(&ids, &mask, &hidden, &[0.6, 0.8], 1e-6);
    }

    #[test]
    fn embed_tokens_is_unit_length_for_arbitrary_input() {
        // An independent-source check: whatever the model produces, the
        // embedder's output must be a unit vector.
        let ids = [1i64, 2, 3];
        let mask = [1i64, 1, 1];
        let hidden: Vec<f32> = (0..9).map(|i| (i as f32) * 0.1).collect();
        let got = BgeM3Embedder::embed_tokens(&ids, &mask, &hidden, 3);
        let norm: f32 = got.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    // Real-model integration test: runs when `RAG_MCP_EMBEDDING_MODEL_DIR`
    // points at a directory with the ONNX graph + tokenizer.json (see
    // module docs). Skipped -- not failed -- when unset.
    fn model_dir() -> Option<std::path::PathBuf> {
        std::env::var_os("RAG_MCP_EMBEDDING_MODEL_DIR").map(Into::into)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[tokio::test]
    async fn loads_model_and_embeds_chinese_and_english() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: RAG_MCP_EMBEDDING_MODEL_DIR not set");
            return;
        };
        let embedder = BgeM3Embedder::load(&dir).expect("model should load");

        for text in ["苹果", "apple", "error code connection failed"] {
            let v = embedder.embed(text).await.expect("embedding should succeed");
            assert_eq!(v.len(), EMBEDDING_DIM, "BGE-M3 should produce {EMBEDDING_DIM}-dim vectors");
            let norm: f32 = v.iter().map(|x| x * x).sum();
            assert!((norm - 1.0).abs() < 1e-2, "output should be unit length, norm {norm}");
        }
    }

    #[tokio::test]
    async fn semantically_related_terms_closer_than_unrelated() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: RAG_MCP_EMBEDDING_MODEL_DIR not set");
            return;
        };
        let embedder = BgeM3Embedder::load(&dir).expect("model should load");

        let apple = embedder.embed("苹果").await.unwrap();
        let apple_juice = embedder.embed("苹果汁").await.unwrap();
        let quantum = embedder.embed("量子力学").await.unwrap();

        let related = cosine(&apple, &apple_juice);
        let unrelated = cosine(&apple, &quantum);
        assert!(
            related > unrelated,
            "semantically related terms should be closer (related={related}, unrelated={unrelated})"
        );
    }
}
