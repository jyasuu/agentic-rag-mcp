//! End-to-end integration tests for the *wiring* — the real funnel built by
//! `build_funnel` (the exact code path `main` runs), exercised through the
//! funnel's public API. The funnel logic itself is already unit-tested in
//! `rag-core`; these tests cover the "integration wiring only" ticket: do the
//! real backends, wired together, actually answer searches?
//!
//! Env-gated like every backend test in this crate: they run only when
//! `RAG_MCP_DATABASE_URL`, `RAG_MCP_ELASTICSEARCH_URL`, and an embedding
//! backend (`RAG_MCP_OLLAMA_URL` or `RAG_MCP_EMBEDDING_MODEL_DIR`) are all
//! set, and are skipped -- not failed -- otherwise.
//!
//! Each test builds its own funnel *on its own tokio runtime*. A shared
//! `OnceCell` is deliberately NOT used: `#[tokio::test]` gives every test a
//! fresh runtime, and sqlx pools / reqwest clients bind background tasks to
//! the runtime that created them. A pool or HTTP client created on test A's
//! runtime silently breaks when test A finishes and its runtime is dropped
//! (connects time out with `PoolTimedOut`, reqwest dies with "dispatch task
//! is gone") -- so per-test construction is the correct, if slightly more
//! expensive, pattern here.

use std::sync::Arc;
use std::time::Duration;

use rag_core::{Embedder, RetrievalFunnel, SearchFilters, SearchMode};

use crate::config::Config;
use crate::embedder::{BgeM3Embedder, OllamaEmbedder};
use crate::es::EsClient;
use crate::es::IK_ANALYZER;
use crate::state::AppState;
use crate::testutil::{
    apply_schema, cleanup_documents, insert_document, test_pool, unique_token,
};
use crate::wiring::build_funnel;

struct Integration {
    funnel: Arc<RetrievalFunnel>,
    pool: sqlx::PgPool,
    es: EsClient,
    index: String,
}

async fn integration() -> Option<Arc<Integration>> {
    let (pool, es_url, embed) = match (
        test_pool().await,
        std::env::var("RAG_MCP_ELASTICSEARCH_URL").ok(),
        ollama_backend().or_else(model_dir_backend),
    ) {
        (Some(pool), Some(es_url), Some(embed)) => (pool, es_url, embed),
        _ => {
            eprintln!(
                "skipping integration tests: RAG_MCP_DATABASE_URL / \
                 RAG_MCP_ELASTICSEARCH_URL / (RAG_MCP_OLLAMA_URL or \
                 RAG_MCP_EMBEDDING_MODEL_DIR) not all set"
            );
            return None;
        }
    };

    apply_schema(&pool).await;
    // Generous timeout: this suite runs against the same single-node
    // cluster as the ES pre-filter tests, concurrently.
    let es = EsClient::new(&es_url, Duration::from_secs(30))
        .expect("RAG_MCP_ELASTICSEARCH_URL set but client failed to build");
    let index = format!("rag-itg-{}", unique_token("idx"));
    es.ensure_index(&index, IK_ANALYZER)
        .await
        .expect("integration test index should be created");

    let config = Config {
        bind_addr: "127.0.0.1:0".parse().expect("static addr"),
        auth_token: "test".into(),
        database_url: std::env::var("RAG_MCP_DATABASE_URL").unwrap(),
        elasticsearch_url: es_url,
        es_index: index.clone(),
        embedding_model_dir: embed.model_dir.clone(),
        ollama_url: embed.ollama_url.clone(),
        ollama_model: embed.ollama_model.clone(),
        connect_timeout: Duration::from_secs(5),
    };
    let funnel = build_funnel(
        &config,
        AppState {
            pg_pool: pool.clone(),
            es: es.clone(),
        },
    )
    .expect("funnel should build");
    Some(Arc::new(Integration { funnel, pool, es, index }))
}

/// Which embedding backend the integration suite should use: the remote
/// Ollama endpoint when `RAG_MCP_OLLAMA_URL` is set (preferred -- no local
/// ONNX session to serialize on), else the local ONNX model dir.
struct EmbedBackend {
    ollama_url: Option<String>,
    ollama_model: String,
    model_dir: Option<std::path::PathBuf>,
}

fn ollama_backend() -> Option<EmbedBackend> {
    let url = std::env::var("RAG_MCP_OLLAMA_URL").ok()?;
    Some(EmbedBackend {
        ollama_url: Some(url),
        ollama_model: std::env::var("RAG_MCP_OLLAMA_MODEL").unwrap_or_else(|_| "bge-m3".into()),
        model_dir: None,
    })
}

fn model_dir_backend() -> Option<EmbedBackend> {
    Some(EmbedBackend {
        ollama_url: None,
        ollama_model: "bge-m3".into(),
        model_dir: std::env::var_os("RAG_MCP_EMBEDDING_MODEL_DIR").map(Into::into),
    })
}

/// ES is near-real-time: poll until `id` is searchable in the shared index.
async fn search_until_visible(itg: &Integration, query: &str, id: &str) {
    for _ in 0..40 {
        if let Ok(hits) = itg.funnel.keyword_search(query, Some(50)).await {
            if hits.iter().any(|h| h.id == id) {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("expected {id} to become searchable in the shared index");
}

async fn cleanup_es_doc(itg: &Integration, id: &str) {
    let _ = itg.es.delete_document(&itg.index, id).await;
}

#[tokio::test]
async fn keyword_search_returns_tsvector_hits_end_to_end() {
    let Some(itg) = integration().await else {
        return;
    };
    let token = unique_token("kw");
    let id = format!("itg-{token}");
    insert_document(
        &itg.pool,
        &id,
        "wiki/api.md",
        &format!("The function {token} validates the payload and returns a token."),
    )
    .await;

    let hits = itg.funnel.keyword_search(&token, None).await.expect("keyword search should work");

    assert!(
        hits.iter().any(|h| h.id == id),
        "tsvector exact match should surface {id}, got {hits:?}"
    );

    cleanup_documents(&itg.pool, "itg-kw").await;
    cleanup_es_doc(&itg, &id).await;
}

#[tokio::test]
async fn keyword_search_falls_back_to_pg_trgm_for_chinese() {
    let Some(itg) = integration().await else {
        return;
    };
    let token = unique_token("trgm");
    // The unique token is pure hex (from the testutil `unique_term` style) so
    // trigram similarity, not ES, is what matches: this exercises the
    // fallback branch of `FallbackPreFilter` (ES has no match -> pg_trgm).
    let term = crate::testutil::unique_term();
    let id = format!("itg-{token}");
    insert_document(
        &itg.pool,
        &id,
        "wiki/zh.md",
        &format!("系统发生错误 {term} 需要立即处理"),
    )
    .await;

    let hits = itg.funnel.keyword_search(&term, None).await.expect("keyword search should work");

    assert!(
        hits.iter().any(|h| h.id == id),
        "pg_trgm fallback should surface {id}, got {hits:?}"
    );

    cleanup_documents(&itg.pool, "itg-trgm").await;
    cleanup_es_doc(&itg, &id).await;
}

#[tokio::test]
async fn vector_search_returns_semantic_results_end_to_end() {
    let Some(itg) = integration().await else {
        return;
    };
    let pool = &itg.pool;
    let token = unique_token("vec");
    let id_a = format!("itg-{token}-apple");
    let id_b = format!("itg-{token}-physics");

    // Seed embeddings with a temporary embedder (dropped before the shared
    // funnel's query-side embedding runs, keeping memory flat). Prefer the
    // remote Ollama backend when configured; fall back to the local ONNX
    // session.
    let seeder: Box<dyn Embedder> = match std::env::var("RAG_MCP_OLLAMA_URL").ok() {
        Some(url) => Box::new(
            OllamaEmbedder::new(
                url,
                std::env::var("RAG_MCP_OLLAMA_MODEL").unwrap_or_else(|_| "bge-m3".into()),
            )
            .expect("ollama client should build"),
        ),
        None => {
            let model_dir = std::env::var("RAG_MCP_EMBEDDING_MODEL_DIR").expect("set in integration()");
            Box::new(BgeM3Embedder::load(std::path::Path::new(&model_dir)).expect("model loads"))
        }
    };
    let va = seeder.embed("苹果汁的制作方法介绍").await.expect("embed a");
    let vb = seeder.embed("量子力学的基本理论").await.expect("embed b");

    insert_document(&pool, &id_a, "wiki/zh.md", &format!("苹果汁的制作方法介绍 {token}")).await;
    insert_document(&pool, &id_b, "wiki/zh.md", &format!("量子力学的基本理论 {token}")).await;
    insert_embedding(&pool, &id_a, &va).await;
    insert_embedding(&pool, &id_b, &vb).await;
    drop(seeder);

    let hits = itg.funnel.vector_search("苹果", None).await.expect("vector search should work");

    assert!(
        hits.iter().any(|h| h.id == id_a),
        "semantically related doc should rank, got {hits:?}"
    );

    cleanup_documents(&pool, &format!("itg-{token}")).await;
    cleanup_es_doc(&itg, &id_a).await;
    cleanup_es_doc(&itg, &id_b).await;
}

#[tokio::test]
async fn hybrid_search_returns_results_and_honors_mode() {
    let Some(itg) = integration().await else {
        return;
    };
    let pool = &itg.pool;
    let token = unique_token("hyb");
    let id = format!("itg-{token}");
    insert_document(
        pool,
        &id,
        "wiki/errors.md",
        &format!("{token} is raised when a connection attempt fails."),
    )
    .await;
    itg.es
        .index_document(&itg.index, &id, "wiki/errors.md", &format!("{token} connection failed"))
        .await
        .expect("indexing should succeed");

    // Default (hybrid) search returns the doc; the ES index holds it, so the
    // pre-filter surfaces it even though the funnel also runs ANN (only one
    // pre-filter hit -> not "confident" by the short-circuit config).
    search_until_visible(&itg, &token, &id).await;
    let hits = itg
        .funnel
        .search(&token, SearchMode::Hybrid, SearchFilters::default(), None)
        .await
        .expect("hybrid search should work");
    assert!(hits.iter().any(|h| h.id == id), "hybrid should surface {id}, got {hits:?}");

    // Explicit keyword mode also works and, for a unique exact token, does
    // not need the ANN stage at all.
    let kw_hits = itg
        .funnel
        .search(&token, SearchMode::Keyword, SearchFilters::default(), None)
        .await
        .expect("keyword mode should work");
    assert!(kw_hits.iter().any(|h| h.id == id), "keyword mode should surface {id}");

    cleanup_documents(pool, "itg-hyb").await;
    cleanup_es_doc(&itg, &id).await;
}

#[tokio::test]
async fn fetch_by_id_returns_full_content_and_not_found() {
    let Some(itg) = integration().await else {
        return;
    };
    let pool = &itg.pool;
    let token = unique_token("fetch");
    let id = format!("itg-{token}");
    let content = format!("The full body of document {token}, beyond any snippet.");
    insert_document(pool, &id, "wiki/api.md", &content).await;

    let doc = itg.funnel.fetch_by_id(&id).await.expect("valid id should fetch");
    assert_eq!(doc.content, content);

    let missing = format!("itg-{token}-never-inserted");
    let err = itg.funnel.fetch_by_id(&missing).await.expect_err("unknown id should error");
    assert!(err.to_string().contains(&missing), "error should name the id, got {err}");

    cleanup_documents(pool, "itg-fetch").await;
}

// Test-only helpers, private to this module.

async fn insert_embedding(pool: &sqlx::PgPool, id: &str, embedding: &[f32]) {
    let literal = format!(
        "[{}]",
        embedding.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
    );
    sqlx::query("INSERT INTO chunk_embeddings (id, embedding) VALUES ($1, $2::vector)")
        .bind(id)
        .bind(&literal)
        .execute(pool)
        .await
        .unwrap();
}
