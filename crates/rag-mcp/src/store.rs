//! Postgres `ContentStore` — backs `fetch_by_id`, which returns full chunk
//! content after an agent has reviewed snippets from a prior search
//! (SPEC.md user story: progressive disclosure). Queries the `documents`
//! table (see `migrations/0001_documents.sql`) by primary key, returning a
//! clear not-found error for unknown ids (SPEC.md: full content requires a
//! follow-up `fetch_by_id` call).

use async_trait::async_trait;
use rag_core::{ContentStore, Document, RagError, RagResult};
use sqlx::PgPool;
use sqlx::Row;

pub struct PostgresContentStore {
    pool: PgPool,
}

impl PostgresContentStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl ContentStore for PostgresContentStore {
    async fn fetch(&self, id: &str) -> RagResult<Document> {
        let row = sqlx::query(
            "SELECT id, source, content, metadata::text AS metadata FROM documents WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| RagError::ContentStore(format!("fetch query failed: {e}")))?;

        let Some(row) = row else {
            return Err(RagError::NotFound(id.to_string()));
        };

        Ok(Document {
            id: row.get::<String, _>("id"),
            source: row.get::<String, _>("source"),
            content: row.get::<String, _>("content"),
            metadata: serde_json::from_str(&row.get::<String, _>("metadata"))
                .unwrap_or(serde_json::Value::Null),
        })
    }
}

// Real-Postgres integration tests, following the trigram.rs pattern: run
// against `RAG_MCP_DATABASE_URL`, skipped (not failed) when unset. Each test
// embeds a unique token in the fixture id so concurrent runs never collide.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{apply_schema, cleanup_documents, insert_document, test_pool, unique_token};

    #[tokio::test]
    async fn returns_full_document_for_valid_id() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        apply_schema(&pool).await;
        let token = unique_token("store");
        let id = format!("store-{token}");
        let content = format!("The full body of document {token}.");
        insert_document(&pool, &id, "wiki/api.md", &content).await;

        let store = PostgresContentStore::new(pool.clone());
        let doc = store.fetch(&id).await.expect("valid id should fetch");

        assert_eq!(doc.id, id);
        assert_eq!(doc.source, "wiki/api.md");
        assert_eq!(doc.content, content);

        cleanup_documents(&pool, "store").await;
    }

    #[tokio::test]
    async fn unknown_id_returns_clear_not_found_error() {
        let Some(pool) = test_pool().await else {
            eprintln!("skipping: RAG_MCP_DATABASE_URL not set");
            return;
        };
        apply_schema(&pool).await;
        let token = unique_token("missing");
        let id = format!("store-{token}-never-inserted");

        let store = PostgresContentStore::new(pool);
        let err = store.fetch(&id).await.expect_err("unknown id should error");

        match err {
            RagError::NotFound(got) => assert_eq!(got, id),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }
}
