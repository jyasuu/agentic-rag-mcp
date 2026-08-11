//! Test-only helpers shared by the backend integration tests (tsvector,
//! content store, ES-backed strategies, and the end-to-end wiring suite).
//! Mirrors the pattern established by the `tsvector` tests (SPEC.md: "each
//! [strategy] against a minimal fixture dataset"): real-Postgres tests run
//! against `RAG_MCP_DATABASE_URL` and are skipped -- not failed -- when it
//! isn't set, so `cargo test` stays green without Postgres.
//!
//! The `migrations/*.sql` files are applied idempotently so tests can create
//! whatever schema they need (extensions, tables, indexes) on a disposable
//! or shared dev database without depending on a migration runner.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns `Some(pool)` when `RAG_MCP_DATABASE_URL` is set, else `None`.
/// Uses an explicit max-connections and a generous acquire timeout: many
/// tests create their own pool (this helper is called per test), and the
/// shared sandbox runs the whole suite concurrently with Elasticsearch, so a
/// transiently slow Postgres must not translate into a pool exhaustion
/// failure.
pub async fn test_pool() -> Option<PgPool> {
    let url = std::env::var("RAG_MCP_DATABASE_URL").ok()?;
    Some(
        PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(60))
            .connect(&url)
            .await
            .expect("RAG_MCP_DATABASE_URL set but Postgres unreachable"),
    )
}

/// Applies every `migrations/*.sql` file in lexicographic order, splitting on
/// `;` so each statement runs independently. Idempotent -- the migrations use
/// `IF NOT EXISTS` -- so it is safe to call from any test that needs the
/// schema, even while other tests are inserting/querying the same tables.
///
/// Runs on one dedicated connection under a session-level advisory lock:
/// Postgres `CREATE TABLE IF NOT EXISTS` is not race-free (two transactions
/// can both pass the catalog existence check and then one fails inserting
/// into `pg_type`), and tests in this file run concurrently.
pub async fn apply_schema(pool: &PgPool) {
    const SCHEMA_LOCK_KEY: i64 = 72_901;
    let mut conn = pool
        .acquire()
        .await
        .expect("pool should be able to acquire a connection");
    sqlx::query("SELECT pg_advisory_lock($1)")
        .bind(SCHEMA_LOCK_KEY)
        .execute(&mut *conn)
        .await
        .expect("advisory lock should be acquirable");

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("migrations dir should exist")
        .map(|e| e.expect("read_dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "sql"))
        .collect();
    files.sort();

    for file in files {
        let sql = std::fs::read_to_string(&file).expect("migration should be readable");
        for stmt in sql.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            sqlx::query(stmt)
                .execute(&mut *conn)
                .await
                .expect("schema statement should apply");
        }
    }

    sqlx::query("SELECT pg_advisory_unlock($1)")
        .bind(SCHEMA_LOCK_KEY)
        .execute(&mut *conn)
        .await
        .expect("advisory unlock should succeed");
}

/// A per-process unique token so fixtures never collide with other rows in a
/// shared/development database, in the same style as the tsvector tests.
pub fn unique_token(test_name: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{test_name}{}{n}", std::process::id())
}

pub async fn insert_document(pool: &PgPool, id: &str, source: &str, content: &str) {
    sqlx::query("INSERT INTO documents (id, source, content) VALUES ($1, $2, $3)")
        .bind(id)
        .bind(source)
        .bind(content)
        .execute(pool)
        .await
        .unwrap();
}

pub async fn cleanup_documents(pool: &PgPool, id_prefix: &str) {
    sqlx::query("DELETE FROM documents WHERE id LIKE $1")
        .bind(format!("{id_prefix}%"))
        .execute(pool)
        .await
        .unwrap();
}
