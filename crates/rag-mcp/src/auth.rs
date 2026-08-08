use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};

#[derive(Clone)]
pub struct BearerToken(pub String);

/// Minimal shared-secret auth: expects `Authorization: Bearer <token>`.
/// Sufficient for a server serving Jyasu's own agents (Claude Code/Desktop)
/// rather than external/multi-tenant clients — see SPEC.md. Upgrading to
/// OAuth later is additive since `rmcp` already supports it; this
/// middleware layer doesn't need to change shape, just be replaced.
pub async fn require_bearer_token(
    State(expected): State<BearerToken>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match header {
        Some(h) if h.strip_prefix("Bearer ") == Some(expected.0.as_str()) => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
