use std::future::Future;
use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::tool::Parameters,
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    handler::server::router::tool::ToolRouter,
};

use rag_core::{RetrievalFunnel, SearchFilters, SearchMode};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The search query text.
    pub query: String,
    /// "keyword" | "semantic" | "hybrid" (default: "hybrid"). Use "keyword"
    /// for exact terms (error codes, function names, precise Chinese
    /// terms); "semantic" for vague, intent-based queries; leave unset to
    /// let the server decide via its short-circuit heuristic.
    pub mode: Option<String>,
    pub source: Option<String>,
    pub language: Option<String>,
    /// Max results to return (default: 10).
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct KeywordSearchParams {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct VectorSearchParams {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FetchByIdParams {
    /// The id of a chunk/document previously returned by search,
    /// keyword_search, or vector_search.
    pub id: String,
}

fn parse_mode(mode: Option<String>) -> SearchMode {
    match mode.as_deref() {
        Some("keyword") => SearchMode::Keyword,
        Some("semantic") => SearchMode::Semantic,
        _ => SearchMode::Hybrid,
    }
}

fn to_mcp_error(err: rag_core::RagError) -> McpError {
    match &err {
        // Not-found is a data condition, not an internal failure -- mapping
        // it to its own error class (rather than "internal error") lets the
        // calling agent tell "bad id" apart from "backend broken".
        rag_core::RagError::NotFound(id) => {
            McpError::invalid_request(format!("document not found: {id}"), None)
        }
        _ => McpError::internal_error(err.to_string(), None),
    }
}

fn to_json(value: impl serde::Serialize) -> Result<String, McpError> {
    serde_json::to_string(&value)
        .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))
}

/// Thin MCP-facing wrapper around `RetrievalFunnel`. All reasoning about
/// query shape, re-querying, and when to fetch full content stays with the
/// calling agent — this struct only maps MCP tool calls onto funnel calls
/// and shapes the JSON response (see SPEC.md: progressive disclosure).
#[derive(Clone)]
pub struct RagMcpServer {
    funnel: Arc<RetrievalFunnel>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl RagMcpServer {
    pub fn new(funnel: Arc<RetrievalFunnel>) -> Self {
        Self {
            funnel,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Run the full retrieval funnel (keyword pre-filter -> conditional ANN -> scoring) and return ranked, snippet-level results. Default mode is 'hybrid': the server automatically decides whether to run semantic search based on how confident the keyword match is. Use fetch_by_id afterward to get full content for a specific result."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<String, McpError> {
        let mode = parse_mode(params.mode);
        let filters = SearchFilters {
            source: params.source,
            language: params.language,
        };
        let results = self
            .funnel
            .search(&params.query, mode, filters, params.limit)
            .await
            .map_err(to_mcp_error)?;
        to_json(results)
    }

    #[tool(
        description = "Run only the exact/keyword pre-filter stage (Elasticsearch, tsvector, or pg_trgm depending on content). Use this when you already know the query is an exact term -- an error code, function name, or precise phrase -- and want to skip semantic search entirely for speed and precision."
    )]
    async fn keyword_search(
        &self,
        Parameters(params): Parameters<KeywordSearchParams>,
    ) -> Result<String, McpError> {
        let results = self
            .funnel
            .keyword_search(&params.query, params.limit)
            .await
            .map_err(to_mcp_error)?;
        to_json(results)
    }

    #[tool(
        description = "Run only the ANN vector search stage. Use this when the query is vague or intent-based and keyword matching is unlikely to help."
    )]
    async fn vector_search(
        &self,
        Parameters(params): Parameters<VectorSearchParams>,
    ) -> Result<String, McpError> {
        let results = self
            .funnel
            .vector_search(&params.query, params.limit)
            .await
            .map_err(to_mcp_error)?;
        to_json(results)
    }

    #[tool(
        description = "Fetch the full content of a specific chunk or document by id, after reviewing snippets from search/keyword_search/vector_search."
    )]
    async fn fetch_by_id(
        &self,
        Parameters(params): Parameters<FetchByIdParams>,
    ) -> Result<String, McpError> {
        let doc = self
            .funnel
            .fetch_by_id(&params.id)
            .await
            .map_err(to_mcp_error)?;
        to_json(doc)
    }
}

#[tool_handler]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            instructions: Some(
                "Retrieval primitives over a Chinese/English/code knowledge base. \
                 Prefer `search` by default; use `keyword_search`/`vector_search` \
                 directly when you already know the query shape."
                    .into(),
            ),
            ..Default::default()
        }
    }
}
