"""
tools/rag_tools.py
==================
Factory for per-agent Confluence documentation search tools (RAG).

Each sub-agent gets its own ``search_docs`` tool that is pre-scoped to its
module, so it only searches its own Confluence space.

Usage
-----
    from tools.rag_tools import make_search_docs_tool

    kpi_search = make_search_docs_tool(module="kpi", prefix="kpi")
    # Pass kpi_search into create_react_agent's tools list.
"""
from __future__ import annotations

import logging

from langchain.tools import tool as lc_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings
from db.docs import search_docs as _search_docs

logger = logging.getLogger(__name__)

# Singleton embedding model — created once and reused across all tools.
_embeddings_model: GoogleGenerativeAIEmbeddings | None = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model
        )
    return _embeddings_model


def _docs_dsn() -> str:
    """Return the DSN to query agent_docs from. Falls back to conv_db_dsn."""
    return settings.docs_db_dsn or settings.conv_db_dsn


def make_search_docs_tool(module: str, prefix: str):
    """
    Build a LangChain @tool that performs semantic search over the
    agent_docs table scoped to *module*.

    Parameters
    ----------
    module : str
        One of 'kpi', 'iot', 'amm'.  Used to filter the vector table.
    prefix : str
        Short label used to name the tool (e.g. 'kpi' → 'kpi_search_docs').
    """
    tool_name = f"{prefix}_search_docs"
    tool_description = (
        f"Search {module.upper()} module documentation imported from Confluence. "
        f"Use this to understand business rules, field meanings, formulas, "
        f"or module-specific concepts BEFORE writing a SQL query. "
        f"Input: a natural-language question or keyword phrase."
    )

    @lc_tool(name=tool_name, description=tool_description)
    def _search(query: str) -> str:
        """Perform semantic search against the module's Confluence docs."""
        if not query or not query.strip():
            return "Please provide a non-empty search query."

        dsn = _docs_dsn()
        try:
            embeddings = _get_embeddings()
            query_vec = embeddings.embed_query(query)
            results = _search_docs(dsn, module, query_vec, top_k=5)

            if not results:
                return (
                    f"No documentation found for '{query}' in the {module.upper()} space. "
                    "Proceed using your knowledge of the schema."
                )

            parts: list[str] = []
            for r in results:
                sim_pct = round(r["similarity"] * 100, 1)
                src = r.get("url") or "N/A"
                parts.append(
                    f"### {r['title']}  (similarity {sim_pct}%)\n"
                    f"{r['content_chunk'].strip()}\n"
                    f"_Source: {src}_"
                )
            return "\n\n---\n\n".join(parts)

        except Exception as exc:
            logger.error("rag_tool '%s' error: %s", tool_name, exc, exc_info=True)
            return f"Documentation search failed: {exc}"

    return _search
