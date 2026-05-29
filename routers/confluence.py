"""
routers/confluence.py
=====================
FastAPI router for syncing Confluence documentation into the pgvector store.

Endpoints
---------
POST /docs/sync    — Fetch pages, embed, clear old data, insert new chunks.
GET  /docs/status  — Row counts and last-sync timestamps per module.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

import psycopg2
import requests
from fastapi import APIRouter, HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

from config import settings
from db.docs import clear_module_docs, get_doc_stats, insert_docs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/docs", tags=["Documentation"])

# ── Text helpers ──────────────────────────────────────────────────────────────

_CHUNK_SIZE = 800       # characters (~200 tokens)
_CHUNK_OVERLAP = 100    # overlap between consecutive chunks


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)      # basic HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + _CHUNK_SIZE])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


# ── Confluence fetching ───────────────────────────────────────────────────────

def _fetch_confluence_pages(space_key: str) -> list[dict]:
    """
    Paginate through all pages in a Confluence space.

    Returns
    -------
    list of dicts with keys: page_id, title, url, content (plain text)
    """
    if not (settings.confluence_url and settings.confluence_username and settings.confluence_api_token):
        raise ValueError(
            "Confluence credentials missing. Set CONFLUENCE_URL, "
            "CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN in .env."
        )

    base = settings.confluence_url.rstrip("/")
    auth = (settings.confluence_username, settings.confluence_api_token)
    headers = {"Accept": "application/json"}

    pages: list[dict] = []
    start = 0
    limit = 50

    while True:
        resp = requests.get(
            f"{base}/rest/api/content",
            params={
                "spaceKey": space_key,
                "type": "page",
                "expand": "body.storage",
                "start": start,
                "limit": limit,
            },
            auth=auth,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        for page in results:
            body_html = page.get("body", {}).get("storage", {}).get("value", "")
            content = _strip_html(body_html)
            if len(content) < 50:
                continue  # skip empty / boilerplate pages
            pages.append(
                {
                    "page_id": page["id"],
                    "title": page["title"],
                    "url": f"{base}/wiki{page.get('_links', {}).get('webui', '')}",
                    "content": content,
                }
            )

        if len(results) < limit:
            break
        start += limit
        logger.info("Fetched %d pages so far from space '%s' ...", len(pages), space_key)

    return pages


# ── Request / response schemas ────────────────────────────────────────────────

class SyncRequest(BaseModel):
    module: Literal["kpi", "iot", "amm"] = Field(
        description="Which agent module to sync docs for."
    )
    space_key: str = Field(
        description="Confluence space key (e.g. 'KPISPACE', 'IOTDOCS', 'ASSETMGMT')."
    )


class SyncResponse(BaseModel):
    module: str
    space_key: str
    pages_fetched: int
    chunks_inserted: int
    deleted_previous: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/sync", response_model=SyncResponse, summary="Sync Confluence space to vector store")
async def sync_confluence_docs(req: SyncRequest):
    """
    Full refresh for one module:
    1. Fetch all pages from the Confluence space.
    2. Strip HTML and split into overlapping text chunks.
    3. Generate embeddings with Google text-embedding-004.
    4. Delete all previous chunks for this module.
    5. Insert the new chunks.
    """
    dsn = settings.docs_db_dsn or settings.conv_db_dsn

    try:
        logger.info(
            "Confluence sync started — module=%s space=%s", req.module, req.space_key
        )

        # 1. Fetch pages
        pages = _fetch_confluence_pages(req.space_key)
        if not pages:
            raise HTTPException(
                status_code=422,
                detail=f"No usable pages found in Confluence space '{req.space_key}'.",
            )
        logger.info("Fetched %d pages from space '%s'.", len(pages), req.space_key)

        # 2. Chunk content
        chunk_triples: list[tuple[dict, str, int]] = []  # (page_meta, chunk_text, idx)
        for page in pages:
            for i, chunk in enumerate(_chunk_text(page["content"])):
                chunk_triples.append((page, chunk, i))

        logger.info("Created %d chunks total.", len(chunk_triples))

        # 3. Embed in batches (Google API has per-request limits)
        embeddings_model = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
        chunk_texts = [ct[1] for ct in chunk_triples]
        vectors = embeddings_model.embed_documents(chunk_texts)

        # 4. Clear previous data
        deleted = clear_module_docs(dsn, req.module)

        # 5. Build rows and insert
        rows = [
            {
                "module": req.module,
                "page_id": page["page_id"],
                "title": page["title"],
                "url": page["url"],
                "content_chunk": chunk_text,
                "chunk_index": chunk_idx,
                "embedding": "[" + ",".join(map(str, vec)) + "]",
            }
            for (page, chunk_text, chunk_idx), vec in zip(chunk_triples, vectors)
        ]
        inserted = insert_docs(dsn, rows)

        logger.info(
            "Confluence sync complete — module=%s pages=%d chunks=%d",
            req.module, len(pages), inserted,
        )
        return SyncResponse(
            module=req.module,
            space_key=req.space_key,
            pages_fetched=len(pages),
            chunks_inserted=inserted,
            deleted_previous=deleted,
        )

    except requests.HTTPError as exc:
        logger.error("Confluence API HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Confluence API error: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected sync error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status", summary="Documentation sync status per module")
async def docs_status():
    """Return chunk counts, page counts, and last-sync timestamps per module."""
    dsn = settings.docs_db_dsn or settings.conv_db_dsn
    try:
        stats = get_doc_stats(dsn)
        return {"modules": stats}
    except psycopg2.Error as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
