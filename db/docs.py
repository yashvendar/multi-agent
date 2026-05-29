"""
db/docs.py
==========
Vector storage for agent documentation (Confluence pages).

Requires the pgvector extension on the target PostgreSQL database:
    CREATE EXTENSION IF NOT EXISTS vector;

One shared table ``agent_docs`` holds chunks for all three modules,
discriminated by the ``module`` column ('kpi' | 'iot' | 'amm').
"""
from __future__ import annotations

import logging

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ── SQL ───────────────────────────────────────────────────────────────────────

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_docs (
    id            BIGSERIAL    PRIMARY KEY,
    module        VARCHAR(50)  NOT NULL,          -- 'kpi' | 'iot' | 'amm'
    page_id       VARCHAR(255) NOT NULL,
    title         TEXT         NOT NULL,
    url           TEXT,
    content_chunk TEXT         NOT NULL,
    chunk_index   INTEGER      DEFAULT 0,
    embedding     vector(768),                    -- text-embedding-004 = 768 dims
    synced_at     TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_docs_module
    ON agent_docs(module);

CREATE INDEX IF NOT EXISTS idx_agent_docs_embedding
    ON agent_docs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


# ── Public helpers ────────────────────────────────────────────────────────────

def init_docs_table(dsn: str) -> None:
    """Create the agent_docs table and pgvector indexes if they don't exist."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_INIT_SQL)
        conn.commit()
    logger.info("agent_docs table initialised.")


def clear_module_docs(dsn: str, module: str) -> int:
    """Delete all existing chunks for *module*. Returns the number of rows deleted."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_docs WHERE module = %s", (module,))
            deleted = cur.rowcount
        conn.commit()
    logger.info("Cleared %d rows for module '%s'.", deleted, module)
    return deleted


def insert_docs(dsn: str, rows: list[dict]) -> int:
    """
    Bulk-insert doc chunks.

    Each dict in *rows* must have keys:
        module, page_id, title, url, content_chunk, chunk_index, embedding

    ``embedding`` must be a pgvector-compatible string: "[0.1, 0.2, ...]"
    """
    if not rows:
        return 0

    sql = """
        INSERT INTO agent_docs
            (module, page_id, title, url, content_chunk, chunk_index, embedding)
        VALUES
            (%(module)s, %(page_id)s, %(title)s, %(url)s,
             %(content_chunk)s, %(chunk_index)s, %(embedding)s::vector)
    """
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=100)
        conn.commit()
    logger.info("Inserted %d doc chunks for module '%s'.", len(rows), rows[0]["module"])
    return len(rows)


def search_docs(
    dsn: str,
    module: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Cosine-similarity search.

    Returns a list of dicts with keys:
        title, url, content_chunk, similarity
    sorted by descending similarity.
    """
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    sql = """
        SELECT
            title,
            url,
            content_chunk,
            1 - (embedding <=> %s::vector) AS similarity
        FROM agent_docs
        WHERE module = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with psycopg2.connect(dsn) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (embedding_str, module, embedding_str, top_k))
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_doc_stats(dsn: str) -> list[dict]:
    """Return chunk counts and last sync time grouped by module."""
    sql = """
        SELECT
            module,
            COUNT(*)          AS chunks,
            COUNT(DISTINCT page_id) AS pages,
            MAX(synced_at)    AS last_synced
        FROM agent_docs
        GROUP BY module
        ORDER BY module
    """
    with psycopg2.connect(dsn) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return [dict(r) for r in rows]
