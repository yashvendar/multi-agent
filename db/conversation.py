"""
db/conversation.py
==================
Conversation history CRUD using the CONV_DB (read + write).

Tables created on first use:
  - conversations       (session metadata)
  - conversation_messages  (per-turn messages + serialised reasoning)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras

from config import settings
from models.schemas import ConversationHistory, ConversationMessage, ReasoningTrace

logger = logging.getLogger("db.conversation")

# ─────────────────────────────────────────────────────────────────────────────
# DDL — run once on startup
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id   TEXT        PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id           TEXT        PRIMARY KEY,
    session_id   TEXT        NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role         TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT        NOT NULL,
    reasoning    JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conv_messages_session
    ON conversation_messages (session_id, created_at);
"""


def _get_conn() -> psycopg2.extensions.connection:
    """Open a new connection to the conversation DB."""
    return psycopg2.connect(settings.conv_db_dsn)


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES_SQL)
        conn.commit()
    logger.info("Conversation DB tables verified / created.")


# ─────────────────────────────────────────────────────────────────────────────
# CRUD operations
# ─────────────────────────────────────────────────────────────────────────────

def ensure_session(session_id: str) -> None:
    """Insert a session row if it doesn't already exist."""
    sql = """
        INSERT INTO conversations (session_id)
        VALUES (%s)
        ON CONFLICT (session_id) DO UPDATE
            SET updated_at = NOW();
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
        conn.commit()


def save_message(msg: ConversationMessage) -> None:
    """Persist a single message (user or assistant) to the conversation DB."""
    ensure_session(msg.session_id)

    reasoning_json: str | None = None
    if msg.reasoning is not None:
        reasoning_json = msg.reasoning.model_dump_json()

    sql = """
        INSERT INTO conversation_messages (id, session_id, role, content, reasoning, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    msg.id,
                    msg.session_id,
                    msg.role,
                    msg.content,
                    reasoning_json,
                    msg.created_at,
                ),
            )
        conn.commit()


def get_history(session_id: str, limit: int = 50) -> ConversationHistory:
    """Fetch the last *limit* messages for a session."""
    sql = """
        SELECT id, session_id, role, content, reasoning, created_at
        FROM conversation_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        LIMIT %s;
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (session_id, limit))
            rows = cur.fetchall()

    messages: list[ConversationMessage] = []
    for row in rows:
        reasoning: ReasoningTrace | None = None
        if row["reasoning"]:
            data: Any = row["reasoning"]
            if isinstance(data, str):
                data = json.loads(data)
            reasoning = ReasoningTrace.model_validate(data)

        messages.append(
            ConversationMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                reasoning=reasoning,
                created_at=row["created_at"],
            )
        )

    # Get session timestamps
    session_sql = "SELECT created_at, updated_at FROM conversations WHERE session_id = %s;"
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(session_sql, (session_id,))
            session_row = cur.fetchone()

    return ConversationHistory(
        session_id=session_id,
        messages=messages,
        created_at=session_row["created_at"] if session_row else None,
        updated_at=session_row["updated_at"] if session_row else None,
    )


def get_recent_messages_for_context(
    session_id: str, limit: int = 10
) -> list[dict]:
    """
    Return the last *limit* messages as a list of dicts suitable for
    passing to LangChain as chat history.

    Returns: [{"role": "user"|"assistant", "content": str}, ...]
    """
    history = get_history(session_id, limit=limit)
    return [{"role": m.role, "content": m.content} for m in history.messages]
