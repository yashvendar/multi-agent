"""
tools/db_tools.py
=================
Factory that creates LangChain @tool functions scoped to a specific
PostgreSQL DSN. All tools are read-only by design.

Usage
-----
    from tools.db_tools import make_db_tools

    kpi_tools = make_db_tools(
        dsn=settings.kpi_db_dsn,
        prefix="kpi",         # tool names: kpi_query_db, kpi_get_schema
    )
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger("tools.db_tools")

# ─────────────────────────────────────────────────────────────────────────────
# Safety guard — block write statements on read-only connections
# ─────────────────────────────────────────────────────────────────────────────

_WRITE_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|REPLACE)\b",
    re.IGNORECASE,
)


def _is_write_query(sql: str) -> bool:
    return bool(_WRITE_PATTERN.match(sql.strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Core execution helpers (pure functions, no LangChain dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _execute_readonly_query(
    dsn: str,
    query: str,
    max_rows: int = 500,
) -> str:
    """
    Execute a read-only SQL query against *dsn* and return JSON results.
    Raises ValueError for write statements.
    """
    if _is_write_query(query):
        raise ValueError(
            "Write operations (INSERT/UPDATE/DELETE/DDL) are not allowed on "
            "this read-only connection."
        )

    try:
        with psycopg2.connect(dsn) as conn:
            # Enforce read-only at the session level
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchmany(max_rows)
                col_names = [desc.name for desc in cur.description] if cur.description else []

        return json.dumps(
            {
                "columns": col_names,
                "rows": [dict(r) for r in rows],
                "row_count": len(rows),
                "truncated": len(rows) == max_rows,
            },
            default=str,  # handles datetime, Decimal, etc.
        )
    except psycopg2.Error as exc:
        logger.error("DB query error: %s", exc)
        return json.dumps({"error": str(exc)})


def _get_schema(dsn: str, schemas: list[str] | None = None) -> str:
    """
    Return a compact schema summary (tables, columns, types) as JSON.
    Excludes system schemas and sub-partition tables.
    If 'schemas' is provided, only extracts from those specific schemas.
    """
    query = """
        SELECT
            n.nspname                             AS schema,
            c.relname                             AS table,
            a.attname                             AS column,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
            NOT a.attnotnull                      AS nullable,
            COALESCE(pk.is_pk, false)             AS primary_key
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        LEFT JOIN (
            SELECT conrelid, unnest(conkey) AS attnum, true AS is_pk
            FROM pg_catalog.pg_constraint
            WHERE contype = 'p'
        ) pk ON pk.conrelid = c.oid AND pk.attnum = a.attnum
        WHERE
            c.relkind IN ('r', 'p', 'f', 'v', 'm')  -- tables, partitioned, foreign, views, mat-views
            AND a.attnum > 0
            AND NOT a.attisdropped
            AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            AND c.relispartition = false   -- exclude sub-partitions
    """
    
    if schemas:
        # Create placeholders for the IN clause
        placeholders = ', '.join(['%s'] * len(schemas))
        query += f" AND n.nspname IN ({placeholders})"
        
    query += "\n        ORDER BY n.nspname, c.relname, a.attnum;"

    try:
        with psycopg2.connect(dsn) as conn:
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if schemas:
                    cur.execute(query, tuple(schemas))
                else:
                    cur.execute(query)
                rows = cur.fetchall()

        # Group into schema → table → columns
        schema_map: dict = {}
        for row in rows:
            s, t = row["schema"], row["table"]
            schema_map.setdefault(s, {}).setdefault(t, []).append(
                {
                    "column": row["column"],
                    "type": row["data_type"],
                    "nullable": row["nullable"],
                    "pk": row["primary_key"],
                }
            )

        return json.dumps(schema_map, default=str)
    except psycopg2.Error as exc:
        logger.error("Schema extraction error: %s", exc)
        return json.dumps({"error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_db_tools(
    dsn: str,
    prefix: str,
    max_rows: int = 500,
    schemas: list[str] | None = None,
) -> list:
    """
    Create a pair of LangChain tools scoped to *dsn* (read-only).

    Parameters
    ----------
    dsn:      PostgreSQL connection string.
    prefix:   Short identifier for the DB (e.g. "kpi", "iot", "asset").
              Used as the tool name prefix.
    max_rows: Maximum rows any single query may return.
    schemas:  Optional list of specific schemas to extract. If None, extracts all non-system schemas.

    Returns
    -------
    list[BaseTool]  — [query_tool, schema_tool]
    """
    from langchain_core.tools import tool

    @tool(name_or_callable=f"{prefix}_query_db")
    def query_db(sql: str) -> str:
        f"""
        Execute a read-only SQL query against the {prefix.upper()} database.

        Use this tool to retrieve data. Write operations (INSERT, UPDATE,
        DELETE, DDL) are blocked at the connection level.

        Args:
            sql: A valid SELECT (or read-only) SQL statement.

        Returns:
            JSON string with keys: columns, rows, row_count, truncated.
        """
        logger.info("[%s] query_db called: %.120s", prefix, sql)
        return _execute_readonly_query(dsn, sql, max_rows)

    @tool(name_or_callable=f"{prefix}_get_schema")
    def get_schema() -> str:
        f"""
        Return the full database schema for the {prefix.upper()} database.

        Returns a JSON object mapping schema → table → list of column definitions
        (name, data type, nullable, primary key flag).

        Use this BEFORE writing queries so you know the correct table and
        column names.

        Returns:
            JSON string with the full schema metadata.
        """
        logger.info("[%s] get_schema called", prefix)
        return _get_schema(dsn, schemas)

    return [query_db, get_schema]
