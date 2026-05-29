"""
agents/kpi_agent.py
===================
KPI Configurator sub-agent.

Responsibilities
----------------
- KPI definitions, formulas, thresholds, units
- Calculated / aggregated KPI values for assets and time ranges

Model: gemini-2.5-pro  (deep reasoning for formula analysis)

Note: Cross-domain queries (e.g. needing asset metadata alongside KPI data)
are handled by the Supervisor, which calls this agent and the AMM/Data Explorer
sequentially and synthesises the combined result.
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import settings
from prompts.agents import KPI_AGENT_SYSTEM_PROMPT
from tools.agent_tools import AgentRegistry
from tools.db_tools import make_db_tools
from tools.rag_tools import make_search_docs_tool

logger = logging.getLogger("agents.kpi")


def build_kpi_agent():
    """
    Construct and register the KPI Configurator agent.
    Must be called AFTER the other agents are already registered
    (or those agents will just fail gracefully via the depth guard).
    """
    # ── LLM ─────────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=settings.subagent_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        temperature=0,
        max_retries=2,
    )

    # ── Tools ────────────────────────────────────────────────────────────────
    db_tools = make_db_tools(
        dsn=settings.kpi_db_dsn,
        prefix="kpi",
        max_rows=settings.db_max_rows,
        schemas=settings.kpi_schemas_list,
    )
    rag_tool = make_search_docs_tool(module="kpi", prefix="kpi")

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = create_react_agent(
        model=llm,
        tools=[*db_tools, rag_tool],
        prompt=SystemMessage(content=KPI_AGENT_SYSTEM_PROMPT),
    )


    AgentRegistry.register("kpi_configurator", agent)
    logger.info("KPI Configurator agent built and registered.")
    return agent
