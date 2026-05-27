"""
agents/kpi_agent.py
===================
KPI Configurator sub-agent.

Responsibilities
----------------
- KPI definitions, formulas, thresholds, units
- Calculated / aggregated KPI values for assets and time ranges
- Cross-calls to Data Explorer (raw data) and AMM (asset metadata)

Model: gemini-2.5-pro  (deep reasoning for formula analysis)
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.prebuilt import create_react_agent

from config import settings
from prompts.agents import KPI_AGENT_SYSTEM_PROMPT
from tools.agent_tools import AgentRegistry, make_cross_agent_tools
from tools.db_tools import make_db_tools

logger = logging.getLogger("agents.kpi")


def build_kpi_agent():
    """
    Construct and register the KPI Configurator agent.
    Must be called AFTER the other agents are already registered
    (or those agents will just fail gracefully via the depth guard).
    """
    # ── LLM ─────────────────────────────────────────────────────────────────
    llm = ChatVertexAI(
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
    cross_tools = make_cross_agent_tools(exclude=["kpi_configurator"])

    all_tools = db_tools + cross_tools

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SystemMessage(content=KPI_AGENT_SYSTEM_PROMPT),
    )

    AgentRegistry.register("kpi_configurator", agent)
    logger.info("KPI Configurator agent built and registered.")
    return agent
