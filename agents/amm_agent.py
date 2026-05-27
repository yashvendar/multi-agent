"""
agents/amm_agent.py
===================
AMM (Asset Model Manager) sub-agent.

Responsibilities
----------------
- Asset hierarchy and parent/child relationships
- Asset metadata: type, location, manufacturer, install date, status
- Mapping assets to sensors/tags
- Cross-calls to KPI Configurator and Data Explorer as needed

Model: gemini-2.5-pro
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.prebuilt import create_react_agent

from config import settings
from prompts.agents import AMM_AGENT_SYSTEM_PROMPT
from tools.agent_tools import AgentRegistry, make_cross_agent_tools
from tools.db_tools import make_db_tools

logger = logging.getLogger("agents.amm")


def build_amm_agent():
    """Construct and register the AMM agent."""
    llm = ChatVertexAI(
        model=settings.subagent_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        temperature=0,
        max_retries=2,
    )

    db_tools = make_db_tools(
        dsn=settings.asset_db_dsn,
        prefix="asset",
        max_rows=settings.db_max_rows,
        schemas=settings.asset_schemas_list,
    )
    cross_tools = make_cross_agent_tools(exclude=["amm"])

    agent = create_react_agent(
        model=llm,
        tools=db_tools + cross_tools,
        prompt=SystemMessage(content=AMM_AGENT_SYSTEM_PROMPT),
    )

    AgentRegistry.register("amm", agent)
    logger.info("AMM agent built and registered.")
    return agent
