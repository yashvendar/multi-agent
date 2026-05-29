"""
agents/amm_agent.py
===================
AMM (Asset Model Manager) sub-agent.

Responsibilities
----------------
- Asset hierarchy and parent/child relationships
- Asset metadata: type, location, manufacturer, install date, status
- Mapping assets to sensors/tags

Model: gemini-2.5-pro

Note: Cross-domain queries are handled by the Supervisor, which chains
agents sequentially and synthesises the combined result.
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import settings
from prompts.agents import AMM_AGENT_SYSTEM_PROMPT
from tools.agent_tools import AgentRegistry
from tools.db_tools import make_db_tools
from tools.rag_tools import make_search_docs_tool

logger = logging.getLogger("agents.amm")


def build_amm_agent():
    """Construct and register the AMM agent."""
    llm = ChatGoogleGenerativeAI(
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
    rag_tool = make_search_docs_tool(module="amm", prefix="asset")

    agent = create_react_agent(
        model=llm,
        tools=[*db_tools, rag_tool],
        prompt=SystemMessage(content=AMM_AGENT_SYSTEM_PROMPT),
    )

    AgentRegistry.register("amm", agent)
    logger.info("AMM agent built and registered.")
    return agent
