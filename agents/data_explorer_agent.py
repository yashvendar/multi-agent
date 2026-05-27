"""
agents/data_explorer_agent.py
==============================
Data Explorer sub-agent.

Responsibilities
----------------
- Raw IoT sensor / tag values from field devices
- Time-series telemetry: latest readings, historical trends, gap detection

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
from prompts.agents import DATA_EXPLORER_SYSTEM_PROMPT
from tools.agent_tools import AgentRegistry
from tools.db_tools import make_db_tools

logger = logging.getLogger("agents.data_explorer")


def build_data_explorer_agent():
    """Construct and register the Data Explorer agent."""
    llm = ChatGoogleGenerativeAI(
        model=settings.subagent_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        temperature=0,
        max_retries=2,
    )

    db_tools = make_db_tools(
        dsn=settings.iot_db_dsn,
        prefix="iot",
        max_rows=settings.db_max_rows,
        schemas=settings.iot_schemas_list,
    )

    agent = create_react_agent(
        model=llm,
        tools=db_tools,
        prompt=SystemMessage(content=DATA_EXPLORER_SYSTEM_PROMPT),
    )

    AgentRegistry.register("data_explorer", agent)
    logger.info("Data Explorer agent built and registered.")
    return agent
