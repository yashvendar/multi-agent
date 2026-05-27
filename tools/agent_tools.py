"""
tools/agent_tools.py
====================
Cross-agent call tools and the AgentRegistry singleton.

Pattern
-------
1. Each sub-agent registers itself via ``AgentRegistry.register(name, agent)``.
2. Cross-agent tools look up the registry lazily at call-time, so registration
   order doesn't matter and there are no circular imports.
3. A ``call_depth`` context-var prevents infinite cross-agent loops.

Usage
-----
    from tools.agent_tools import make_cross_agent_tools

    # Pass all tools *except* the agent itself (avoid self-calling)
    kpi_cross_tools = make_cross_agent_tools(exclude=["kpi_configurator"])
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import HumanMessage

from config import settings

logger = logging.getLogger("tools.agent_tools")

# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Singleton registry mapping agent names to their compiled LangGraph graphs.
    Agents register themselves after creation; cross-agent tools look up here.
    """
    _agents: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, agent: Any) -> None:
        cls._agents[name] = agent
        logger.info("AgentRegistry: registered '%s'", name)

    @classmethod
    def get(cls, name: str) -> Any:
        agent = cls._agents.get(name)
        if agent is None:
            raise RuntimeError(
                f"Agent '{name}' is not registered. "
                f"Registered agents: {list(cls._agents)}"
            )
        return agent

    @classmethod
    def all_names(cls) -> list[str]:
        return list(cls._agents.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Call-depth guard (prevents infinite agent→agent recursion)
# ─────────────────────────────────────────────────────────────────────────────

_call_depth: ContextVar[int] = ContextVar("_call_depth", default=0)


def _invoke_agent(agent_name: str, query: str) -> str:
    """
    Invoke a registered sub-agent with *query* and return its final answer.
    Enforces ``settings.max_agent_call_depth`` to prevent infinite loops.
    """
    depth = _call_depth.get()
    if depth >= settings.max_agent_call_depth:
        return (
            f"[Cross-agent call to '{agent_name}' blocked] "
            f"Maximum call depth ({settings.max_agent_call_depth}) reached. "
            "Please answer based on what you already know."
        )

    token = _call_depth.set(depth + 1)
    try:
        agent = AgentRegistry.get(agent_name)
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 10},
        )
        # The last message is the agent's final answer
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return "No response from agent."
    except Exception as exc:
        logger.error("Cross-agent call to '%s' failed: %s", agent_name, exc)
        return f"[Error calling {agent_name}]: {exc}"
    finally:
        _call_depth.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-agent tool factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cross_agent_tools(exclude: list[str] | None = None) -> list:
    """
    Return LangChain tools for calling peer sub-agents.

    Parameters
    ----------
    exclude: Agent names to omit (pass the calling agent's own name to
             prevent self-calls).

    Returns
    -------
    list[BaseTool]
    """
    from langchain_core.tools import tool

    excluded = set(exclude or [])
    result_tools = []

    # ── KPI Configurator ────────────────────────────────────────────────────
    if "kpi_configurator" not in excluded:
        @tool
        def call_kpi_configurator(query: str) -> str:
            """
            Call the KPI Configurator agent when you need:
            - Definitions or formulas of KPIs
            - Calculated / aggregated KPI values for assets or time ranges
            - KPI configuration details (thresholds, units, calculation rules)

            Args:
                query: A natural-language question or data request for the
                       KPI Configurator.

            Returns:
                The KPI Configurator agent's answer as a string.
            """
            logger.info("[cross-agent] → kpi_configurator: %.100s", query)
            return _invoke_agent("kpi_configurator", query)

        result_tools.append(call_kpi_configurator)

    # ── Data Explorer ────────────────────────────────────────────────────────
    if "data_explorer" not in excluded:
        @tool
        def call_data_explorer(query: str) -> str:
            """
            Call the Data Explorer agent when you need:
            - Raw IoT sensor / tag values for a device or time range
            - Telemetry data points (temperature, pressure, vibration, etc.)
            - Recent or historical raw measurements from field devices

            Args:
                query: A natural-language question or data request for the
                       Data Explorer.

            Returns:
                The Data Explorer agent's answer as a string.
            """
            logger.info("[cross-agent] → data_explorer: %.100s", query)
            return _invoke_agent("data_explorer", query)

        result_tools.append(call_data_explorer)

    # ── AMM (Asset Model Manager) ────────────────────────────────────────────
    if "amm" not in excluded:
        @tool
        def call_amm(query: str) -> str:
            """
            Call the AMM (Asset Model Manager) agent when you need:
            - Asset hierarchy, parent/child relationships
            - Asset metadata (location, type, manufacturer, install date)
            - Which sensors / tags are associated with an asset
            - Asset operational status or configuration

            Args:
                query: A natural-language question or data request for the
                       AMM agent.

            Returns:
                The AMM agent's answer as a string.
            """
            logger.info("[cross-agent] → amm: %.100s", query)
            return _invoke_agent("amm", query)

        result_tools.append(call_amm)

    return result_tools
