"""
agents/supervisor/graph.py
===========================
Builds and compiles the supervisor LangGraph.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from config import settings
from agents.supervisor.state import PlannerDecision, RouterDecision, SupervisorState
from agents.supervisor.nodes.planner import make_planner_node, route_after_planner
from agents.supervisor.nodes.supervisor_node import make_supervisor_node, route_after_supervisor
from agents.supervisor.nodes.sub_agent import make_sub_agent_node
from agents.supervisor.nodes.summarise import make_summarise_node

logger = logging.getLogger("agents.supervisor.graph")


def _build_supervisor_llm() -> Any:
    return ChatGoogleGenerativeAI(
        model=settings.supervisor_model,
        temperature=0,
        max_retries=2,
    ).with_structured_output(RouterDecision)


def _build_planner_llm() -> Any:
    return ChatGoogleGenerativeAI(
        model=settings.supervisor_model,
        temperature=0,
        max_retries=2,
    ).with_structured_output(PlannerDecision)


def _build_synthesis_llm() -> Any:
    """Plain (non-structured) LLM used by the summarise node."""
    return ChatGoogleGenerativeAI(
        model=settings.supervisor_model,
        temperature=0,
        max_retries=2,
    )


def build_supervisor_graph():
    """
    Build and compile the supervisor LangGraph.
    All sub-agents must be registered in AgentRegistry before calling this.

    Graph topology:
        START → planner ──(is_direct)──────────────────────────────→ END
                        └──(has steps)──→ supervisor → kpi/data/amm → supervisor
                                                     └──(all done)──→ summarise → END
    """
    supervisor_llm = _build_supervisor_llm()
    planner_llm = _build_planner_llm()
    synthesis_llm = _build_synthesis_llm()

    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("planner", make_planner_node(planner_llm))
    graph.add_node("supervisor", make_supervisor_node(supervisor_llm))
    graph.add_node("kpi_configurator", make_sub_agent_node("kpi_configurator"))
    graph.add_node("data_explorer", make_sub_agent_node("data_explorer"))
    graph.add_node("amm", make_sub_agent_node("amm"))
    graph.add_node("summarise", make_summarise_node(synthesis_llm))

    # Edges
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"supervisor": "supervisor", END: END},
    )
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "kpi_configurator": "kpi_configurator",
            "data_explorer": "data_explorer",
            "amm": "amm",
            "supervisor": "supervisor",
            "summarise": "summarise",
            END: END,
        },
    )
    # Sub-agents always return to supervisor for the next routing decision.
    graph.add_edge("kpi_configurator", "supervisor")
    graph.add_edge("data_explorer", "supervisor")
    graph.add_edge("amm", "supervisor")
    # Summarise goes straight to END — no further routing.
    graph.add_edge("summarise", END)

    compiled = graph.compile()
    logger.info("Supervisor graph compiled.")
    return compiled
