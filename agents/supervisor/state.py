"""
agents/supervisor/state.py
==========================
LangGraph state definition and all Pydantic schemas shared across nodes.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    # Populated by planner/supervisor, consumed by routing functions
    next_agent: str
    supervisor_reasoning: str
    # operator.add appends new steps rather than replacing
    reasoning_trace: Annotated[list[dict], operator.add]
    # Tracks "agent_name::query_hash" pairs for calls already made this turn
    routing_history: Annotated[list[str], operator.add]
    # Planner output — set once by planner_node, consumed step-by-step by supervisor
    execution_plan: list[dict]   # list of {"agent": str, "goal": str, "order": int}
    plan_step: int               # index of the next unexecuted step


# ─────────────────────────────────────────────────────────────────────────────
# Planner schemas
# ─────────────────────────────────────────────────────────────────────────────

class PlanStep(BaseModel):
    """A single step in the planner's execution plan."""
    order: int = Field(description="1-based execution order.")
    agent: Literal["kpi_configurator", "data_explorer", "amm", "supervisor"] = Field(
        description="Which sub-agent will execute this step ('supervisor' for federated SQL)."
    )
    goal: str = Field(
        description=(
            "One sentence describing exactly what data to retrieve in this step. "
            "May reference outputs of earlier steps by order number, e.g. "
            "'Fetch energy_efficiency KPI values for the asset_ids returned in step 1'."
        )
    )


class PlannerDecision(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the chosen plan.")
    is_direct: bool = Field(
        description="True if the question can be answered without any data lookup (greetings, general knowledge, etc.)."
    )
    direct_response: str | None = Field(
        default=None,
        description="Populated only when is_direct=True. The full user-facing answer.",
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of agent steps to execute. Empty when is_direct=True.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor routing schema (fallback / federated path)
# ─────────────────────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    reasoning: str = Field(
        description="One or two sentences explaining WHY you are routing to this agent or finishing."
    )
    next: Literal["kpi_configurator", "data_explorer", "amm", "supervisor", "summarise", "FINISH"] = Field(
        description=(
            "Which agent to invoke next, or "
            "'summarise' once all data is collected (triggers synthesis node), or "
            "'FINISH' only for direct replies that need no synthesis (greetings, etc.), or "
            "'supervisor' when running a federated query."
        )
    )
    agent_instruction: str | None = Field(
        default=None,
        description=(
            "Specific instruction to send to the next agent. "
            "REQUIRED when routing to an agent that has already been called this turn — "
            "give it a clear, targeted task based on data collected so far "
            "(e.g. 'Fetch KPI values for asset_id=42 for the past 7 days'). "
            "Leave None for the first call to an agent (it will use the user's original message)."
        ),
    )
    direct_response: str | None = Field(
        default=None,
        description="Only populated when next=FINISH. The supervisor's final synthesised answer.",
    )
    execute_federated_query: str | None = Field(
        default=None,
        description=(
            "A SQL query to execute against the federated database (which connects KPI, IOT, and AMM data via postgres_fdw). "
            "Use this ONLY when you need to join massive datasets across domains and have already obtained the schema/IDs "
            "from sub-agents. If you provide this, you MUST set next='supervisor' so you can see the query result next turn."
        ),
    )
