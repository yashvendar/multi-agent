"""
agents/supervisor/nodes/planner.py
===================================
Planner node — runs ONCE at START to decompose the user's question into an
ordered list of execution steps.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from models.schemas import AgentStepTrace
from agents.supervisor.state import PlannerDecision, SupervisorState

logger = logging.getLogger("agents.supervisor.planner")

_PLANNER_SYSTEM_PROMPT = """You are a planning agent for an industrial IoT assistant.

Your job is to analyse the user's question and create an ordered execution plan.

Available agents:
| Agent            | Handles |
|---|---|
| kpi_configurator | KPI definitions, formulas, calculated metric values, thresholds |
| data_explorer    | Raw IoT sensor / tag readings, time-series telemetry |
| amm              | Asset metadata, hierarchy, asset-to-sensor mapping |
| supervisor       | Cross-domain SQL via federated database (use only for massive JOINs) |

Planning rules:
- If the user is greeting you or asking a general question with no data → set is_direct=True, direct_response=<friendly reply>.
- If the question mentions a specific site, plant, or asset name → the FIRST step must always be an `amm` step to resolve the asset identifier.
- Each step must be self-contained and reference earlier step outputs by order number if needed.
- Keep the plan as short as possible. Avoid redundant steps.
"""


def make_planner_node(planner_llm):
    """
    Runs once at graph START. Decomposes the user question into an ordered plan.
    Direct answers (greetings/chitchat) are returned immediately without entering
    the supervisor loop.
    """
    def planner_node(state: SupervisorState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_msg = human_msgs[-1] if human_msgs else state["messages"][-1]

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            last_msg,
        ]
        plan: PlannerDecision = planner_llm.invoke(messages)

        logger.info(
            "Planner: is_direct=%s steps=%d reasoning=%s",
            plan.is_direct,
            len(plan.steps),
            plan.reasoning,
        )

        trace_entry = AgentStepTrace(
            type="route",
            agent="planner",
            content=plan.reasoning,
        ).model_dump(mode="json")

        if plan.is_direct:
            reply = plan.direct_response or "Hello! How can I help you with your IoT data today?"
            return {
                "messages": [AIMessage(content=reply)],
                "next_agent": "FINISH",
                "supervisor_reasoning": plan.reasoning,
                "reasoning_trace": [trace_entry],
                "execution_plan": [],
                "plan_step": 0,
            }

        steps_dicts = [s.model_dump() for s in plan.steps]
        return {
            "next_agent": "supervisor",
            "supervisor_reasoning": plan.reasoning,
            "reasoning_trace": [trace_entry],
            "execution_plan": steps_dicts,
            "plan_step": 0,
        }

    return planner_node


def route_after_planner(state: SupervisorState) -> str:
    """Direct FINISH for greetings; otherwise enter supervisor loop."""
    if state.get("next_agent") == "FINISH":
        return END
    return "supervisor"
