"""
agents/supervisor/nodes/sub_agent.py
=====================================
Factory for sub-agent nodes (kpi_configurator, data_explorer, amm).
Each node invokes the registered LangGraph sub-agent and extracts a
structured tool-call trace from the result messages.
"""
from __future__ import annotations

import hashlib
import logging

from langchain_core.messages import AIMessage, HumanMessage

from models.schemas import AgentStepTrace, ToolCallTrace
from tools.agent_tools import AgentRegistry
from agents.supervisor.state import SupervisorState

logger = logging.getLogger("agents.supervisor.sub_agent")


def make_sub_agent_node(agent_name: str):
    """
    Create a graph node that invokes the registered sub-agent and
    appends its tool-call trace to the state.
    Sub-agents do NOT set next_agent — the supervisor owns all routing.
    """
    def sub_agent_node(state: SupervisorState) -> dict:
        agent = AgentRegistry.get(agent_name)

        # Pass only the last user message so the sub-agent isn't confused
        # by supervisor routing messages in the full history.
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user_msg = user_messages[-1] if user_messages else state["messages"][-1]

        result = agent.invoke(
            {"messages": [last_user_msg]},
            config={"recursion_limit": 15},
        )

        sub_messages = result.get("messages", [])

        # ── Build tool_call_id → tool_name map ──────────────────────────────
        # ToolMessage only carries tool_call_id, not the name reliably.
        tool_id_to_name: dict[str, str] = {}
        for msg in sub_messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []) or []:
                    call_id = tc.get("id") or tc.get("tool_call_id", "")
                    name = tc.get("name", "unknown")
                    if call_id:
                        tool_id_to_name[call_id] = name

        # ── Extract reasoning trace ──────────────────────────────────────────
        trace_steps: list[dict] = []
        pending_reasoning: str | None = None

        for msg in sub_messages:
            if isinstance(msg, AIMessage):
                text = msg.content if isinstance(msg.content, str) else ""
                if text.strip():
                    pending_reasoning = text.strip()

                for tc in getattr(msg, "tool_calls", []) or []:
                    tool_trace = ToolCallTrace(
                        agent=agent_name,
                        tool=tc.get("name", "unknown"),
                        reasoning=pending_reasoning,
                        input=tc.get("args"),
                    )
                    trace_steps.append(
                        AgentStepTrace(
                            type="tool_call",
                            agent=agent_name,
                            tool_call=tool_trace,
                        ).model_dump(mode="json")
                    )
                    pending_reasoning = None  # consumed

            else:
                # ToolMessage — resolve name via id→name map
                call_id = getattr(msg, "tool_call_id", "") or ""
                tool_name = (
                    tool_id_to_name.get(call_id)
                    or getattr(msg, "name", None)
                    or "unknown_tool"
                )
                output_str = str(getattr(msg, "content", ""))[:800]
                tool_trace = ToolCallTrace(
                    agent=agent_name, tool=tool_name, output=output_str,
                )
                trace_steps.append(
                    AgentStepTrace(
                        type="tool_result", agent=agent_name, tool_call=tool_trace,
                    ).model_dump(mode="json")
                )

        # ── Final answer ─────────────────────────────────────────────────────
        final_answer = ""
        if sub_messages:
            last = sub_messages[-1]
            final_answer = getattr(last, "content", "")
            if isinstance(final_answer, list):
                final_answer = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in final_answer
                )

        trace_steps.append(
            AgentStepTrace(
                type="answer", agent=agent_name, content=final_answer,
            ).model_dump(mode="json")
        )

        # ── Routing-history fingerprint ──────────────────────────────────────
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        query_content = human_msgs[-1].content if human_msgs else ""
        query_hash = hashlib.md5(query_content.encode()).hexdigest()[:10]
        route_key = f"{agent_name}::{query_hash}"

        return {
            "messages": [AIMessage(content=final_answer)],
            "reasoning_trace": trace_steps,
            # Sub-agents do NOT set next_agent — supervisor owns all routing.
            "routing_history": [route_key],
        }

    return sub_agent_node
