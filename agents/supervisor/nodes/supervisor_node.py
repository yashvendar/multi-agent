"""
agents/supervisor/nodes/supervisor_node.py
==========================================
Supervisor routing node — follows the planner's execution plan step-by-step.
Falls back to LLM-based routing for unplanned / federated edge cases.
"""
from __future__ import annotations

import hashlib
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from config import settings
from models.schemas import AgentStepTrace, ToolCallTrace
from prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from tools.db_tools import _execute_readonly_query
from agents.supervisor.state import RouterDecision, SupervisorState

logger = logging.getLogger("agents.supervisor.node")


def make_supervisor_node(llm_with_structured_output):
    """
    Supervisor node factory.

    When an execution_plan is present (planner ran successfully), the supervisor
    executes steps from the plan without making any LLM routing calls.
    Once all steps are done → routes to 'summarise'.

    Falls back to LLM-based routing when no plan exists (federated / edge cases).
    """
    def supervisor_node(state: SupervisorState) -> dict:
        """Follow the execution plan and route to the next agent."""

        execution_plan: list[dict] = state.get("execution_plan", [])
        plan_step: int = state.get("plan_step", 0)
        routing_history: list[str] = state.get("routing_history", [])

        # ── Plan-following mode ──────────────────────────────────────────────
        if execution_plan and plan_step < len(execution_plan):
            current_step = execution_plan[plan_step]
            agent = current_step["agent"]
            goal = current_step["goal"]

            # Build agent_instruction that includes any data already collected.
            ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
            if ai_messages and plan_step > 0:
                prior_data = "\n".join(
                    f"[Step {i+1} result] {ai_messages[i].content[:400]}"
                    for i in range(min(len(ai_messages), plan_step))
                )
                instruction = f"{goal}\n\nData collected so far:\n{prior_data}"
            else:
                instruction = goal

            step_label = f"Step {plan_step+1}/{len(execution_plan)}: {goal[:60]}"
            reasoning = f"I'm executing the plan — {step_label}."
            logger.info(
                "Planner step %d/%d → agent='%s' goal='%s'",
                plan_step + 1, len(execution_plan), agent, goal,
            )

            trace_entry = AgentStepTrace(
                type="route", agent="supervisor", content=reasoning,
            ).model_dump(mode="json")

            new_messages: list[BaseMessage] = []
            if agent != "supervisor":
                new_messages = [HumanMessage(content=instruction)]

            return {
                "next_agent": agent,
                "supervisor_reasoning": reasoning,
                "reasoning_trace": [trace_entry],
                "messages": new_messages,
                "plan_step": plan_step + 1,
                "routing_history": [],
            }

        # ── All plan steps done → summarise ─────────────────────────────────
        if execution_plan and plan_step >= len(execution_plan):
            reasoning = "I've collected all the required data. Synthesising the final answer."
            trace_entry = AgentStepTrace(
                type="route", agent="supervisor", content=reasoning,
            ).model_dump(mode="json")
            return {
                "next_agent": "summarise",
                "supervisor_reasoning": reasoning,
                "reasoning_trace": [trace_entry],
                "messages": [],
                "routing_history": [],
            }

        # ── Fallback: no plan — LLM-based routing ───────────────────────────
        context_lines = []
        if routing_history:
            context_lines.append(
                f"\n[CONTEXT] Agent calls already made this turn: {', '.join(routing_history)}."
                " You MAY call the same agent again IF you have a different/enriched query."
                " You MUST NOT send the exact same query to the same agent again."
                " Set next=FINISH when you have all the data you need."
            )
            ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                summaries = [
                    f"  [{i+1}] {ai_messages[i].content[:300]}"
                    for i in range(min(len(ai_messages), len(routing_history)))
                ]
                context_lines.append("[DATA COLLECTED SO FAR]\n" + "\n".join(summaries))

        context_block = "\n".join(context_lines)
        system_content = SUPERVISOR_SYSTEM_PROMPT + ("\n" + context_block if context_block else "")

        messages = [SystemMessage(content=system_content), *state["messages"]]
        decision: RouterDecision = llm_with_structured_output.invoke(messages)
        next_agent = decision.next

        logger.info(
            "Supervisor LLM decision: next=%s routing_history=%s reasoning=%s",
            next_agent, routing_history, decision.reasoning,
        )

        trace_entry = AgentStepTrace(
            type="route", agent="supervisor", content=decision.reasoning,
        ).model_dump(mode="json")

        new_messages_fb: list[BaseMessage] = []
        trace_entries = [trace_entry]

        if decision.execute_federated_query:
            if not settings.federated_db_dsn:
                result_str = "Error: FEDERATED_DB_DSN not configured."
            else:
                try:
                    result_str = _execute_readonly_query(
                        settings.federated_db_dsn, decision.execute_federated_query, max_rows=500
                    )
                except Exception as e:
                    result_str = f"Error executing federated query: {e}"

            new_messages_fb = [
                AIMessage(content=f"Running federated query:\n```sql\n{decision.execute_federated_query}\n```"),
                HumanMessage(content=f"Federated Query Result:\n{result_str}"),
            ]
            tool_trace = ToolCallTrace(
                agent="supervisor", tool="federated_query_db", output=result_str[:800],
            )
            trace_entries.append(
                AgentStepTrace(
                    type="tool_result", agent="supervisor", tool_call=tool_trace,
                ).model_dump(mode="json")
            )
            logger.info("Supervisor executed federated query.")

        elif next_agent == "FINISH":
            if decision.direct_response:
                new_messages_fb = [AIMessage(content=decision.direct_response)]
            else:
                has_prior_ai = any(isinstance(m, AIMessage) for m in state.get("messages", []))
                if has_prior_ai:
                    logger.info("Supervisor returned FINISH with no response — auto-redirecting to summarise.")
                    next_agent = "summarise"
                else:
                    new_messages_fb = [AIMessage(content="Hello! How can I help you with your IoT data today?")]

        elif next_agent not in ["FINISH", "supervisor"] and decision.agent_instruction:
            new_messages_fb = [HumanMessage(content=decision.agent_instruction)]
            logger.info(
                "Supervisor injected instruction for %s: %.120s",
                next_agent, decision.agent_instruction,
            )

        new_routing_history = []
        if decision.execute_federated_query:
            query_hash = hashlib.md5(decision.execute_federated_query.encode()).hexdigest()[:10]
            new_routing_history = [f"supervisor::federated_query_{query_hash}"]

        return {
            "next_agent": next_agent,
            "supervisor_reasoning": decision.reasoning,
            "reasoning_trace": trace_entries,
            "messages": new_messages_fb,
            "routing_history": new_routing_history,
        }

    return supervisor_node


def route_after_supervisor(state: SupervisorState) -> str:
    """Route to the next agent or END based on the supervisor's decision."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    routing_history: list[str] = state.get("routing_history", [])
    logger.info(
        "Routing to '%s' | call history this turn: %s",
        next_agent, routing_history or "none",
    )
    return next_agent
