"""
agents/supervisor.py
====================
Supervisor LangGraph — routes user queries to the appropriate sub-agent
and streams reasoning + tool traces back to the API layer.

Graph structure
---------------
  START → supervisor_node → {kpi_node | data_explorer_node | amm_node | END}
                ↑_____________________|  (after sub-agent responds, supervisor
                                          decides to FINISH or route again)

Model: gemini-2.0-flash  (fast routing, low latency)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, AsyncIterator, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config import settings
from models.schemas import (
    AgentStepTrace,
    ContentFormat,
    ReasoningTrace,
    SSEAnswerEvent,
    SSEErrorEvent,
    SSERouteEvent,
    SSETokenEvent,
    SSEToolCallEvent,
    SSEToolResultEvent,
    ToolCallTrace,
)
from prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from tools.agent_tools import AgentRegistry

logger = logging.getLogger("agents.supervisor")


# ─────────────────────────────────────────────────────────────────────────────
# Content-format detection
# ─────────────────────────────────────────────────────────────────────────────

_MD_PATTERNS = (
    r"^#{1,6} ",        # ATX headings
    r"\*\*.+?\*\*",     # bold
    r"^\s*[-*+] ",      # unordered list
    r"^\s*\d+\. ",      # ordered list
    r"```",             # code fence
    r"\|.+\|",          # table row
    r"\[.+?\]\(.+?\)",  # link
)
import re as _re
_MD_RE = _re.compile("|".join(_MD_PATTERNS), _re.MULTILINE)

# HTML detection — matches an opening tag or <!DOCTYPE at the start of content
_HTML_RE = _re.compile(
    r"^\s*(?:<!DOCTYPE\s+html|<html|<head|<body|<div|<p|<span|<table|<ul|<ol|<h[1-6]|<article|<section|<main)",
    _re.IGNORECASE,
)


def detect_content_format(
    content: str,
    preferred: ContentFormat | None = None,
) -> ContentFormat:
    """
    Determine how a content string should be rendered.

    Priority
    --------
    1. ``preferred`` — explicit client hint always wins.
    2. Valid JSON object/array → ``"json"``.
    3. Starts with an HTML tag / DOCTYPE → ``"html"``.
    4. Contains Markdown syntax → ``"markdown"``.
    5. Fallback → ``"text"``.
    """
    if preferred is not None:
        return preferred

    stripped = content.strip()

    # JSON detection
    if stripped and stripped[0] in ("{", "["):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass

    # HTML detection (before markdown — HTML is unambiguous)
    if _HTML_RE.match(stripped):
        return "html"

    # Markdown detection
    if _MD_RE.search(stripped):
        return "markdown"

    return "text"



# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    # Populated by supervisor_node, consumed by sub-agent nodes
    next_agent: str
    supervisor_reasoning: str
    # Accumulated reasoning trace for the current turn
    reasoning_trace: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor structured output schema
# ─────────────────────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    reasoning: str = Field(
        description="One or two sentences explaining WHY you are routing to this agent or finishing."
    )
    next: Literal["kpi_configurator", "data_explorer", "amm", "FINISH"] = Field(
        description="Which agent to invoke next, or FINISH if the response is ready."
    )
    direct_response: str | None = Field(
        default=None,
        description="Only populated when next=FINISH. The supervisor's direct answer.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM — gemini-flash for fast routing
# ─────────────────────────────────────────────────────────────────────────────

def _build_supervisor_llm() -> Any:
    return ChatVertexAI(
        model=settings.supervisor_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        temperature=0,
        max_retries=2,
    ).with_structured_output(RouterDecision)


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

def make_supervisor_node(llm_with_structured_output):
    def supervisor_node(state: SupervisorState) -> dict:
        """Analyse the conversation and decide routing."""
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            *state["messages"],
        ]
        decision: RouterDecision = llm_with_structured_output.invoke(messages)

        logger.info(
            "Supervisor decision: next=%s reasoning=%s",
            decision.next,
            decision.reasoning,
        )

        trace_entry = AgentStepTrace(
            type="route",
            agent="supervisor",
            content=decision.reasoning,
        ).model_dump(mode="json")

        new_messages = []
        if decision.next == "FINISH" and decision.direct_response:
            new_messages = [AIMessage(content=decision.direct_response)]

        return {
            "next_agent": decision.next,
            "supervisor_reasoning": decision.reasoning,
            "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
            "messages": new_messages,
        }

    return supervisor_node


def make_sub_agent_node(agent_name: str):
    """
    Create a graph node that invokes the registered sub-agent and
    appends its tool-call trace to the state.
    """
    def sub_agent_node(state: SupervisorState) -> dict:
        agent = AgentRegistry.get(agent_name)

        # Pass only the last user message to the sub-agent (not full history)
        # so the sub-agent doesn't get confused by supervisor routing messages
        user_messages = [
            m for m in state["messages"] if isinstance(m, HumanMessage)
        ]
        last_user_msg = user_messages[-1] if user_messages else state["messages"][-1]

        result = agent.invoke(
            {"messages": [last_user_msg]},
            config={"recursion_limit": 15},
        )

        sub_messages: list[BaseMessage] = result.get("messages", [])

        # ── Extract reasoning trace from sub-agent messages ──────────────────
        trace_steps: list[dict] = []
        pending_reasoning: str | None = None

        for msg in sub_messages:
            if isinstance(msg, AIMessage):
                # Text content before tool calls = reasoning
                text = msg.content if isinstance(msg.content, str) else ""
                if text.strip():
                    pending_reasoning = text.strip()

                # Tool calls made by this message
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
                # ToolMessage — the result of a tool call
                tool_name = getattr(msg, "name", "unknown_tool")
                output_str = str(getattr(msg, "content", ""))[:800]
                tool_trace = ToolCallTrace(
                    agent=agent_name,
                    tool=tool_name,
                    output=output_str,
                )
                trace_steps.append(
                    AgentStepTrace(
                        type="tool_result",
                        agent=agent_name,
                        tool_call=tool_trace,
                    ).model_dump(mode="json")
                )

        # Final answer from the sub-agent
        final_answer = ""
        if sub_messages:
            last = sub_messages[-1]
            final_answer = getattr(last, "content", "")
            if isinstance(final_answer, list):
                # Gemini sometimes returns list of content parts
                final_answer = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in final_answer
                )

        trace_steps.append(
            AgentStepTrace(
                type="answer",
                agent=agent_name,
                content=final_answer,
            ).model_dump(mode="json")
        )

        return {
            "messages": [AIMessage(content=final_answer)],
            "reasoning_trace": state.get("reasoning_trace", []) + trace_steps,
            "next_agent": "FINISH",
        }

    return sub_agent_node


# ─────────────────────────────────────────────────────────────────────────────
# Routing function
# ─────────────────────────────────────────────────────────────────────────────

def route_after_supervisor(state: SupervisorState) -> str:
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent  # node name matches agent name


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_supervisor_graph():
    """
    Build and compile the supervisor LangGraph.
    All sub-agents must be registered in AgentRegistry before calling this.
    """
    supervisor_llm = _build_supervisor_llm()

    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("supervisor", make_supervisor_node(supervisor_llm))
    graph.add_node("kpi_configurator", make_sub_agent_node("kpi_configurator"))
    graph.add_node("data_explorer", make_sub_agent_node("data_explorer"))
    graph.add_node("amm", make_sub_agent_node("amm"))

    # Edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "kpi_configurator": "kpi_configurator",
            "data_explorer": "data_explorer",
            "amm": "amm",
            END: END,
        },
    )
    # After any sub-agent finishes, go back to supervisor to decide FINISH/re-route
    graph.add_edge("kpi_configurator", "supervisor")
    graph.add_edge("data_explorer", "supervisor")
    graph.add_edge("amm", "supervisor")

    compiled = graph.compile()
    logger.info("Supervisor graph compiled.")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# Streaming helper — yield SSE-formatted events from graph execution
# ─────────────────────────────────────────────────────────────────────────────

async def stream_graph_events(
    compiled_graph,
    user_message: str,
    session_id: str,
    history: list[dict] | None = None,
    preferred_format: ContentFormat | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted JSON lines.

    Each line is a JSON object with a "type" field:
      route | tool_call | tool_result | token | answer | error

    Parameters
    ----------
    preferred_format:
        Client-requested content format (from ChatRequest.response_format).
        Passed through to every ``answer`` event's ``content_format`` field.
        When None, the format is auto-detected per-event.
    """
    t_start = time.time()

    # Build initial messages (history + current)
    init_messages: list[BaseMessage] = []
    for h in (history or []):
        if h["role"] == "user":
            init_messages.append(HumanMessage(content=h["content"]))
        else:
            init_messages.append(AIMessage(content=h["content"]))
    init_messages.append(HumanMessage(content=user_message))

    input_state: SupervisorState = {
        "messages": init_messages,
        "session_id": session_id,
        "next_agent": "",
        "supervisor_reasoning": "",
        "reasoning_trace": [],
    }

    final_answer = ""

    try:
        # Use stream_mode="updates" to get each node's output as it completes
        async for chunk in compiled_graph.astream(
            input_state,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                if node_name == "supervisor":
                    reasoning = node_output.get("supervisor_reasoning", "")
                    next_agent = node_output.get("next_agent", "FINISH")

                    if next_agent != "FINISH":
                        event = SSERouteEvent(
                            agent=next_agent,
                            reasoning=reasoning,
                        )
                        yield f"data: {event.model_dump_json()}\n\n"
                    else:
                        # FINISH with direct response
                        msgs = node_output.get("messages", [])
                        if msgs:
                            final_answer = getattr(msgs[-1], "content", "")
                            event = SSEAnswerEvent(
                                content=final_answer,
                                session_id=session_id,
                                content_format=detect_content_format(
                                    final_answer, preferred_format
                                ),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"

                else:
                    # Sub-agent node output — emit tool call trace events
                    trace_steps: list[dict] = node_output.get("reasoning_trace", [])
                    for step in trace_steps:
                        step_type = step.get("type")

                        if step_type == "tool_call":
                            tc = step.get("tool_call", {})
                            event = SSEToolCallEvent(
                                agent=step.get("agent", node_name),
                                tool=tc.get("tool", "unknown"),
                                reasoning=tc.get("reasoning"),
                                input=tc.get("input"),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"

                        elif step_type == "tool_result":
                            tc = step.get("tool_call", {})
                            raw_output = tc.get("output", "")
                            event = SSEToolResultEvent(
                                agent=step.get("agent", node_name),
                                tool=tc.get("tool", "unknown"),
                                output=raw_output,
                                # DB tools always emit JSON; cross-agent tool
                                # results may be markdown or plain text
                                content_format=detect_content_format(raw_output),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"

                        elif step_type == "answer":
                            final_answer = step.get("content", "")
                            event = SSEAnswerEvent(
                                content=final_answer,
                                session_id=session_id,
                                content_format=detect_content_format(
                                    final_answer, preferred_format
                                ),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"

    except Exception as exc:
        logger.error("stream_graph_events error: %s", exc, exc_info=True)
        err_event = SSEErrorEvent(message=str(exc))
        yield f"data: {err_event.model_dump_json()}\n\n"

    duration_ms = (time.time() - t_start) * 1000
    logger.info(
        "Stream complete — session=%s duration=%.0f ms answer=%.80s",
        session_id,
        duration_ms,
        final_answer,
    )


async def invoke_graph(
    compiled_graph,
    user_message: str,
    session_id: str,
    history: list[dict] | None = None,
    preferred_format: ContentFormat | None = None,
) -> tuple[str, ContentFormat, ReasoningTrace]:
    """
    Non-streaming invocation. Returns (answer, content_format, reasoning_trace).
    """
    t_start = time.time()

    init_messages: list[BaseMessage] = []
    for h in (history or []):
        if h["role"] == "user":
            init_messages.append(HumanMessage(content=h["content"]))
        else:
            init_messages.append(AIMessage(content=h["content"]))
    init_messages.append(HumanMessage(content=user_message))

    input_state: SupervisorState = {
        "messages": init_messages,
        "session_id": session_id,
        "next_agent": "",
        "supervisor_reasoning": "",
        "reasoning_trace": [],
    }

    result = await compiled_graph.ainvoke(
        input_state,
        config={"configurable": {"thread_id": session_id}},
    )

    messages: list[BaseMessage] = result.get("messages", [])
    answer = ""
    if messages:
        last = messages[-1]
        answer = getattr(last, "content", "")
        if isinstance(answer, list):
            answer = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in answer
            )

    raw_trace: list[dict] = result.get("reasoning_trace", [])
    steps = [AgentStepTrace.model_validate(s) for s in raw_trace]

    reasoning = ReasoningTrace(
        supervisor_reasoning=result.get("supervisor_reasoning"),
        routed_to=steps[0].content if steps else None,
        steps=steps,
        total_duration_ms=(time.time() - t_start) * 1000,
    )

    fmt = detect_content_format(answer, preferred_format)
    return answer, fmt, reasoning
