"""
agents/supervisor/streaming.py
================================
SSE streaming and non-streaming invocation helpers.
"""
from __future__ import annotations

import json
import logging
import re as _re
import time
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage

from models.schemas import (
    AgentStepTrace,
    ContentFormat,
    ReasoningTrace,
    SSEAnswerEvent,
    SSEErrorEvent,
    SSERouteEvent,
    SSEToolCallEvent,
    SSEToolResultEvent,
)
from agents.supervisor.state import SupervisorState

logger = logging.getLogger("agents.supervisor.streaming")


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
_MD_RE = _re.compile("|".join(_MD_PATTERNS), _re.MULTILINE)
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

    if stripped and stripped[0] in ("{", "["):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass

    if _HTML_RE.match(stripped):
        return "html"

    if _MD_RE.search(stripped):
        return "markdown"

    return "text"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming helper
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

    Each line is a JSON object with a ``type`` field:
      route | tool_call | tool_result | token | answer | error
    """
    t_start = time.time()

    init_messages = []
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
        "routing_history": [],
        "execution_plan": [],
        "plan_step": 0,
    }

    final_answer = ""

    try:
        async for chunk in compiled_graph.astream(
            input_state,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                if node_name in ("planner", "supervisor"):
                    reasoning = node_output.get("supervisor_reasoning", "")
                    next_agent = node_output.get("next_agent", "FINISH")

                    if node_name == "planner" and next_agent == "FINISH":
                        # Direct answer from planner (greeting / chitchat)
                        msgs = node_output.get("messages", [])
                        if msgs:
                            final_answer = getattr(msgs[-1], "content", "")
                            event = SSEAnswerEvent(
                                content=final_answer,
                                session_id=session_id,
                                content_format=detect_content_format(final_answer, preferred_format),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"
                        continue

                    if next_agent not in ("FINISH", "", "summarise"):
                        event = SSERouteEvent(agent=next_agent, reasoning=reasoning)
                        yield f"data: {event.model_dump_json()}\n\n"
                    elif next_agent == "FINISH":
                        msgs = node_output.get("messages", [])
                        if msgs:
                            final_answer = getattr(msgs[-1], "content", "")
                            event = SSEAnswerEvent(
                                content=final_answer,
                                session_id=session_id,
                                content_format=detect_content_format(final_answer, preferred_format),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"
                        # else: route_after_supervisor redirected to summarise — nothing to do here.

                else:
                    # Sub-agent or summarise node output — emit trace events
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
                                content_format=detect_content_format(raw_output),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"

                        elif step_type == "answer":
                            final_answer = step.get("content", "")
                            event = SSEAnswerEvent(
                                content=final_answer,
                                session_id=session_id,
                                content_format=detect_content_format(final_answer, preferred_format),
                            )
                            yield f"data: {event.model_dump_json()}\n\n"

    except Exception as exc:
        logger.error("stream_graph_events error: %s", exc, exc_info=True)
        err_event = SSEErrorEvent(message=str(exc))
        yield f"data: {err_event.model_dump_json()}\n\n"

    duration_ms = (time.time() - t_start) * 1000
    logger.info(
        "Stream complete — session=%s duration=%.0f ms answer=%.80s",
        session_id, duration_ms, final_answer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming invocation
# ─────────────────────────────────────────────────────────────────────────────

async def invoke_graph(
    compiled_graph,
    user_message: str,
    session_id: str,
    history: list[dict] | None = None,
    preferred_format: ContentFormat | None = None,
) -> tuple[str, ContentFormat, ReasoningTrace]:
    """Non-streaming invocation. Returns (answer, content_format, reasoning_trace)."""
    t_start = time.time()

    init_messages = []
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
        "routing_history": [],
        "execution_plan": [],
        "plan_step": 0,
    }

    result = await compiled_graph.ainvoke(
        input_state,
        config={"configurable": {"thread_id": session_id}},
    )

    messages = result.get("messages", [])
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
