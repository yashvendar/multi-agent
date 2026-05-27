"""
models/schemas.py
=================
Pydantic models for API request/response bodies and internal state.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# Supported content formats — tells the client how to render a piece of content.
# markdown : rich text with headers, tables, code fences, bullet lists
# text     : plain unformatted string
# json     : raw JSON object/array (stringified)
# html     : HTML markup — render directly in a browser / innerHTML
ContentFormat = Literal["markdown", "text", "json", "html"]


# ─────────────────────────────────────────────────────────────────────────────
# API  Request / Response
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat request from the client."""
    message: str = Field(..., description="User's message")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Conversation session ID. Pass the same ID to continue a conversation.",
    )
    stream: bool = Field(
        default=False,
        description="If true, use the /chat/stream endpoint for SSE streaming instead.",
    )
    response_format: ContentFormat | None = Field(
        default=None,
        description=(
            "Preferred content format for the assistant's final answer. "
            "If omitted the server auto-detects: json when the answer is a "
            "valid JSON object/array, markdown when markdown syntax is detected, "
            "otherwise text."
        ),
    )


class ToolCallTrace(BaseModel):
    """Reasoning + input/output for a single tool call."""
    agent: str = Field(..., description="Which agent made this tool call")
    tool: str = Field(..., description="Tool name")
    reasoning: str | None = Field(None, description="Why the agent decided to call this tool")
    input: Any = Field(None, description="Tool input arguments")
    output: str | None = Field(None, description="Tool output (truncated)")
    duration_ms: float | None = None


class AgentStepTrace(BaseModel):
    """A single reasoning step from an agent."""
    type: Literal["route", "tool_call", "tool_result", "cross_agent_call", "answer"]
    agent: str
    content: str | None = None
    tool_call: ToolCallTrace | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReasoningTrace(BaseModel):
    """Full reasoning trace for a single chat turn."""
    supervisor_reasoning: str | None = None
    routed_to: str | None = None
    steps: list[AgentStepTrace] = Field(default_factory=list)
    total_duration_ms: float | None = None


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response."""
    session_id: str
    answer: str
    content_format: ContentFormat = Field(
        "text",
        description="How to render the answer: markdown, text, or json.",
    )
    reasoning: ReasoningTrace | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming SSE event shapes  (each serialised as a JSON line)
# ─────────────────────────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    """Base SSE event."""
    type: str


class SSERouteEvent(SSEEvent):
    type: Literal["route"] = "route"
    agent: str
    reasoning: str


class SSEToolCallEvent(SSEEvent):
    type: Literal["tool_call"] = "tool_call"
    agent: str
    tool: str
    reasoning: str | None = None
    input: Any = None


class SSEToolResultEvent(SSEEvent):
    type: Literal["tool_result"] = "tool_result"
    agent: str
    tool: str
    output: str
    content_format: ContentFormat = Field(
        "json",
        description="DB tool results are always JSON; cross-agent results may be markdown or text.",
    )


class SSETokenEvent(SSEEvent):
    type: Literal["token"] = "token"
    agent: str
    content: str
    content_format: ContentFormat = Field(
        "text",
        description="Format of this token chunk — set once at stream start, does not change mid-stream.",
    )


class SSEAnswerEvent(SSEEvent):
    type: Literal["answer"] = "answer"
    content: str
    session_id: str
    content_format: ContentFormat = Field(
        "text",
        description=(
            "How the client should render this answer. "
            "markdown: render with a Markdown parser. "
            "json: parse as JSON and display as structured data. "
            "text: display as plain string."
        ),
    )


class SSEErrorEvent(SSEEvent):
    type: Literal["error"] = "error"
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Conversation History
# ─────────────────────────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    """Single persisted message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    reasoning: ReasoningTrace | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: list[ConversationMessage]
    created_at: datetime | None = None
    updated_at: datetime | None = None
