"""
main.py
=======
FastAPI application entrypoint.

Endpoints
---------
POST /chat                      — Full (non-streaming) chat response
GET  /chat/stream               — SSE streaming chat with reasoning events
GET  /conversations/{session_id} — Conversation history
GET  /health                    — Health check
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.amm_agent import build_amm_agent
from agents.data_explorer_agent import build_data_explorer_agent
from agents.kpi_agent import build_kpi_agent
from agents.supervisor import build_supervisor_graph, invoke_graph, stream_graph_events
from config import settings
from db.conversation import get_history, get_recent_messages_for_context, init_db, save_message
from db.docs import init_docs_table
from models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationMessage,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
# App lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

compiled_graph = None  # set during startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph

    logger.info("=== IC-Agent startup ===")

    # 1. Initialise conversation DB (creates tables if needed)
    init_db()

    # 2. Initialise docs vector table
    docs_dsn = settings.docs_db_dsn or settings.conv_db_dsn
    init_docs_table(docs_dsn)

    # 2. Build sub-agents (register in AgentRegistry)
    #    Order matters only for logging; cross-agent calls are lazy lookups
    build_amm_agent()
    build_data_explorer_agent()
    build_kpi_agent()

    # 3. Build supervisor graph (needs all agents registered)
    compiled_graph = build_supervisor_graph()

    logger.info("=== IC-Agent ready ===")
    yield

    logger.info("=== IC-Agent shutdown ===")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IC-Agent — Industrial IoT Multi-Agent Chat",
    description=(
        "Supervisor + KPI Configurator + Data Explorer + AMM agents "
        "powered by LangGraph and Gemini (Google ADC)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers.confluence import router as docs_router  # noqa: E402
app.include_router(docs_router)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Quick health check — returns service status and timestamp."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "graph_ready": compiled_graph is not None,
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message and receive a full response with reasoning trace.

    The supervisor decides which sub-agent handles the request.
    The reasoning trace shows the routing decision, tool calls (with WHY
    each tool was called), and the final answer.
    """
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not ready yet.")

    session_id = request.session_id or str(uuid.uuid4())

    # Load recent history for context
    history = get_recent_messages_for_context(session_id, limit=10)

    # Persist user message
    save_message(
        ConversationMessage(
            session_id=session_id,
            role="user",
            content=request.message,
        )
    )

    # Invoke graph
    answer, content_format, reasoning = await invoke_graph(
        compiled_graph,
        user_message=request.message,
        session_id=session_id,
        history=history,
        preferred_format=request.response_format,
    )

    # Persist assistant response (with reasoning)
    save_message(
        ConversationMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            reasoning=reasoning,
        )
    )

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        content_format=content_format,
        reasoning=reasoning,
    )


@app.get("/chat/stream", tags=["Chat"])
async def chat_stream(
    message: str = Query(..., description="User's message"),
    session_id: str = Query(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID — omit to start a new conversation",
    ),
    response_format: str | None = Query(
        default=None,
        description=(
            "Preferred content format for the answer event: "
            "'markdown' | 'text' | 'json'. "
            "Omit for auto-detection."
        ),
    ),
):
    """
    Stream the agent's response as Server-Sent Events (SSE).

    Each event is a JSON object with a **type** field:

    | type | description |
    |---|---|
    | `route` | Supervisor decided to route to an agent (includes reasoning) |
    | `tool_call` | An agent is calling a tool (includes WHY + input) |
    | `tool_result` | Result from a tool call |
    | `token` | A streamed token from the LLM |
    | `answer` | Final answer |
    | `error` | An error occurred |

    **Client example (JavaScript)**
    ```js
    const es = new EventSource(`/chat/stream?message=...&session_id=...`);
    es.onmessage = e => {
      const event = JSON.parse(e.data);
      if (event.type === 'route')      console.log('Routing to:', event.agent, '—', event.reasoning);
      if (event.type === 'tool_call')  console.log('Tool call:', event.tool, '—', event.reasoning);
      if (event.type === 'answer')     console.log('Answer:', event.content);
    };
    ```
    """
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not ready yet.")

    # Validate response_format
    valid_formats = {"markdown", "text", "json", "html", None}
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=422,
            detail=f"response_format must be one of: markdown, text, json, html. Got: '{response_format}'",
        )
    preferred_fmt = response_format  # already validated, mypy-safe as str|None

    session_id = session_id or str(uuid.uuid4())
    history = get_recent_messages_for_context(session_id, limit=10)

    # Persist user message immediately
    save_message(
        ConversationMessage(session_id=session_id, role="user", content=message)
    )

    # Collect answer for persistence after stream completes
    collected_events: list[dict] = []

    async def event_generator():
        answer_text = ""
        async for line in stream_graph_events(
            compiled_graph,
            user_message=message,
            session_id=session_id,
            history=history,
            preferred_format=preferred_fmt,  # type: ignore[arg-type]
        ):
            yield line
            # Parse to collect the final answer
            if line.startswith("data: "):
                try:
                    ev = json.loads(line[6:])
                    collected_events.append(ev)
                    if ev.get("type") == "answer":
                        answer_text = ev.get("content", "")
                except json.JSONDecodeError:
                    pass

        # After stream: persist the assistant answer
        if answer_text:
            save_message(
                ConversationMessage(
                    session_id=session_id,
                    role="assistant",
                    content=answer_text,
                )
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
        },
    )


@app.get(
    "/conversations/{session_id}",
    response_model=ConversationHistory,
    tags=["Conversations"],
)
def get_conversation(
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
):
    """Retrieve conversation history for a session, including per-turn reasoning traces."""
    return get_history(session_id, limit=limit)


@app.get("/conversations/{session_id}/reasoning", tags=["Conversations"])
def get_reasoning_trace(
    session_id: str,
    limit: int = Query(10, ge=1, le=50),
):
    """
    Return only the reasoning traces (tool calls + routing decisions)
    for the last *limit* assistant turns in a session.
    """
    history = get_history(session_id, limit=limit * 2)
    traces = []
    for msg in history.messages:
        if msg.role == "assistant" and msg.reasoning:
            traces.append(
                {
                    "message_id": msg.id,
                    "answer_preview": msg.content[:120] + "..."
                    if len(msg.content) > 120
                    else msg.content,
                    "reasoning": msg.reasoning.model_dump(),
                    "created_at": msg.created_at.isoformat(),
                }
            )
    return {"session_id": session_id, "traces": traces[-limit:]}


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
