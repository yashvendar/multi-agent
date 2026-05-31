"""
agents/supervisor/nodes/summarise.py
=====================================
Summarise node — runs once at the end of the pipeline to synthesise
all collected sub-agent data into a polished, user-facing response.
Goes directly to END after producing the answer.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models.schemas import AgentStepTrace
from agents.supervisor.state import SupervisorState

logger = logging.getLogger("agents.supervisor.summarise")

_SYNTHESIS_SYSTEM_PROMPT = """You are the final response writer for an industrial IoT assistant.

You have been given the original user question and all data collected by specialist agents.
Your ONLY job is to write a clear, professional, user-facing answer.

Rules:
- Synthesise all data into a single coherent narrative.
- Use plain language. Do not expose SQL, internal IDs, or JSON blobs.
- Include units (MW, %, °C, etc.) for every numeric value.
- Use a markdown table when the result has multiple rows or assets.
- Write a paragraph summary when the result is a single value.
- If no data was found, say so clearly and suggest what the user might check.
- Do not mention agent names, routing steps, or internal process details.
"""


def make_summarise_node(synthesis_llm):
    """
    Dedicated synthesis node. Reads the full conversation (including all
    sub-agent responses), writes the final user-facing answer, then routes
    straight to END. Cannot call any sub-agent or itself.
    """
    async def summarise_node(state: SupervisorState) -> dict:
        messages = [
            SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
            *state["messages"],
            HumanMessage(
                content="Based on all the data above, write the final answer for the user's original question."
            ),
        ]

        result = await synthesis_llm.ainvoke(messages)
        final_answer = result.content

        logger.info("Summarise node produced answer (%.80s).", final_answer)

        trace_entry = AgentStepTrace(
            type="answer",
            agent="summarise",
            content=final_answer,
        ).model_dump(mode="json")

        return {
            "messages": [AIMessage(content=final_answer)],
            "reasoning_trace": [trace_entry],
        }

    return summarise_node
