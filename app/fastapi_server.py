"""Keboli Interview Agent — FastAPI server.

Provides HTTP endpoints for the interview chat loop and skill graph
generation, used by both the WebSocket-based and LiveKit-based
interview flows.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.exceptions import AppError, ExternalServiceError, ValidationError
from app.graph import interview_agent
from app.keboli_client import keboli_client
from app.llm import llm
from app.prompt_manager import SKILL_EXTRACTION_PROMPT, SkillGraph

logger = logging.getLogger("keboli-fastapi")

app = FastAPI(
    title="Keboli Interview Agent",
    description="Real-time AI interviewer API using LangGraph.",
    version="1.0.0",
)




@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Convert AppError exceptions into structured JSON error responses.

    Args:
        request: The incoming HTTP request.
        exc: The AppError instance containing error details.

    Returns:
        A JSONResponse with the appropriate status code and error body.
    """
    logger.error(
        "app_error: %s (code=%s, status=%d)",
        exc.message,
        exc.error_code,
        exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )




class InterviewTurnRequest(BaseModel):
    """Payload for a single interview chat turn."""

    session_id: str
    assessment_id: str
    last_message: str | None = None
    state: dict[str, Any] | None = None


class SkillGraphRequest(BaseModel):
    """Payload for triggering skill graph generation."""

    assessment_id: str


class ServiceStatus(BaseModel):
    """Status of individual backing services."""

    llm: str = "unknown"
    keboli_client: str = "unknown"


class HealthResponse(BaseModel):
    """Structured health-check response."""

    status: str = "ok"
    timestamp: str = ""
    services: ServiceStatus = ServiceStatus()
    error: str | None = None


class SkillGraphResponse(BaseModel):
    """Response returned after skill graph generation."""

    status: str
    skill_graph: dict[str, Any]


class ChatResponse(BaseModel):
    """Response returned after processing a chat turn."""

    response: str
    is_completed: bool
    state: dict[str, Any]




@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description="Verify LLM and backend client readiness.",
)
async def health_check() -> JSONResponse | HealthResponse:
    """Check application health by verifying LLM and client availability.

    Returns:
        HealthResponse with service statuses and overall health.
    """
    services = ServiceStatus()

    try:
        if llm is not None:
            services.llm = "ok"

        if keboli_client:
            services.keboli_client = "ok"

        is_healthy = services.llm == "ok" and services.keboli_client == "ok"

        payload = HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            services=services,
        )

        if not is_healthy:
            return JSONResponse(
                status_code=503,
                content=payload.model_dump(),
            )

        return payload

    except Exception as e:
        logger.error("Critical failure in health check endpoint: %s", e)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@app.post(
    "/generate-skill-graph",
    response_model=SkillGraphResponse,
    summary="Generate a skill graph from a job description",
    description="Analyze an assessment's job description to extract a structured skill graph using the LLM.",
)
async def generate_skill_graph(request: SkillGraphRequest) -> SkillGraphResponse:
    """Analyze an assessment's job description to generate a structured skill graph.

    Uses a structured LLM output to extract specific skills,
    required experience levels, and reasoning from raw text. Persists
    the graph back to the assessment record.

    Args:
        request: SkillGraphRequest containing the assessment_id to analyze.

    Returns:
        SkillGraphResponse with the status and generated skill_graph.

    Raises:
        ValidationError: If the assessment has no job description.
        ExternalServiceError: If LLM generation or database update fails.
    """
    try:
        assessment = await keboli_client.get_assessment(request.assessment_id)

        existing_graph = assessment.get("skill_graph")
        if existing_graph:
            return SkillGraphResponse(status="exists", skill_graph=existing_graph)

        job_description = assessment.get("job_description")
        if not job_description:
            raise ValidationError(
                message="Assessment has no job description",
                field="job_description",
            )

        difficulty_level = assessment.get("difficulty_level", "medium")

        logger.info(
            "Generating skill graph for assessment %s (difficulty=%s).",
            request.assessment_id,
            difficulty_level,
        )

        await keboli_client.post_log(
            {
                "level": "INFO",
                "service": "interview_agent",
                "component": "skill_graph",
                "event_type": "graph_generation_started",
                "assessment_id": request.assessment_id,
                "message": f"Started skill graph generation for assessment {request.assessment_id} (difficulty={difficulty_level})",
            }
        )

        prompt = ChatPromptTemplate.from_template(SKILL_EXTRACTION_PROMPT)
        structured_llm = llm.with_structured_output(SkillGraph)
        chain = prompt | structured_llm

        result = await chain.ainvoke(
            {
                "job_description": job_description,
                "difficulty_level": difficulty_level,
            }
        )

        skill_graph: dict[str, Any] = result  

        logger.info(
            "Skill graph generated — experience_level=%s, %d skills extracted",
            skill_graph.get("experience_level"),
            len(skill_graph.get("skills", [])),
        )

        await keboli_client.post_log(
            {
                "level": "INFO",
                "service": "interview_agent",
                "component": "skill_graph",
                "event_type": "graph_generation_completed",
                "assessment_id": request.assessment_id,
                "message": "Skill graph generated successfully",
                "details": {
                    "experience_level": skill_graph.get("experience_level"),
                    "skills_extracted": len(skill_graph.get("skills", [])),
                },
            }
        )

        await keboli_client.update_assessment_skills(request.assessment_id, skill_graph)

        return SkillGraphResponse(status="generated", skill_graph=skill_graph)

    except AppError:
        raise
    except Exception as e:
        await keboli_client.post_log(
            {
                "level": "ERROR",
                "service": "interview_agent",
                "component": "skill_graph",
                "event_type": "graph_generation_failed",
                "assessment_id": request.assessment_id,
                "message": f"Error generating skill graph: {e!s}",
                "error_stack": str(e),
            }
        )
        logger.error("Error generating skill graph: %s", e, exc_info=True)
        raise ExternalServiceError(
            service_name="LLM",
            message=f"Failed to generate skill graph: {e!s}",
        ) from e


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process a single interview chat turn",
    description="Receives the candidate's message, runs the LangGraph agent, and returns the interviewer's response.",
)
async def chat(request: InterviewTurnRequest) -> ChatResponse:
    """Process a single turn of the interview conversation.

    Maintains the state machine for the interview. Receives the
    candidate's last message, runs the LangGraph agent to determine
    the next question or response, and returns the updated state.

    Args:
        request: InterviewTurnRequest containing session/assessment IDs and message history.

    Returns:
        ChatResponse with the AI's response, completion status, and serializable state.

    Raises:
        AppError: If the LangGraph invocation or message serialization fails.
    """
    if request.state:
        state: dict[str, Any] = request.state
        msg_objs: list[HumanMessage | AIMessage] = []
        for m in state.get("messages", []):
            if m["role"] == "human":
                msg_objs.append(HumanMessage(content=m["content"]))
            else:
                msg_objs.append(AIMessage(content=m["content"]))
        state["messages"] = msg_objs
    else:
        state = {
            "session_id": request.session_id,
            "assessment_id": request.assessment_id,
            "messages": [],
            "current_skill_index": 0,
            "current_skill_depth": 0,
            "elapsed_time_seconds": 0,
            "is_completed": False,
            "conversation_phase": "greeting",
            "previous_skill_name": None,
            "nudge_count": 0,
            "closing_phase": None,
            "closing_reason": None,
            "time_warning_given": False,
            "qa_phase": False,
            "qa_turns": 0,
        }

    if request.last_message:
        state["messages"].append(HumanMessage(content=request.last_message))

    try:
        final_state: dict[str, Any] = await interview_agent.ainvoke(state)

        last_ai_message = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                content = m.content
                last_ai_message = (
                    str(content)
                    if isinstance(content, str)
                    else str(content[0])
                    if isinstance(content, list) and content
                    else ""
                )
                break

        serializable_messages: list[dict[str, str]] = []
        for m in final_state["messages"]:
            if isinstance(m, HumanMessage):
                hcontent = m.content
                hstr = (
                    str(hcontent)
                    if isinstance(hcontent, str)
                    else str(hcontent[0])
                    if isinstance(hcontent, list) and hcontent
                    else ""
                )
                serializable_messages.append({"role": "human", "content": hstr})
            elif isinstance(m, AIMessage):
                acontent = m.content
                astr = (
                    str(acontent)
                    if isinstance(acontent, str)
                    else str(acontent[0])
                    if isinstance(acontent, list) and acontent
                    else ""
                )
                serializable_messages.append({"role": "ai", "content": astr})

        response_state: dict[str, Any] = {
            **{k: v for k, v in final_state.items() if k != "messages"},
            "messages": serializable_messages,
        }

        await keboli_client.post_log(
            {
                "level": "INFO",
                "service": "interview_agent",
                "component": "chat",
                "event_type": "chat_turn_completed",
                "session_id": request.session_id,
                "assessment_id": request.assessment_id,
                "message": "Processed interview chat turn",
                "details": {"is_completed": final_state.get("is_completed", False)},
            }
        )

        return ChatResponse(
            response=last_ai_message,
            is_completed=final_state.get("is_completed", False),
            state=response_state,
        )
    except AppError:
        raise
    except Exception as e:
        await keboli_client.post_log(
            {
                "level": "ERROR",
                "service": "interview_agent",
                "component": "chat",
                "event_type": "chat_turn_failed",
                "session_id": request.session_id,
                "assessment_id": request.assessment_id,
                "message": f"Error processing chat turn: {e!s}",
                "error_stack": str(e),
            }
        )
        raise AppError(
            message=f"Chat turn failed: {e!s}",
            status_code=500,
            error_code="CHAT_TURN_FAILED",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
