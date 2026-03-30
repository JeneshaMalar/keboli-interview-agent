"""LiveKit LLM adapter bridging real-time voice agents to a LangGraph interview state machine.

This module implements a custom LiveKit LLM that intercepts voice transcripts,
routes them through a LangGraph-based interview workflow, and streams the
interviewer's AI-generated responses back to the LiveKit room.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from app.config import settings
from app.graph import interview_agent
from app.keboli_client import keboli_client
from app.observability import langfuse_handler

logger = logging.getLogger("keboli-llm-adapter")

CLOSING_KEYWORDS = [
    "thank you for your time",
    "that wraps up",
    "we've covered everything",
    "best of luck",
    "wish you well",
    "covered all the areas",
    "thanks for taking the time",
    "good luck with your application",
    "it was great talking to you",
    "we've covered all",
]


def _is_closing_message(text: str) -> bool:
    """Detect if the LLM response is a closing message (safety net)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CLOSING_KEYWORDS)


class InterviewLLM(llm.LLM):  # type: ignore[type-arg]
    """A custom LiveKit LLM implementation that acts as a bridge between
    LiveKit's real-time agents and a LangGraph-based interview state machine.
    """

    def __init__(self, session_id: str, assessment_id: str):
        super().__init__()
        self._session_id = session_id
        self._assessment_id = assessment_id
        self._state: dict[str, Any] | None = None
        self._initialized = False
        self._start_time: float | None = None
        self._room: rtc.Room | None = None

    def set_room(self, room: rtc.Room) -> None:
        """Assign the LiveKit room reference to allow the adapter to
        send control signals (like timer sync or interview end) to the frontend.

        Args:
            room: The active LiveKit RTC room instance.
        """
        self._room = room

    async def initialize(self) -> None:
        """Setup the initial interview state, including skill tracking,
        timing metrics, and phase flags.

        Raises:
            RuntimeError: If the interview state cannot be set up.
        """
        if self._initialized:
            return

        logger.info(
            "Initializing interview state for assessment %s",
            self._assessment_id,
        )

        initial_state = {
            "session_id": self._session_id,
            "assessment_id": self._assessment_id,
            "messages": [],
            "current_skill_index": 0,
            "current_skill_depth": 0,
            "elapsed_time_seconds": 0,
            "is_completed": False,
            "nudge_count": 0,
            "closing_phase": None,
            "closing_reason": None,
            "time_warning_given": False,
            "qa_phase": False,
            "qa_turns": 0,
        }

        self._state = initial_state
        self._initialized = True
        self._start_time = time.time()

    async def _emit_interview_ended(self, reason: str) -> None:
        """Broadcast a 'interview_ended' signal to the frontend via
        the LiveKit Data Channel.

        Args:
            reason: String explaining why the interview stopped
                    (e.g., 'completed', 'timeout').
        """
        if not self._room:
            logger.warning("No room reference — cannot emit interview_ended signal")
            return

        try:
            payload = json.dumps(
                {
                    "type": "interview_ended",
                    "reason": reason,
                    "auto_submit": True,
                    "session_id": self._session_id,
                }
            ).encode("utf-8")

            await self._room.local_participant.publish_data(
                payload,
                reliable=True,
                topic="interview_control",
            )
            logger.info("Emitted interview_ended signal (reason=%s) to room", reason)
        except (ConnectionError, RuntimeError) as e:
            logger.exception("Failed to emit interview_ended: %s", e)

    async def _emit_timer_sync(self, remaining_seconds: int) -> None:
        """Send the current remaining interview time to the frontend
        to ensure the UI clock matches the agent's internal state.

        Args:
            remaining_seconds: Number of seconds left before the interview expires.
        """
        if not self._room:
            return
        try:
            payload = json.dumps(
                {
                    "type": "timer_sync",
                    "remaining_seconds": remaining_seconds,
                }
            ).encode("utf-8")
            await self._room.local_participant.publish_data(
                payload,
                reliable=True,
                topic="interview_control",
            )
        except (ConnectionError, RuntimeError) as e:
            logger.warning("Failed to emit timer_sync: %s", e, exc_info=True)

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        tool_choice: llm.ToolChoice | None = None,  # type: ignore[override]
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        **kwargs: Any,
    ) -> InterviewLLMStream:
        """Create a new LLM stream that processes a single chat turn.

        Extracts the latest user message from the chat context, invokes
        the LangGraph interview agent, and returns a stream that emits
        the interviewer's response.

        Args:
            chat_ctx: The current LiveKit chat context with message history.
            tools: Optional list of tools (unused in interview mode).
            tool_choice: Optional tool selection preference.
            conn_options: API connection options for the stream.
            parallel_tool_calls: Whether to allow parallel tool calls.
            **kwargs: Additional keyword arguments.

        Returns:
            An InterviewLLMStream that will execute the interview turn.
        """
        return InterviewLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            state=self._state or {},
            interview_agent=interview_agent,
        )

    def _update_state(self, new_state: dict[str, Any]) -> None:
        if self._state is None:
            self._state = {}
        for key, value in new_state.items():
            if key == "messages":
                self._state["messages"] = value
            else:
                self._state[key] = value

    def get_elapsed_seconds(self) -> int:
        """Calculate the number of full seconds elapsed since the interview started.

        Returns:
            Integer seconds elapsed, or 0 if the interview has not started.
        """
        if self._start_time is None:
            return 0
        return int(time.time() - self._start_time)


class InterviewLLMStream(llm.LLMStream):
    """Handles the execution logic for a single chat turn.

    Processes the user's voice transcript through LangGraph and streams
    the interviewer's response back to LiveKit.
    """

    def __init__(
        self,
        *,
        llm: InterviewLLM,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        state: dict[str, Any],
        interview_agent: Any,
    ):
        super().__init__(
            llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._interview_llm = llm
        self._state = state or {}
        self._interview_agent = interview_agent

    def _extract_latest_user_message(self) -> str:
        """Extract the most recent user message from the chat context.

        Returns:
            The user's latest message text, or empty string if none found.
        """
        for msg in reversed(self._chat_ctx.items):
            if not getattr(msg, "role", None) or not getattr(msg, "content", None):
                continue

            role_str = str(getattr(msg, "role", "")).lower()
            if "user" in role_str or "human" in role_str:
                msg_content = getattr(msg, "content", "")
                if isinstance(msg_content, str):
                    return msg_content
                if isinstance(msg_content, list):
                    for part in msg_content:
                        if hasattr(part, "text"):
                            return str(getattr(part, "text", ""))
                        if isinstance(part, str):
                            return part
                break
        return ""

    def _extract_ai_response(self, result: dict[str, Any]) -> str:
        """Extract the latest AI response from the graph result.

        Args:
            result: The LangGraph invocation result dict.

        Returns:
            The AI response text, or empty string if none found.
        """
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage):
                content = m.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list) and content:
                    return str(content[0])
                return ""
        return ""

    async def _log_debug_context(self) -> None:
        """Log chat context details to the configured debug log file."""
        debug_log_path = getattr(settings, "DEBUG_LOG_PATH", None)
        if not debug_log_path:
            return

        try:
            with open(debug_log_path, "a") as f:
                f.write("\n--- Chat Turn ---\n")
                f.write(f"Chat Context Items: {len(self._chat_ctx.items)}\n")
                for i, msg in enumerate(self._chat_ctx.items):
                    role = getattr(msg, "role", "N/A")
                    content_type = type(getattr(msg, "content", None))
                    f.write(
                        f"Item {i}: role={role} (type={type(msg)}), content_type={content_type}\n"
                    )
        except OSError as e:
            logger.warning("Could not write to debug log %s: %s", debug_log_path, e)

    async def _handle_completion(
        self, result: dict[str, Any], ai_response: str
    ) -> None:
        """Handle interview completion: emit signals and trigger evaluation.

        Args:
            result: The LangGraph result dict.
            ai_response: The final AI response text.
        """
        is_completed = result.get("is_completed", False)
        closing_reason = result.get("closing_reason", "completed")

        if not is_completed and _is_closing_message(ai_response):
            logger.warning(
                "Safety-net: closing keywords detected in response but "
                "is_completed=False. Forcing completion."
            )
            is_completed = True
            closing_reason = "auto_detected_closing"

        if is_completed:
            logger.info(
                "Interview %s marked as COMPLETED (reason=%s). Triggering evaluation...",
                self._state["session_id"],
                closing_reason,
            )

            await self._interview_llm._emit_interview_ended(closing_reason)

            try:
                await keboli_client.complete_session(self._state["session_id"])
            except (ConnectionError, OSError, ValueError) as e:
                logger.exception("Failed to trigger session completion: %s", e)

    async def _run(self) -> None:
        """Main execution loop for the stream.

        Extracts user intent, invokes the LangGraph agent, updates state,
        and handles session completion logic.
        """
        try:
            self._state["elapsed_time_seconds"] = (
                self._interview_llm.get_elapsed_seconds()
            )

            await self._log_debug_context()

            latest_user_msg = self._extract_latest_user_message()

            if latest_user_msg:
                self._state["messages"].append(HumanMessage(content=latest_user_msg))
                try:
                    await keboli_client.append_transcript(
                        self._state["session_id"], "candidate", latest_user_msg
                    )
                except (ConnectionError, OSError, ValueError) as e:
                    logger.exception("Failed to append candidate transcript: %s", e)

            logger.info(
                "Invoking LangGraph with user message: %.80s...", latest_user_msg
            )
            config: dict[str, Any] = {
                "run_name": f"interview-{self._state['session_id']}"
            }
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

            result = await self._interview_agent.ainvoke(self._state, config=config)

            ai_response = self._extract_ai_response(result)

            self._interview_llm._update_state(result)

            total_secs = (
                result.get(
                    "total_duration_minutes",
                    self._state.get("total_duration_minutes", 30),
                )
                * 60
            )
            elapsed = self._interview_llm.get_elapsed_seconds()
            remaining = max(0, total_secs - elapsed)
            await self._interview_llm._emit_timer_sync(remaining)

            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=self._llm._label,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content=ai_response,
                    ),
                )
            )

            try:
                await keboli_client.append_transcript(
                    self._state["session_id"], "interviewer", ai_response
                )
            except (ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "Failed to append interviewer transcript: %s", e, exc_info=True
                )

            logger.info("Agent response: %.80s...", ai_response)

            await self._handle_completion(result, ai_response)

        except (KeyError, TypeError, ValueError, RuntimeError) as e:
            logger.exception("Error in InterviewLLMStream: %s", e)
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=self._llm._label,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content="I'm sorry, I had a momentary issue. Could you please repeat that?",
                    ),
                )
            )
