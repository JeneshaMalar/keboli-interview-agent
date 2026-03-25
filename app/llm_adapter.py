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
    """
    A custom LiveKit LLM implementation that acts as a bridge between
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
        """
        Assign the LiveKit room reference to allow the adapter to
        send control signals (like timer sync or interview end) to the frontend.

        Args:
            room: The active LiveKit RTC room instance.
        """
        self._room = room

    async def initialize(self) -> None:
        """
        Setup the initial interview state, including skill tracking,
        timing metrics, and phase flags.

        Args:
            None

        Returns:
            None
        """
        if self._initialized:
            return

        logger.info(
            f"Initializing interview state for assessment {self._assessment_id}"
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
        """
        Broadcast a 'interview_ended' signal to the frontend via
        the LiveKit Data Channel.

        Args:
            reason: String explaining why the interview stopped (e.g., 'completed', 'timeout').

        Returns:
            None
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
            logger.info(f"Emitted interview_ended signal (reason={reason}) to room")
        except Exception as e:
            logger.error(f"Failed to emit interview_ended: {e}")

    async def _emit_timer_sync(self, remaining_seconds: int) -> None:
        """
        Send the current remaining interview time to the frontend
        to ensure the UI clock matches the agent's internal state.

        Args:
            remaining_seconds: Number of seconds left before the interview expires.

        Returns:
            None
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
        except Exception as e:
            logger.warning(f"Failed to emit timer_sync: {e}")

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
        if self._start_time is None:
            return 0
        return int(time.time() - self._start_time)


class InterviewLLMStream(llm.LLMStream):
    """
    Handles the execution logic for a single chat turn. It processes
    the user's voice transcript through LangGraph and streams the
    interviewer's response back to LiveKit.
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

    async def _run(self) -> None:
        """
        Main execution loop for the stream. Extracts user intent,
        invokes the LangGraph agent, updates state, and handles
        session completion logic.

        Args:
            None

        Returns:
            None (streams results through self._event_ch)
        """
        try:
            self._state["elapsed_time_seconds"] = (
                self._interview_llm.get_elapsed_seconds()
            )

            latest_user_msg = ""
            with open("/tmp/keboli_debug.log", "a") as f:
                f.write("\n--- Chat Turn ---\n")
                f.write(f"Chat Context Items: {len(self._chat_ctx.items)}\n")
                for i, msg in enumerate(self._chat_ctx.items):
                    role = getattr(msg, "role", "N/A")
                    content_type = type(getattr(msg, "content", None))
                    f.write(
                        f"Item {i}: role={role} (type={type(msg)}), content_type={content_type}\n"
                    )

            for msg in reversed(self._chat_ctx.items):
                if not getattr(msg, "role", None) or not getattr(msg, "content", None):
                    continue

                role_str = str(getattr(msg, "role", "")).lower()
                if "user" in role_str or "human" in role_str:
                    msg_content = getattr(msg, "content", "")
                    if isinstance(msg_content, str):
                        latest_user_msg = msg_content
                    elif isinstance(msg_content, list):
                        for part in msg_content:
                            if hasattr(part, "text"):
                                latest_user_msg = str(getattr(part, "text", ""))
                                break
                            elif isinstance(part, str):
                                latest_user_msg = part
                                break
                    break

            if latest_user_msg:
                with open("/tmp/keboli_debug.log", "a") as f:
                    f.write(f"Extracted user message: {latest_user_msg}\n")

                self._state["messages"].append(HumanMessage(content=latest_user_msg))
                try:
                    await keboli_client.append_transcript(
                        self._state["session_id"], "candidate", latest_user_msg
                    )
                except Exception as e:
                    logger.error(f"Failed to append candidate transcript: {e}")

            logger.info(
                f"Invoking LangGraph with user message: {latest_user_msg[:80]}..."
            )
            config: dict[str, Any] = {
                "run_name": f"interview-{self._state['session_id']}"
            }
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

            result = await self._interview_agent.ainvoke(self._state, config=config)

            ai_response = ""
            for m in reversed(result.get("messages", [])):
                if isinstance(m, AIMessage):
                    content = m.content
                    ai_response = (
                        str(content)
                        if isinstance(content, str)
                        else str(content[0])
                        if isinstance(content, list) and content
                        else ""
                    )
                    break

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
            except Exception as e:
                logger.warning(f"Failed to append interviewer transcript: {e}")

            logger.info(f"Agent response: {ai_response[:80]}...")

            is_completed = result.get("is_completed", False)
            closing_reason = result.get("closing_reason", "completed")

            if not is_completed and _is_closing_message(ai_response):
                logger.warning(
                    "Safety-net: closing keywords detected in response but is_completed=False. Forcing completion."
                )
                is_completed = True
                closing_reason = "auto_detected_closing"

            if is_completed:
                logger.info(
                    f"Interview {self._state['session_id']} marked as COMPLETED (reason={closing_reason}). Triggering evaluation..."
                )

                await self._interview_llm._emit_interview_ended(closing_reason)

                try:
                    await keboli_client.complete_session(self._state["session_id"])
                except Exception as e:
                    logger.error(f"Failed to trigger session completion: {e}")

        except Exception as e:
            logger.error(f"Error in InterviewLLMStream: {e}", exc_info=True)
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=self._llm._label,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content="I'm sorry, I had a momentary issue. Could you please repeat that?",
                    ),
                )
            )
