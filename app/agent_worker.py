"""LiveKit AI agent worker — Entry point for the real-time interview agent.

Connects to a LiveKit room, initializes the multimodal pipeline
(VAD + STT + LLM + TTS), and runs the interview session.
"""

import asyncio
import logging
import time

from langchain_core.messages import HumanMessage
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import bey, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.llm_adapter import InterviewLLM

logger = logging.getLogger("keboli-agent")
logger.setLevel(logging.INFO)


def _parse_room_ids(room_name: str) -> tuple[str, str]:
    """Extract session_id and assessment_id from a LiveKit room name.

    Room names follow the convention: ``<prefix>_<session_id>_<assessment_id>``.

    Args:
        room_name: The LiveKit room name string.

    Returns:
        A tuple of (session_id, assessment_id).
    """
    parts = room_name.split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    if room_name == "console":
        logger.info("Running in console mode, using mock IDs")
        return "mock_session", "mock_assessment"
    return "unknown", "unknown"


def _create_agent_session(interview_llm: InterviewLLM) -> AgentSession:
    """Build and return a configured AgentSession with the full multimodal pipeline.

    Args:
        interview_llm: The InterviewLLM adapter to use as the LLM backend.

    Returns:
        A fully configured AgentSession instance.
    """
    return AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.3,
            prefix_padding_duration=0.2,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en",
            interim_results=True,
        ),
        llm=interview_llm,
        tts=deepgram.TTS(
            model="aura-orion-en",
        ),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_endpointing_delay=0.2,
        max_endpointing_delay=1.0,
        min_interruption_duration=0.4,
        discard_audio_if_uninterruptible=True,
        use_tts_aligned_transcript=True,
        preemptive_generation=True,
    )


async def entrypoint(ctx: JobContext) -> None:
    """The main execution loop for the LiveKit AI Agent.

    Sets up the multimodal session and connects the LLM to the room.

    Args:
        ctx: The JobContext containing room information and connection methods.

    Returns:
        None (starts an asynchronous background session).
    """
    room_name = ctx.room.name
    logger.info("Agent entrypoint called for room: %s", room_name)

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info("Participant joined: %s", participant.identity)

    session_id, assessment_id = _parse_room_ids(room_name)

    logger.info("Session: %s, Assessment: %s", session_id, assessment_id)

    interview_llm = InterviewLLM(
        session_id=session_id,
        assessment_id=assessment_id,
    )

    try:
        await interview_llm.initialize()
    except (RuntimeError, ValueError, OSError) as e:
        logger.exception("Failed to initialize InterviewLLM: %s", e)

    interview_llm.set_room(ctx.room)

    avatar = bey.AvatarSession()

    lk_agent = Agent(
        instructions="You are a helpful and friendly interviewer. Conduct the interview naturally.",
    )

    session = _create_agent_session(interview_llm)

    try:
        await avatar.start(session, room=ctx.room)
    except (ConnectionError, RuntimeError) as e:
        logger.warning(
            "Could not start Avatar (expected in console mode): %s",
            e,
            exc_info=True,
        )

    await session.start(
        lk_agent,
        room=ctx.room,
    )

    await session.generate_reply()

    logger.info("Agent session started successfully")

    last_interaction = time.time()
    silence_prompt_given = False

    @session.on("user_speech_committed")
    def on_user_speech(*args, **kwargs):
        nonlocal last_interaction, silence_prompt_given
        last_interaction = time.time()
        silence_prompt_given = False

    @session.on("agent_speech_committed")
    def on_agent_speech(*args, **kwargs):
        nonlocal last_interaction
        last_interaction = time.time()

    async def _inactivity_monitor():
        nonlocal last_interaction, silence_prompt_given
        while True:
            await asyncio.sleep(5)
            if ctx.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                break
                
            elapsed = time.time() - last_interaction
            if elapsed >= 60.0 and not silence_prompt_given:
                logger.info("Silence timeout (60s). Prompting candidate.")
                silence_prompt_given = True
                last_interaction = time.time()
                
                if interview_llm._state is not None:
                    interview_llm._state["messages"].append(
                        HumanMessage(content="[SYSTEM_INTERRUPT] Silence detected.")
                    )
                    try:
                        asyncio.create_task(session.generate_reply())
                    except RuntimeError as e:
                        logger.warning("Could not generate reply for silence timeout: %s", e)

    asyncio.create_task(_inactivity_monitor())



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
