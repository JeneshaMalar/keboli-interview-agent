"""LiveKit AI agent worker — Entry point for the real-time interview agent.

Connects to a LiveKit room, initializes the multimodal pipeline
(VAD + STT + LLM + TTS), and runs the interview session.
"""

import logging

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import bey, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.llm_adapter import InterviewLLM

logger = logging.getLogger("keboli-agent")
logger.setLevel(logging.INFO)


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

    parts = room_name.split("_")

    session_id = "unknown"
    assessment_id = "unknown"

    if len(parts) >= 3:
        session_id = parts[1]
        assessment_id = parts[2]
    elif room_name == "console":
        logger.info("Running in console mode, using mock IDs")
        session_id = "mock_session"
        assessment_id = "mock_assessment"

    logger.info("Session: %s, Assessment: %s", session_id, assessment_id)

    interview_llm = InterviewLLM(
        session_id=session_id,
        assessment_id=assessment_id,
    )

    try:
        await interview_llm.initialize()
    except Exception as e:
        logger.error("Failed to initialize InterviewLLM: %s", e)

    interview_llm.set_room(ctx.room)

    avatar = bey.AvatarSession()

    lk_agent = Agent(
        instructions="You are a helpful and friendly interviewer. Conduct the interview naturally.",
    )

    session: AgentSession = AgentSession(  
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.5,
            prefix_padding_duration=0.2,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en",
        ),
        llm=interview_llm,
        tts=deepgram.TTS(
            model="aura-orion-en",
        ),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_endpointing_delay=1.0,
        max_endpointing_delay=3.0,
        min_interruption_duration=0.5,
        discard_audio_if_uninterruptible=True,
    )

    try:
        await avatar.start(session, room=ctx.room)
    except Exception as e:
        logger.warning("Could not start Avatar (expected in console mode): %s", e)

    await session.start(
        lk_agent,
        room=ctx.room,
    )

    await session.generate_reply()

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
