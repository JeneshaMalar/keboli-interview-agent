import logging
from dotenv import load_dotenv

load_dotenv()

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, silero, turn_detector, bey
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from app.llm_adapter import InterviewLLM

logger = logging.getLogger("keboli-agent")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    """
    The main execution loop for the LiveKit AI Agent. Sets up the 
    multimodal session and connects the LLM to the room.

    Args:
        ctx: The JobContext containing room information and connection methods.

    Raises:
        Exception: If InterviewLLM initialization or AgentSession startup fails.

    Returns:
        None (Starts an asynchronous background session)
    """
    room_name = ctx.room.name
    logger.info(f"Agent entrypoint called for room: {room_name}")

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

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

    logger.info(f"Session: {session_id}, Assessment: {assessment_id}")

    interview_llm = InterviewLLM(
        session_id=session_id,
        assessment_id=assessment_id,
    )

    try:
        await interview_llm.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize InterviewLLM: {e}")
    
    interview_llm.set_room(ctx.room)

    avatar = bey.AvatarSession()

    lk_agent = Agent(
        instructions="You are a helpful and friendly interviewer. Conduct the interview naturally.",
    )

    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.8, 
            prefix_padding_duration=0.2,
        ),

        stt=deepgram.STT(
            model="nova-2",
            language="en",
        ),

        llm=interview_llm,
        # aura-asteria-en
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
        avatar_id = "694c83e2-8895-4a98-bd16-56332ca3f449"
        await avatar.start(session, room=ctx.room)
    except Exception as e:
        logger.warning(f"Could not start Avatar (expected in console mode): {e}")

    await session.start(
        lk_agent,
        room=ctx.room,
    )


    await session.generate_reply()

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
