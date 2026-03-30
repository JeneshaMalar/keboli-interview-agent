"""Interview workflow nodes for the LangGraph-based interview state machine.

Contains the core nodes (greeting, interview, closing, finalize) that drive
the adaptive interview loop, including skill progression, nudging, time
management, and graceful session closure.
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.llm import llm
from app.prompt_manager import (
    ADAPTIVE_INTERVIEW_PROMPT,
    CLOSING_PROMPT,
    GREETING_PROMPT,
    NUDGE_PROMPT,
    SKILL_TRANSITION_PROMPT,
    TIME_WARNING_PROMPT,
    WARMUP_TRANSITION_PROMPT,
    get_prompt,
)
from app.state import InterviewState
from app.timing import compute_interview_timing, should_start_new_skill, should_ask_followup, with_default_pacing


def get_last_human_message(messages: list[BaseMessage]) -> str:
    """Extract the content of the last human message from a message list."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(getattr(m, "content", ""))
    return ""


def get_last_ai_message(messages: list[BaseMessage]) -> str:
    """Extract the content of the last AI message from a message list."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return str(getattr(m, "content", ""))
    return ""


async def greeting_node(state: InterviewState) -> dict[str, Any]:
    """Initial node to greet the candidate and set the tone for the interview."""
    prompt_template = get_prompt("GREETING_PROMPT", GREETING_PROMPT)
    prompt = prompt_template.format(
        title=state.get("title", "this position"),
        duration=state.get("total_duration_minutes", 5),
        difficulty_level=state.get("difficulty_level", "medium"),
        experience_level=state.get("experience_level", "mid-level"),
        experience_reasoning=state.get("experience_reasoning", ""),
    )

    response = await llm.ainvoke(prompt)

    return {
        "messages": [AIMessage(content=response.content)],
        "current_skill_index": 0,
        "current_skill_depth": 0,
        "conversation_phase": "warmup_transition", 
        "nudge_count": 0,
    }


def _is_weak_answer(message: str) -> bool:
    """Check if the candidate's answer is weak/non-substantive (for depth rollback)."""
    if not message:
        return True
    words = message.strip().split()
    if len(words) < 5:
        return True
    non_answer_phrases = [
        "i don't know",
        "not sure",
        "no idea",
        "can you repeat",
        "i'm not familiar",
        "i have no experience",
        "i haven't used",
        "i can't answer",
        "i'm not sure",
        "pass",
        "next question",
    ]
    lower_msg = message.lower()
    return bool(any(phrase in lower_msg for phrase in non_answer_phrases))


def _build_closing_prompt(state: InterviewState) -> str:
    """Generate the closing prompt for the interview's wrap-up phase."""
    prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
    return prompt_template.format(
        title=state.get("title", "this position"),
        closing_phase="ask_questions",
        candidate_questions_response="",
    )


async def _closing_response(state: InterviewState) -> dict[str, Any]:
    """Invoke the LLM with the closing prompt and return the closing state update."""
    prompt = _build_closing_prompt(state)
    response = await llm.ainvoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "conversation_phase": "closing_ask_questions",
        "qa_phase": True,
        "qa_turns": 0,
        "nudge_count": 0,
    }


def _check_early_exit(last_human_message: str) -> bool:
    """Detect if the candidate initiated an early exit from the interview."""
    if not last_human_message:
        return False
    exit_phrases = ["goodbye", "end interview", "i'm done", "exit", "bye"]
    return any(ext in last_human_message.lower() for ext in exit_phrases)


def _check_silence_interrupt(last_human_message: str) -> bool:
    """Detect if the system injected a silence interrupt marker."""
    if not last_human_message:
        return False
    return "[system_interrupt] silence detected" in last_human_message.lower()


async def _handle_silence_interrupt() -> dict[str, Any]:
    """Handle a silence interrupt by prompting the candidate to check in."""
    prompt = get_prompt(
        "SILENCE_PROMPT",
        "The candidate has been silent for 60 seconds. Briefly check in to see "
        "if they are still there or if they need help. Keep it very short. "
        "Do not prompt the next interview question yet.",
    )
    response = await llm.ainvoke(prompt.format())
    return {
        "messages": [AIMessage(content=response.content)],
    }


async def _handle_time_warning(remaining_seconds: int) -> dict[str, Any]:
    """Send a time-remaining warning to the candidate."""
    prompt_template = get_prompt("TIME_WARNING_PROMPT", TIME_WARNING_PROMPT)
    prompt = prompt_template.format(remaining_minutes=remaining_seconds // 60)
    response = await llm.ainvoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "time_warning_given": True,
    }


async def _handle_warmup_transition(
    state: InterviewState,
    skills: list[dict[str, Any]],
    last_human_message: str,
    difficulty_level: str,
    experience_level: str,
) -> dict[str, Any]:
    """Transition from greeting to the first skill question."""
    first_skill = skills[0]["name"] if skills else "your background"
    prompt_template = get_prompt("WARMUP_TRANSITION_PROMPT", WARMUP_TRANSITION_PROMPT)
    prompt = prompt_template.format(
        title=state.get("title", "this position"),
        candidate_response=last_human_message,
        first_skill=first_skill,
        difficulty_level=difficulty_level,
        experience_level=experience_level,
    )
    response = await llm.ainvoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "current_skill_index": 0,
        "current_skill_depth": 1,
        "nudge_count": 0,
        "skills_remaining_count": max(0, len(skills)),
        "current_skill_question_count": 1,
        "conversation_phase": "interview",
    }


async def _handle_nudge_exhausted(
    state: InterviewState,
    skills: list[dict[str, Any]],
    current_idx: int,
    current_skill: str,
    remaining_seconds: int,
    elapsed_minutes: int,
    total_minutes: int,
    experience_level: str,
) -> dict[str, Any]:
    """Handle the case where nudge attempts are exhausted — move to next skill or close."""
    next_idx = current_idx + 1

    if next_idx >= len(skills) or not should_start_new_skill(state, remaining_seconds):
        return await _closing_response(state)

    new_skill = skills[next_idx]["name"]
    prompt_template = get_prompt("SKILL_TRANSITION_PROMPT", SKILL_TRANSITION_PROMPT)
    prompt = prompt_template.format(
        previous_skill=current_skill,
        new_skill=new_skill,
        elapsed_minutes=elapsed_minutes,
        total_minutes=total_minutes,
        experience_level=experience_level,
    )
    response = await llm.ainvoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "current_skill_index": next_idx,
        "current_skill_depth": 0,
        "previous_skill_name": current_skill,
        "should_nudge": False,
        "nudge_count": 0,
        "skills_remaining_count": max(0, len(skills) - next_idx),
        "current_skill_question_count": 1,
    }


async def _handle_nudge_attempt(
    state: InterviewState,
    messages: list[BaseMessage],
    nudge_count: int,
    last_human_message: str,
) -> dict[str, Any]:
    """Issue a nudge prompt to help the candidate answer."""
    last_ai_question = get_last_ai_message(messages)
    new_nudge_count = nudge_count + 1
    prompt_template = get_prompt("NUDGE_PROMPT", NUDGE_PROMPT)
    nudge_prompt = prompt_template.format(
        last_question=last_ai_question,
        candidate_response=last_human_message,
        nudge_count=new_nudge_count,
    )
    response = await llm.ainvoke(nudge_prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "should_nudge": True,
        "nudge_count": new_nudge_count,
        "current_skill_question_count": state.get("current_skill_question_count", 0) + 1,
    }


async def _handle_skill_transition(
    state: InterviewState,
    skills: list[dict[str, Any]],
    current_skill: str,
    next_idx: int,
    remaining_seconds: int,
    elapsed_minutes: int,
    total_minutes: int,
    experience_level: str,
) -> dict[str, Any] | None:
    """Handle transition to the next skill, or close if not enough time.

    Returns:
        Updated state dict, or None if a different path should be taken.
    """
    if next_idx >= len(skills):
        return await _closing_response(state)

    if not should_start_new_skill(state, remaining_seconds):
        return await _closing_response(state)

    new_skill = skills[next_idx]["name"]
    prompt = SKILL_TRANSITION_PROMPT.format(
        previous_skill=current_skill,
        new_skill=new_skill,
        elapsed_minutes=elapsed_minutes,
        total_minutes=total_minutes,
        experience_level=experience_level,
    )
    response = await llm.ainvoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "current_skill_index": next_idx,
        "current_skill_depth": 0,
        "previous_skill_name": current_skill,
        "should_nudge": False,
        "nudge_count": 0,
        "skills_remaining_count": max(0, len(skills) - next_idx),
        "current_skill_question_count": 1,
    }


async def _handle_followup_question(
    state: InterviewState,
    messages: list[BaseMessage],
    current_skill: str,
    next_depth: int,
    elapsed_minutes: int,
    total_minutes: int,
    current_idx: int,
    skills: list[dict[str, Any]],
    difficulty_level: str,
    experience_level: str,
) -> dict[str, Any]:
    """Generate a follow-up question at the current depth for the current skill."""
    transcript = ""
    for m in messages[-6:]:
        role = "Interviewer" if isinstance(m, AIMessage) else "Candidate"
        m_content = getattr(m, "content", "")
        transcript += f"{role}: {m_content}\n"

    prompt_template = get_prompt("ADAPTIVE_INTERVIEW_PROMPT", ADAPTIVE_INTERVIEW_PROMPT)
    prompt = prompt_template.format(
        title=state.get("title", "this position"),
        current_skill=current_skill,
        depth=next_depth,
        transcript=transcript,
        elapsed_minutes=elapsed_minutes,
        total_minutes=total_minutes,
        skills_covered=current_idx,
        total_skills=len(skills),
        difficulty_level=difficulty_level,
        experience_level=experience_level,
    )

    response = await llm.ainvoke(prompt)
    ai_resp_str = str(response.content)

    return {
        "messages": [AIMessage(content=ai_resp_str)],
        "current_skill_index": current_idx,
        "current_skill_depth": next_depth,
        "previous_skill_name": current_skill,
        "should_nudge": False,
        "nudge_count": 0,
        "skills_remaining_count": max(0, len(skills) - current_idx),
        "current_skill_question_count": state.get("current_skill_question_count", 0) + 1,
    }


async def interview_node(state: InterviewState) -> dict[str, Any]:
    """Core node that handles the adaptive interview logic, including skill progression and nudging."""
    if state.get("interview_locked") or state.get("finalization_started"):
        return {"conversation_phase": "finalizing"}

    state = with_default_pacing(state)

    messages = state.get("messages", [])
    skill_graph = state.get("skill_graph") or {}
    skills_raw = skill_graph.get("skills", []) if isinstance(skill_graph, dict) else []
    skills: list[dict[str, Any]] = skills_raw if isinstance(skills_raw, list) else []
    current_idx = int(state.get("current_skill_index", 0))
    current_depth = int(state.get("current_skill_depth", 0))
    difficulty_level = state.get("difficulty_level", "medium")
    experience_level = state.get("experience_level", "mid-level")
    nudge_count = state.get("nudge_count", 0)

    timing = compute_interview_timing(state)
    elapsed_seconds = timing["elapsed_seconds"]
    remaining_seconds = timing["remaining_seconds"]
    total_seconds = timing["total_seconds"]
    elapsed_minutes = elapsed_seconds // 60
    total_minutes = total_seconds // 60

    time_warning_given = state.get("time_warning_given", False)
    last_human_message = get_last_human_message(messages)

    # 1. Candidate initiated early exit
    if _check_early_exit(last_human_message):
        return {"conversation_phase": "closing_ask_questions"}

    # 1.5 Handle Silence Interrupt
    if _check_silence_interrupt(last_human_message):
        return await _handle_silence_interrupt()

    # 2. Time-based general warning
    time_warning_threshold = state.get("time_warning_threshold_seconds", 300)
    if (
        remaining_seconds <= time_warning_threshold
        and remaining_seconds > state.get("closing_threshold_seconds", 120)
        and not time_warning_given
    ):
        return await _handle_time_warning(remaining_seconds)

    # 3. Handle closing window constraints
    if timing["is_in_closing_window"] or current_idx >= len(skills):
        return await _closing_response(state)

    current_skill = skills[current_idx]["name"]

    # 4. Handle initial transition if coming directly from greeting
    if state.get("conversation_phase") == "warmup_transition":
        return await _handle_warmup_transition(
            state, skills, last_human_message, difficulty_level, experience_level
        )

    # 5. Handle Nudging (if candidate gave non-answer)
    answer_was_weak = _is_weak_answer(last_human_message)
    max_nudges = state.get("max_nudges_per_skill", 1)

    if answer_was_weak:
        if nudge_count >= max_nudges:
            return await _handle_nudge_exhausted(
                state, skills, current_idx, current_skill,
                remaining_seconds, elapsed_minutes, total_minutes, experience_level,
            )

        return await _handle_nudge_attempt(
            state, messages, nudge_count, last_human_message
        )

    # 6. Normal progression (Answer was substantial)
    max_depth = state.get("max_depth_per_skill", 2)
    next_depth = current_depth + 1
    next_idx = current_idx

    if not should_ask_followup(state, remaining_seconds):
        next_depth = max_depth + 1

    if next_depth > max_depth:
        next_depth = 0
        next_idx = current_idx + 1

        result = await _handle_skill_transition(
            state, skills, current_skill, next_idx,
            remaining_seconds, elapsed_minutes, total_minutes, experience_level,
        )
        if result is not None:
            return result

    return await _handle_followup_question(
        state, messages, current_skill, next_depth,
        elapsed_minutes, total_minutes, next_idx, skills,
        difficulty_level, experience_level,
    )


async def closing_node(state: InterviewState) -> dict[str, Any]:
    """Node that orchestrates the closing phase of the interview."""
    state = with_default_pacing(state)
    messages = state.get("messages", [])
    qa_turns = state.get("qa_turns", 0)
    
    timing = compute_interview_timing(state)
    remaining_seconds = timing["remaining_seconds"]
    max_qa_turns = state.get("max_qa_turns", 1)

    last_human_message = get_last_human_message(messages)

    phase = state.get("conversation_phase")
    
    if phase == "closing_final" or qa_turns >= max_qa_turns or remaining_seconds <= 0:
        prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
        prompt = prompt_template.format(
            title=state.get("title", "this position"),
            closing_phase="final_close",
            candidate_questions_response=last_human_message or "",
        )
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "conversation_phase": "finalizing",
        }

    # QA turn
    prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
    prompt = prompt_template.format(
        title=state.get("title", "this position"),
        closing_phase="qa_response",
        candidate_questions_response=last_human_message or "",
    )
    response = await llm.ainvoke(prompt)
    
    ai_resp_lower = str(response.content).lower()
    auto_complete = any(
        phrase in ai_resp_lower
        for phrase in [
            "interview is complete",
            "good luck with your application",
            "that wraps up",
        ]
    )

    return {
        "messages": [AIMessage(content=response.content)],
        "qa_turns": qa_turns + 1,
        "conversation_phase": "closing_final" if auto_complete else "closing_ask_questions",
    }


async def finalize_node(state: InterviewState) -> dict[str, Any]:
    """Final node that transitions the backend state to completed."""
    return {
        "is_completed": True,
        "conversation_phase": "completed",
        "closing_reason": "interview_completed",
        "interview_locked": True,
        "finalization_started": True,
    }
