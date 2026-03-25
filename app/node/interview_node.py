from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

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
        "conversation_phase": "warmup",
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


async def interview_node(state: InterviewState) -> dict[str, Any]:
    """Core node that handles the adaptive interview logic, including skill progression, nudging, and phase transitions."""
    messages = state.get("messages", [])
    skill_graph = state.get("skill_graph") or {}
    skills_raw = skill_graph.get("skills", []) if isinstance(skill_graph, dict) else []
    skills: list[dict[str, Any]] = skills_raw if isinstance(skills_raw, list) else []
    current_idx = int(state.get("current_skill_index", 0))
    current_depth = int(state.get("current_skill_depth", 0))
    phase = state.get("conversation_phase", "interview")
    difficulty_level = state.get("difficulty_level", "medium")
    experience_level = state.get("experience_level", "mid-level")
    nudge_count = state.get("nudge_count", 0)

    last_human_message = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human_message = m.content
            break

    if phase == "closing_ask_questions":
        qa_turns = state.get("qa_turns", 0)
        remaining_time_secs = (
            state.get("total_duration_minutes", 30) * 60
        ) - state.get("elapsed_time_seconds", 0)

        if qa_turns >= 2 or remaining_time_secs <= 0:
            prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
            prompt = prompt_template.format(
                title=state.get("title", "this position"),
                closing_phase="final_close",
                candidate_questions_response=last_human_message or "",
            )
            response = await llm.ainvoke(prompt)
            return {
                "messages": [AIMessage(content=response.content)],
                "is_completed": True,
                "conversation_phase": "completed",
                "closing_reason": "interview_completed",
            }

        prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
        prompt = prompt_template.format(
            title=state.get("title", "this position"),
            closing_phase="qa_response",
            candidate_questions_response=last_human_message or "",
        )
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "qa_phase": True,
            "qa_turns": qa_turns + 1,
            "conversation_phase": "closing_ask_questions",
        }

    if last_human_message and any(
        ext in last_human_message.lower()
        for ext in ["goodbye", "end interview", "i'm done", "exit", "bye"]
    ):
        prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
        prompt = prompt_template.format(
            title=state.get("title", "this position"),
            closing_phase="final_close",
            candidate_questions_response="",
        )
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "is_completed": True,
            "closing_reason": "candidate_exit",
        }

    elapsed_minutes = state.get("elapsed_time_seconds", 0) // 60
    total_minutes = state.get("total_duration_minutes", 30)
    remaining_minutes = max(0, total_minutes - elapsed_minutes)
    time_warning_given = state.get("time_warning_given", False)
    qa_phase = state.get("qa_phase", False)

    if 120 < remaining_minutes * 60 <= 300 and not time_warning_given and not qa_phase:
        prompt_template = get_prompt("TIME_WARNING_PROMPT", TIME_WARNING_PROMPT)
        prompt = prompt_template.format(remaining_minutes=remaining_minutes)
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "time_warning_given": True,
        }

    if (
        remaining_minutes * 60 <= 120
        and not qa_phase
        and phase != "closing_ask_questions"
    ):
        prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
        prompt = prompt_template.format(
            title=state.get("title", "this position"),
            closing_phase="ask_questions",
            candidate_questions_response="",
        )
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "conversation_phase": "closing_ask_questions",
            "qa_phase": True,
            "qa_turns": 0,
        }

    def _start_closing() -> str:
        """Helper function to generate the closing prompt and transition to the closing phase."""
        prompt_template = get_prompt("CLOSING_PROMPT", CLOSING_PROMPT)
        return prompt_template.format(
            title=state.get("title", "this position"),
            closing_phase="ask_questions",
            candidate_questions_response="",
        )

    if elapsed_minutes >= total_minutes:
        prompt = _start_closing()
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "conversation_phase": "closing_ask_questions",
        }

    if current_idx >= len(skills):
        prompt = _start_closing()
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "conversation_phase": "closing_ask_questions",
        }

    current_skill = skills[current_idx]["name"]

    if phase == "warmup":
        first_skill = skills[0]["name"] if skills else "your background"
        prompt_template = get_prompt(
            "WARMUP_TRANSITION_PROMPT", WARMUP_TRANSITION_PROMPT
        )
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
            "conversation_phase": "interview",
            "current_skill_index": 0,
            "current_skill_depth": 1,
            "nudge_count": 0,
        }

    if last_human_message and (
        len(last_human_message.split()) < 5
        or any(
            phrase in last_human_message.lower()
            for phrase in [
                "i don't know",
                "not sure",
                "no idea",
                "can you repeat",
                "i'm not familiar",
            ]
        )
    ):
        if nudge_count >= 2:
            next_depth = 0
            next_idx = current_idx + 1

            if next_idx >= len(skills):
                prompt = _start_closing()
                response = await llm.ainvoke(prompt)
                return {
                    "messages": [AIMessage(content=response.content)],
                    "conversation_phase": "closing_ask_questions",
                    "nudge_count": 0,
                }

            new_skill = skills[next_idx]["name"]
            prompt_template = get_prompt(
                "SKILL_TRANSITION_PROMPT", SKILL_TRANSITION_PROMPT
            )
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
                "current_skill_depth": next_depth,
                "previous_skill_name": current_skill,
                "should_nudge": False,
                "nudge_count": 0,
            }

        last_ai_question = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                last_ai_question = m.content
                break

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
        }

    answer_was_weak = _is_weak_answer(last_human_message)

    if answer_was_weak:
        next_depth = max(0, current_depth - 1) if current_depth > 0 else 0
        next_idx = current_idx
        transitioning = False
    else:
        next_depth = current_depth + 1
        next_idx = current_idx
        transitioning = False

    state.get("previous_skill_name", "")

    if next_depth > 2:
        next_depth = 0
        next_idx = current_idx + 1
        transitioning = True

    if next_idx >= len(skills):
        prompt = _start_closing()
        response = await llm.ainvoke(prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "conversation_phase": "closing_ask_questions",
            "nudge_count": 0,
        }

    if transitioning and next_idx < len(skills):
        new_skill = skills[next_idx]["name"]
        prompt = SKILL_TRANSITION_PROMPT.format(
            previous_skill=current_skill,
            new_skill=new_skill,
            elapsed_minutes=elapsed_minutes,
            total_minutes=total_minutes,
            experience_level=experience_level,
        )
    else:
        transcript = ""
        for m in messages[-6:]:
            role = "Interviewer" if isinstance(m, AIMessage) else "Candidate"
            m_content = getattr(m, "content", "")
            transcript += f"{role}: {m_content}\n"

        prompt_template = get_prompt(
            "ADAPTIVE_INTERVIEW_PROMPT", ADAPTIVE_INTERVIEW_PROMPT
        )
        prompt = prompt_template.format(
            title=state.get("title", "this position"),
            current_skill=current_skill,
            depth=next_depth if not answer_was_weak else current_depth,
            transcript=transcript,
            elapsed_minutes=elapsed_minutes,
            total_minutes=total_minutes,
            skills_covered=current_idx,
            total_skills=len(skills),
            difficulty_level=difficulty_level,
            experience_level=experience_level,
        )

    response = await llm.ainvoke(prompt)

    ai_resp_content = response.content
    ai_resp_str = (
        str(ai_resp_content)
        if isinstance(ai_resp_content, str)
        else str(ai_resp_content[0])
        if isinstance(ai_resp_content, list) and ai_resp_content
        else ""
    )
    ai_resp_lower = ai_resp_str.lower()
    auto_complete = any(
        phrase in ai_resp_lower
        for phrase in [
            "interview is complete",
            "all the questions i have",
            "good luck with your application",
            "that wraps up",
        ]
    )

    return {
        "messages": [AIMessage(content=ai_resp_str)],
        "current_skill_index": next_idx,
        "current_skill_depth": next_depth,
        "previous_skill_name": current_skill,
        "should_nudge": False,
        "nudge_count": 0,
        "is_completed": auto_complete,
    }
