"""Typed state definitions for the LangGraph interview workflow."""

import operator
from typing import Annotated, TypedDict


class SkillScore(TypedDict):
    """Individual skill evaluation result."""

    skill: str
    score: int
    feedback: str


class InterviewState(TypedDict):
    """State object passed through the interview LangGraph nodes.

    Tracks the full conversation lifecycle: skill extraction, greeting,
    interview progression, timing, and completion.
    """

    session_id: str
    assessment_id: str
    title: str
    job_description: str

    skill_graph: dict[str, object] | None

    difficulty_level: str | None
    experience_level: str | None
    experience_reasoning: str | None

    messages: Annotated[list[dict[str, str]], operator.add]
    current_skill_index: int
    current_skill_depth: int
    total_duration_minutes: int
    elapsed_time_seconds: int

    conversation_phase: str  # "greeting" -> "warmup" -> "interview" -> "closing"
    previous_skill_name: str | None

    scores: list[SkillScore]
    final_recommendation: str | None

    is_completed: bool
    should_nudge: bool
    nudge_count: int
    closing_phase: str | None
    closing_reason: str | None

    time_warning_given: bool
    qa_phase: bool
    qa_turns: int
