from app.state import InterviewState

def with_default_pacing(state: InterviewState) -> InterviewState:
    def _get(key, default):
        val = state.get(key)
        return default if val is None else val

    return {
        **state,
        "active_interview_ratio": _get("active_interview_ratio", 0.9),
        "closing_phase_ratio": _get("closing_phase_ratio", 0.07),
        "finalization_buffer_seconds": _get("finalization_buffer_seconds", 30),
        "max_nudges_per_skill": _get("max_nudges_per_skill", 1),
        "max_depth_per_skill": _get("max_depth_per_skill", 2),
        "max_qa_turns": _get("max_qa_turns", 1),
        "min_remaining_time_for_new_skill_seconds": _get("min_remaining_time_for_new_skill_seconds", 180),
        "min_remaining_time_for_followup_seconds": _get("min_remaining_time_for_followup_seconds", 90),
        "time_warning_threshold_seconds": _get("time_warning_threshold_seconds", 300),
        "closing_threshold_seconds": _get("closing_threshold_seconds", 120),
    }

def compute_interview_timing(state: InterviewState) -> dict:
    state = with_default_pacing(state)
    total_seconds = state.get("total_duration_minutes", 30) * 60
    elapsed_seconds = state.get("elapsed_time_seconds", 0)
    remaining_seconds = max(0, total_seconds - elapsed_seconds)

    active_ratio = state.get("active_interview_ratio")
    closing_ratio = state.get("closing_phase_ratio")
    finalization_buffer_seconds = state.get("finalization_buffer_seconds")

    active_cutoff = int(total_seconds * active_ratio)
    closing_window = int(total_seconds * closing_ratio)

    active_interview_ends_at = min(active_cutoff, total_seconds - finalization_buffer_seconds)
    closing_starts_at = max(0, active_interview_ends_at - closing_window)

    return {
        "total_seconds": total_seconds,
        "elapsed_seconds": elapsed_seconds,
        "remaining_seconds": remaining_seconds,
        "active_interview_ends_at": active_interview_ends_at,
        "closing_starts_at": closing_starts_at,
        "is_in_closing_window": elapsed_seconds >= closing_starts_at,
        "is_hard_timeout": elapsed_seconds >= total_seconds,
        "is_near_end": remaining_seconds <= state.get("time_warning_threshold_seconds"),
    }

def should_start_new_skill(state: InterviewState, remaining_seconds: int) -> bool:
    state = with_default_pacing(state)
    min_time = state.get("min_remaining_time_for_new_skill_seconds")
    return remaining_seconds >= min_time

def should_ask_followup(state: InterviewState, remaining_seconds: int) -> bool:
    state = with_default_pacing(state)
    min_time = state.get("min_remaining_time_for_followup_seconds")
    return remaining_seconds >= min_time
