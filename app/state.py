from typing import Annotated, List, Optional, TypedDict
import operator

class SkillScore(TypedDict):
    skill: str
    score: int
    feedback: str

class InterviewState(TypedDict):
    session_id: str
    assessment_id: str
    title: str
    job_description: str
    
    skill_graph: Optional[dict]
    
    difficulty_level: Optional[str]  
    experience_level: Optional[str]  # LLM-determined from JD: "fresher", "junior", "mid-level", "senior", "lead"
    experience_reasoning: Optional[str]  # LLM's reasoning for the experience level determination
    
    messages: Annotated[List[dict], operator.add]
    current_skill_index: int
    current_skill_depth: int 
    total_duration_minutes: int
    elapsed_time_seconds: int
    
    
    conversation_phase: str # "greeting" -> "warmup" -> "interview" -> "closing"
    previous_skill_name: Optional[str]
    
    scores: List[SkillScore]
    final_recommendation: Optional[str]
    
    is_completed: bool
    should_nudge: bool
    nudge_count: int  
    closing_phase: Optional[str]  # "ask_questions" -> "final_close" for two-phase closing
    closing_reason: Optional[str]  # Why the interview ended: "interview_completed", "time_limit", "candidate_exit"
    
    time_warning_given: bool   # True once the 5-min warning has been issued
    qa_phase: bool             # True when we are in the 2-min grace Q&A buffer
    qa_turns: int              # Number of Q&A turns used during grace buffer (max 2)
