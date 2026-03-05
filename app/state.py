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
    
    messages: Annotated[List[dict], operator.add]
    current_skill_index: int
    current_skill_depth: int 
    total_duration_minutes: int
    elapsed_time_seconds: int
    
    # Conversation phase tracking
    # "greeting" -> "warmup" -> "interview" -> "closing"
    conversation_phase: str
    previous_skill_name: Optional[str]
    
    scores: List[SkillScore]
    final_recommendation: Optional[str]
    
    is_completed: bool
    should_nudge: bool
