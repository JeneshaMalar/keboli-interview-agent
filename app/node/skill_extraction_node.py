from app.state import InterviewState
from app.llm import llm
from app.keboli_client import keboli_client
from app.prompt_manager import SKILL_EXTRACTION_PROMPT, SkillGraph, get_prompt
from langchain_core.prompts import ChatPromptTemplate


async def skill_extraction_node(state: InterviewState):
    """Node responsible for extracting the skill graph from the job description. If the skill graph already exists in the state, 
    it will skip extraction and just return the existing data. 
    Otherwise, it will call the LLM to generate the skill graph and update the backend via the KeboliClient."""
    if state.get("skill_graph"):
        assessment_id = state.get("assessment_id")
        if assessment_id and not state.get("experience_level"):
            assessment = await keboli_client.get_assessment(assessment_id)
            skill_graph = assessment.get("skill_graph", {})
            difficulty_level = assessment.get("difficulty_level", "medium")
            experience_level = skill_graph.get("experience_level", "mid-level")
            experience_reasoning = skill_graph.get("experience_reasoning", "")
            return {
                **{k: v for k, v in state.items() if k != "messages"},
                "messages": [],
                "difficulty_level": difficulty_level,
                "experience_level": experience_level,
                "experience_reasoning": experience_reasoning,
            }
        return state

    assessment_id = state.get("assessment_id")
    if not assessment_id:
        return state
    
    assessment = await keboli_client.get_assessment(assessment_id)
    skill_graph = assessment.get("skill_graph")
    difficulty_level = assessment.get("difficulty_level", "medium")

    if not skill_graph:
        
        prompt_template = get_prompt("SKILL_EXTRACTION_PROMPT", SKILL_EXTRACTION_PROMPT)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        structured_llm = llm.with_structured_output(SkillGraph)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "job_description": assessment.get("job_description"),
            "difficulty_level": difficulty_level,
        })
        
        skill_graph = result.model_dump()
        
        
        
        await keboli_client.update_assessment_skills(assessment_id, skill_graph)
    
    experience_level = skill_graph.get("experience_level", "mid-level")
    experience_reasoning = skill_graph.get("experience_reasoning", "")
    
    return {
        "skill_graph": skill_graph,
        "title": assessment.get("title"),
        "job_description": assessment.get("job_description"),
        "total_duration_minutes": assessment.get("duration_minutes", 30),
        "difficulty_level": difficulty_level,
        "experience_level": experience_level,
        "experience_reasoning": experience_reasoning,
    }

