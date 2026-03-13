from app.state import InterviewState
from app.llm import llm
from app.keboli_client import keboli_client
from app.prompt_manager import SKILL_EXTRACTION_PROMPT, SkillGraph, get_prompt
from langchain_core.prompts import ChatPromptTemplate


async def skill_extraction_node(state: InterviewState):
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
    
    print(f"Fetching assessment {assessment_id} for skill extraction...")
    assessment = await keboli_client.get_assessment(assessment_id)
    skill_graph = assessment.get("skill_graph")
    difficulty_level = assessment.get("difficulty_level", "medium")

    if not skill_graph:
        print(f"Extracting skills for assessment {assessment_id} from JD (difficulty={difficulty_level})...")
        print("LLM will analyze JD to determine experience level...")
        
        prompt_template = get_prompt("SKILL_EXTRACTION_PROMPT", SKILL_EXTRACTION_PROMPT)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        structured_llm = llm.with_structured_output(SkillGraph)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "job_description": assessment.get("job_description"),
            "difficulty_level": difficulty_level,
        })
        
        skill_graph = result.model_dump()
        
        print(f"LLM detected experience level: {skill_graph.get('experience_level')} "
              f"— Reason: {skill_graph.get('experience_reasoning')}")
        
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

