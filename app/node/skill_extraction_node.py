from app.state import InterviewState
from app.llm import llm
from app.keboli_client import keboli_client
from app.prompt_manager import SKILL_EXTRACTION_PROMPT, SkillGraph
from langchain_core.prompts import ChatPromptTemplate

async def skill_extraction_node(state: InterviewState):
    if state.get("skill_graph"):
        return state

    assessment_id = state.get("assessment_id")
    if not assessment_id:
        return state
    
    print(f"Fetching assessment {assessment_id} for skill extraction...")
    assessment = await keboli_client.get_assessment(assessment_id)
    skill_graph = assessment.get("skill_graph")

    if not skill_graph:
        print(f"Extracting skills for assessment {assessment_id} from JD...")
        
        prompt = ChatPromptTemplate.from_template(SKILL_EXTRACTION_PROMPT)
        structured_llm = llm.with_structured_output(SkillGraph)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"job_description": assessment.get("job_description")})
        
        skill_graph = result.model_dump()
        
        await keboli_client.update_assessment_skills(assessment_id, skill_graph)
    
    return {
        "skill_graph": skill_graph,
        "title": assessment.get("title"),
        "job_description": assessment.get("job_description"),
        "total_duration_minutes": assessment.get("duration_minutes", 30)
    }
