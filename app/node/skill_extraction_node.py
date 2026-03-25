from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.keboli_client import keboli_client
from app.llm import llm
from app.prompt_manager import SKILL_EXTRACTION_PROMPT, SkillGraph
from app.state import InterviewState


async def skill_extraction_node(state: InterviewState) -> dict[str, Any]:
    """Node responsible for extracting the skill graph from the job description. If the skill graph already exists in the state,
    it will skip extraction and just return the existing data.
    Otherwise, it will call the LLM to generate the skill graph and update the backend via the KeboliClient."""
    if state.get("skill_graph"):
        return {
            "skill_graph": state["skill_graph"],
            "current_skill_index": 0,
            "current_skill_depth": 0,
        }

    assessment_id = state.get("assessment_id")
    if not assessment_id:
        return {
            "skill_graph": {},
            "current_skill_index": 0,
            "current_skill_depth": 0,
        }

    try:
        assessment = await keboli_client.get_assessment(assessment_id)

        job_description = assessment.get("job_description")
        if not job_description:
            return {
                "skill_graph": {},
                "current_skill_index": 0,
                "current_skill_depth": 0,
            }

        difficulty_level = assessment.get("difficulty_level", "medium")

        prompt = ChatPromptTemplate.from_template(SKILL_EXTRACTION_PROMPT)
        structured_llm = llm.with_structured_output(SkillGraph)

        chain = prompt | structured_llm
        result = await chain.ainvoke(
            {
                "job_description": job_description,
                "difficulty_level": difficulty_level,
            }
        )

        skill_graph = result if isinstance(result, dict) else result.model_dump()

        await keboli_client.update_assessment_skills(assessment_id, skill_graph)

    except Exception:
        skill_graph = {}

    experience_level = skill_graph.get("experience_level", "mid-level")
    experience_reasoning = skill_graph.get("experience_reasoning", "")

    return {
        "skill_graph": skill_graph,
        "job_description": assessment.get("job_description"),
        "total_duration_minutes": assessment.get("duration_minutes", 30),
        "difficulty_level": difficulty_level,
        "experience_level": experience_level,
        "experience_reasoning": experience_reasoning,
    }
