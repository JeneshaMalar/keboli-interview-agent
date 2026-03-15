from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.graph import interview_agent
from app.llm import llm
from app.prompt_manager import SKILL_EXTRACTION_PROMPT, SkillGraph
from app.keboli_client import keboli_client
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger("keboli-fastapi")

app = FastAPI(title="Keboli Interview Agent")

class InterviewTurnRequest(BaseModel):
    session_id: str
    assessment_id: str
    last_message: Optional[str] = None
    state: Optional[dict] = None

class SkillGraphRequest(BaseModel):
    assessment_id: str

@app.post("/generate-skill-graph")
async def generate_skill_graph(request: SkillGraphRequest):
    try:
        assessment = await keboli_client.get_assessment(request.assessment_id)
        
        existing_graph = assessment.get("skill_graph")
        if existing_graph:
            return {"status": "exists", "skill_graph": existing_graph}
        
        job_description = assessment.get("job_description")
        if not job_description:
            raise HTTPException(status_code=400, detail="Assessment has no job description")
        
        difficulty_level = assessment.get("difficulty_level", "medium")
        
        logger.info(f"Generating skill graph for assessment {request.assessment_id} "
                     f"(difficulty={difficulty_level}). LLM will analyze JD to determine experience level.")
                     
        await keboli_client.post_log({
            "level": "INFO",
            "service": "interview_agent",
            "component": "skill_graph",
            "event_type": "graph_generation_started",
            "assessment_id": request.assessment_id,
            "message": f"Started skill graph generation for assessment {request.assessment_id} (difficulty={difficulty_level})"
        })
        
        prompt = ChatPromptTemplate.from_template(SKILL_EXTRACTION_PROMPT)
        structured_llm = llm.with_structured_output(SkillGraph)
        chain = prompt | structured_llm
        
        result = await chain.ainvoke({
            "job_description": job_description,
            "difficulty_level": difficulty_level,
        })
        
        skill_graph = result.model_dump()
        
        logger.info(f"Skill graph generated — experience_level={skill_graph.get('experience_level')}, "
                     f"reason={skill_graph.get('experience_reasoning')}, "
                     f"{len(skill_graph.get('skills', []))} skills extracted")
                     
        await keboli_client.post_log({
            "level": "INFO",
            "service": "interview_agent",
            "component": "skill_graph",
            "event_type": "graph_generation_completed",
            "assessment_id": request.assessment_id,
            "message": "Skill graph generated successfully",
            "details": {
                "experience_level": skill_graph.get('experience_level'),
                "skills_extracted": len(skill_graph.get('skills', []))
            }
        })
        
        await keboli_client.update_assessment_skills(request.assessment_id, skill_graph)
        
        return {"status": "generated", "skill_graph": skill_graph}
        
    except HTTPException:
        raise
    except Exception as e:
        await keboli_client.post_log({
            "level": "ERROR",
            "service": "interview_agent",
            "component": "skill_graph",
            "event_type": "graph_generation_failed",
            "assessment_id": request.assessment_id,
            "message": f"Error generating skill graph: {str(e)}",
            "error_stack": str(e)
        })
        logger.error(f"Error generating skill graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate skill graph: {str(e)}")


@app.post("/chat")
async def chat(request: InterviewTurnRequest):
    if request.state:
        state = request.state
        msg_objs = []
        for m in state.get("messages", []):
            if m["role"] == "human":
                msg_objs.append(HumanMessage(content=m["content"]))
            else:
                msg_objs.append(AIMessage(content=m["content"]))
        state["messages"] = msg_objs
    else:
        state = {
            "session_id": request.session_id,
            "assessment_id": request.assessment_id,
            "messages": [],
            "current_skill_index": 0,
            "current_skill_depth": 0,
            "elapsed_time_seconds": 0,
            "is_completed": False,
            "conversation_phase": "greeting",
            "previous_skill_name": None,
            "nudge_count": 0,
            "closing_phase": None,
            "closing_reason": None,
            "time_warning_given": False,
            "qa_phase": False,
            "qa_turns": 0,
        }

    if request.last_message:
        state["messages"].append(HumanMessage(content=request.last_message))

    try:
        final_state = await interview_agent.ainvoke(state)
        
        last_ai_message = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                last_ai_message = m.content
                break
        serializable_messages = []
        for m in final_state["messages"]:
            if isinstance(m, HumanMessage):
                serializable_messages.append({"role": "human", "content": m.content})
            elif isinstance(m, AIMessage):
                serializable_messages.append({"role": "ai", "content": m.content})

        response_payload = {
            "response": last_ai_message,
            "is_completed": final_state.get("is_completed", False),
            "state": {
                **{k: v for k, v in final_state.items() if k != "messages"},
                "messages": serializable_messages
            }
        }
        
        await keboli_client.post_log({
            "level": "INFO",
            "service": "interview_agent",
            "component": "chat",
            "event_type": "chat_turn_completed",
            "session_id": request.session_id,
            "assessment_id": request.assessment_id,
            "message": "Processed interview chat turn",
            "details": {"is_completed": final_state.get("is_completed", False)}
        })
        
        return response_payload
    except Exception as e:
        await keboli_client.post_log({
            "level": "ERROR",
            "service": "interview_agent",
            "component": "chat",
            "event_type": "chat_turn_failed",
            "session_id": request.session_id,
            "assessment_id": request.assessment_id,
            "message": f"Error processing chat turn: {str(e)}",
            "error_stack": str(e)
        })
        print(f"Agent Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

