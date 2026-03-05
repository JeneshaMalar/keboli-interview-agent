from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.graph import interview_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Keboli Interview Agent")

class InterviewTurnRequest(BaseModel):
    session_id: str
    assessment_id: str
    last_message: Optional[str] = None
    state: Optional[dict] = None

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

        return {
            "response": last_ai_message,
            "is_completed": final_state.get("is_completed", False),
            "state": {
                **{k: v for k, v in final_state.items() if k != "messages"},
                "messages": serializable_messages
            }
        }
    except Exception as e:
        print(f"Agent Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
