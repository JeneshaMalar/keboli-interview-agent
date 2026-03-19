from langgraph.graph import StateGraph, END
from app.state import InterviewState
from app.node.skill_extraction_node import skill_extraction_node
from app.node.interview_node import greeting_node, interview_node

def create_interview_graph():
    """Factory function to create and compile the interview workflow graph. 
    It defines the nodes for skill extraction, greeting, and interviewing, 
    and sets up the conditional routing logic based on the state of the interview."""
    
    workflow = StateGraph(InterviewState)

    workflow.add_node("skill_extraction", skill_extraction_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("interview", interview_node)

    workflow.set_entry_point("skill_extraction")

    def router(state: InterviewState):
        if len(state.get("messages", [])) <= 1:
            return "greeting"
        return "interview"

    workflow.add_conditional_edges(
        "skill_extraction",
        router,
        {
            "greeting": "greeting",
            "interview": "interview"
        }
    )
    workflow.add_edge("greeting", END)
    workflow.add_edge("interview", END)
    
  
    
    return workflow.compile()

interview_agent = create_interview_graph()
