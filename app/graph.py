"""Interview workflow graph — Compiles the LangGraph state machine."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.node.interview_node import (
    closing_node,
    finalize_node,
    greeting_node,
    interview_node,
)
from app.node.skill_extraction_node import skill_extraction_node
from app.state import InterviewState


def create_interview_graph() -> CompiledStateGraph:  
    """Factory function to create and compile the interview workflow graph.

    Defines nodes for skill extraction, greeting, and interviewing,
    and sets up conditional routing based on the interview state.

    Returns:
        A compiled LangGraph state machine ready for invocation.
    """
    workflow = StateGraph(InterviewState)

    workflow.add_node("skill_extraction", skill_extraction_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("interview", interview_node)
    workflow.add_node("closing", closing_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("skill_extraction")

    def router(state: InterviewState) -> str:
        """Route to the appropriate node based on the current conversation phase.

        Args:
            state: The current interview state.

        Returns:
            The name of the next node to execute.
        """
        phase = state.get("conversation_phase", "warmup")
        if phase == "warmup":
            return "greeting"
        if phase in ["interview", "followup", "transition", "warmup_transition"]:
            return "interview"
        if phase in ["closing_ask_questions", "closing_final", "closing_wrapup"]:
            return "closing"
        if state.get("is_completed") or phase in ["finalizing", "completed"]:
            return "finalize"
        return "interview"

    workflow.add_conditional_edges(
        "skill_extraction",
        router,
        {
            "greeting": "greeting",
            "interview": "interview",
            "closing": "closing",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("greeting", END)
    workflow.add_edge("interview", END)
    workflow.add_edge("closing", END)
    workflow.add_edge("finalize", END)

    return workflow.compile()


interview_agent = create_interview_graph()
