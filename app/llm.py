"""LLM factory for the interview agent."""

from langchain_groq import ChatGroq

from app.config import settings


def get_llm(temperature: float = 0.5) -> ChatGroq:
    """Create a ChatGroq LLM instance using the configured API key and model.

    Args:
        temperature: Sampling temperature for the LLM.

    Returns:
        A configured ChatGroq instance.
    """
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,  
        model_name=settings.LLM_MODEL,
        temperature=temperature,
    )


llm = get_llm()
