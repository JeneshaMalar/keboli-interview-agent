from langchain_groq import ChatGroq
from app.config import config

def get_llm(temperature=0.5):
    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=temperature
    )

llm = get_llm()
