import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    KEBOLI_BACKEND_URL = os.getenv("MAIN_BACKEND_URL", "http://localhost:8000")
    LLM_MODEL = "llama-3.3-70b-versatile"

    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    BEY_API_KEY = os.getenv("BEY_API_KEY")

config = Config()
