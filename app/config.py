import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    KEBOLI_BACKEND_URL = os.getenv("KEBOLI_BACKEND_URL", "http://localhost:8000")
    LLM_MODEL = "llama-3.3-70b-versatile"

    # LiveKit Cloud
    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

    # Deepgram (used by LiveKit plugins — set via env automatically)
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    # BeyondPresence
    BEY_API_KEY = os.getenv("BEY_API_KEY")

config = Config()
