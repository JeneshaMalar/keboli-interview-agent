import httpx
from app.config import config

class KeboliClient:
    """Client for interacting with the Keboli backend API. Provides methods to fetch assessment details,
update skill graphs, append interview transcripts, and post logs. All methods are asynchronous and handle HTTP errors gracefully."""
    def __init__(self, base_url: str = config.KEBOLI_BACKEND_URL):
        self.base_url = base_url

    async def get_assessment(self, assessment_id: str):
        """Fetch assessment details from the backend using the assessment ID."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/assessment/{assessment_id}")
            response.raise_for_status()
            return response.json()

    async def update_assessment_skills(self, assessment_id: str, skill_graph: dict):
        """Update the skill graph for a specific assessment in the backend."""
        payload = {"skill_graph": skill_graph}
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.base_url}/api/assessment/{assessment_id}/skills", 
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def append_transcript(self, session_id: str, role: str, content: str):
        """Append a new entry to the interview transcript for a given session, specifying the role (e.g., 'interviewer' or 'candidate') and the content of the transcript entry."""
        payload = {"role": role, "content": content}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/livekit/transcript/{session_id}/append",
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def complete_session(self, session_id: str):
        """Mark an interview session as complete in the backend, which may trigger post-interview processing such as report generation."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/livekit/session/{session_id}/complete",
                timeout=300.0
            )
            response.raise_for_status()
            return response.json()

    async def post_log(self, log_data: dict):
        """Post a log entry to the backend logging endpoint. This method is designed to be fire-and-forget, and any exceptions are caught and logged without raising to the caller."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/logs/",
                    json=log_data
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                import logging
                logging.getLogger("keboli-fastapi").error(f"Failed to post log to backend: {e}")
                return None

keboli_client = KeboliClient()
