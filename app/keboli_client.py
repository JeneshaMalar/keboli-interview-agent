"""HTTP client for communicating with the Keboli Main Backend API."""

import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger("keboli-interview-client")


class KeboliClient:
    """Client for interacting with the Keboli backend API.

    Provides methods to fetch assessment details, update skill graphs,
    append interview transcripts, and post logs. All methods are asynchronous.
    """

    def __init__(self, base_url: str = settings.MAIN_BACKEND_URL) -> None:
        self.base_url = base_url

    async def get_assessment(self, assessment_id: str) -> dict[str, Any]:
        """Fetch assessment details from the backend using the assessment ID.

        Args:
            assessment_id: The unique identifier for the assessment.

        Returns:
            A dictionary containing the assessment metadata.

        Raises:
            httpx.HTTPStatusError: If the backend returns a non-2xx response.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/assessment/{assessment_id}"
            )
            response.raise_for_status()
            return response.json()  

    async def update_assessment_skills(
        self, assessment_id: str, skill_graph: dict[str, object]
    ) -> dict[str, Any]:
        """Update the skill graph for a specific assessment in the backend.

        Args:
            assessment_id: The unique identifier for the assessment.
            skill_graph: The structured skill data to persist.

        Returns:
            The backend's response confirming the update.

        Raises:
            httpx.HTTPStatusError: If the backend returns a non-2xx response.
        """
        payload = {"skill_graph": skill_graph}
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.base_url}/api/assessment/{assessment_id}/skills",
                json=payload,
            )
            response.raise_for_status()
            return response.json()  

    async def append_transcript(
        self, session_id: str, role: str, content: str
    ) -> dict[str, Any]:
        """Append a new entry to the interview transcript for a given session.

        Args:
            session_id: The unique identifier for the interview session.
            role: The speaker role (e.g., 'interviewer' or 'candidate').
            content: The text content of the transcript entry.

        Returns:
            The backend's response confirming the append.

        Raises:
            httpx.HTTPStatusError: If the backend returns a non-2xx response.
        """
        payload = {"role": role, "content": content}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/livekit/transcript/{session_id}/append",
                json=payload,
            )
            response.raise_for_status()
            return response.json()  

    async def complete_session(self, session_id: str) -> dict[str, Any]:
        """Mark an interview session as complete in the backend.

        This may trigger post-interview processing such as evaluation.

        Args:
            session_id: The unique identifier for the interview session.

        Returns:
            The backend's response confirming the completion.

        Raises:
            httpx.HTTPStatusError: If the backend returns a non-2xx response.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/livekit/session/{session_id}/complete",
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json() 

    async def post_log(self, log_data: dict[str, object]) -> dict[str, Any] | None:
        """Post a log entry to the backend logging endpoint.

        This method is fire-and-forget; exceptions are caught and logged
        without propagating to the caller.

        Args:
            log_data: Structured log entry matching the backend's LogCreate schema.

        Returns:
            The backend's response on success, or None on failure.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/logs/",
                    json=log_data,
                )
                response.raise_for_status()
                return response.json() 
            except Exception as e:
                logger.error("Failed to post log to backend: %s", e)
                return None


keboli_client = KeboliClient()
