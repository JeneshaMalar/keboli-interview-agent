"""Custom exception hierarchy for the Keboli Interview Agent."""

from typing import Any


class AppError(Exception):
    """Base exception for all application errors.

    Args:
        message: Human-readable error description.
        status_code: HTTP status code to return to the client.
        error_code: Machine-readable error identifier.
        details: Optional structured data providing additional context.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    def __init__(
        self, resource: str = "Resource", resource_id: str | None = None
    ) -> None:
        detail = f"{resource} not found"
        if resource_id:
            detail = f"{resource} with id '{resource_id}' not found"
        super().__init__(message=detail, status_code=404, error_code="NOT_FOUND")


class ValidationError(AppError):
    """Raised when input data fails business-rule validation."""

    def __init__(
        self, message: str = "Validation failed", field: str | None = None
    ) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details={"field": field} if field else {},
        )


class ExternalServiceError(AppError):
    """Raised when a call to an external service fails."""

    def __init__(
        self, service_name: str, message: str = "External service error"
    ) -> None:
        super().__init__(
            message=message,
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service_name},
        )


class InterviewError(AppError):
    """Raised when the interview pipeline encounters an error."""

    def __init__(self, session_id: str, message: str = "Interview error") -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="INTERVIEW_ERROR",
            details={"session_id": session_id},
        )
