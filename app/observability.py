"""Langfuse observability integration for LangChain tracing."""

import logging
import os

from langfuse.langchain import CallbackHandler

logger = logging.getLogger("keboli-observability")


def get_langfuse_handler() -> CallbackHandler | None:
    """Initialize a Langfuse callback handler if credentials are available.

    Returns:
        A CallbackHandler instance if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
        are set, otherwise None.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if public_key and secret_key:
        logger.info("Langfuse tracing enabled.")
        return CallbackHandler()

    logger.warning("Langfuse credentials missing. Tracing disabled.")
    return None


langfuse_handler = get_langfuse_handler()
