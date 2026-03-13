import os
import logging
from langfuse.callback import CallbackHandler

logger = logging.getLogger("keboli-observability")

def get_langfuse_handler():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if public_key and secret_key:
        logger.info("Langfuse tracing enabled.")
        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
    
    logger.warning("Langfuse credentials missing. Tracing disabled.")
    return None

langfuse_handler = get_langfuse_handler()
