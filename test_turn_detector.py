import logging  # noqa: I001
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test():
    try:
        logger.info("Initializing MultilingualModel...")
        model = MultilingualModel()  # noqa: F841
        logger.info("MultilingualModel initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize MultilingualModel: {e}", exc_info=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
