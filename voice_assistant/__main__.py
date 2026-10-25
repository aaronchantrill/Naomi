# voice_assistant/__main__.py
import asyncio
import functools
import logging
import signal
from pathlib import Path

from .assistant import VoiceAssistant

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config.json"


def custom_signal_handler(assistant: VoiceAssistant) -> None:
    log.info("Cleaning up and shutting down")
    # Add your teardown logic here
    assistant.shutdown()

async def start(config_path: Path = DEFAULT_CONFIG) -> None:
    """Async entry point — creates the VoiceAssistant and starts uvicorn.

    ``VoiceAssistant.start()`` is called automatically via the FastAPI
    lifespan handler, so background tasks are started before the first
    request is served. This coroutine blocks until the server shuts down.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    assistant = VoiceAssistant(config_path=config_path)

    # Run uvicorn in a thread so we stay in async context and can await
    # shutdown cleanly if needed in the future.
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, functools.partial(custom_signal_handler, assistant))
    await loop.run_in_executor(None, assistant.run)


if __name__ == "__main__":
    asyncio.run(start())
