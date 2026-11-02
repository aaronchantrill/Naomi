# orion/tts.py
import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)
emoji_filter = re.compile("["
    U"\U0001F600-\U0001F64F"  # emoticons
    U"\U0001F300-\U0001F5FF"  # symbols & pictographs
    U"\U0001F680-\U0001F6FF"  # transport & map symbols
    U"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    U"\U00002702-\U000027B0"
    U"\U000024C2-\U0001F251"
    U"\U0001F900-\U0001F9FF"  # symbols and pictographs, extended
    U"\U0001F000-\U0001F0FF"  # flags
    U"\U0001F180-\U0001F1FF"  # flags
"]+", re.UNICODE)
re_fix_double_asterisks = re.compile(r'\*\*(.+)\*\*')
re_fix_asterisks = re.compile(r'\*([^\*]+)\*')
re_remove_square_brackets = re.compile(r'\[.*?\]')
re_remove_metadata = re.compile(r'^\s*\[USER:.*?\]\[DATETIME:.*?\]')


class TTSEngine:
    """Piper TTS engine.

    Synthesises text to raw 16-bit PCM bytes.
    Piper is CPU-bound and not async-safe, so synthesis runs in a
    dedicated single-thread executor to avoid blocking the event loop.
    """

    def __init__(self, tts_cfg: dict) -> None:
        self.model_file  = os.path.expanduser(tts_cfg.get("model", ""))
        self.speaker_id  = tts_cfg.get("speaker_id", 0)
        self.sample_rate = tts_cfg.get("sample_rate", 22050)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orion-tts")
        self._voice    = None
        self._syn_config = None

        if self.model_file:
            self._load_model()
        else:
            log.warning("TTSEngine: no model file configured — synthesis will be unavailable.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._voice is not None

    async def synthesize(self, text: str) -> bytes:
        """Synthesise *text* and return raw 16-bit PCM bytes.

        Raises ``RuntimeError`` if no model is loaded.
        Runs Piper in a thread pool so the event loop is not blocked.
        """
        if not self.available:
            raise RuntimeError("TTS model is not loaded.")
        text = emoji_filter.sub(r'', text)
        text = re_fix_double_asterisks.sub(r'\1!', text)
        text = re_fix_asterisks.sub(r'\1!', text)
        text = re_remove_square_brackets.sub(r'',text)
        text = re_remove_metadata.sub(f'', text)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._synthesize_sync, text)

    def stop(self) -> None:
        self._executor.shutdown(wait=False)
        log.info("TTS engine stopped.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from piper.voice import PiperVoice  # type: ignore
            log.info(f"Loading Piper TTS model: {self.model_file}")
            self._voice = PiperVoice.load(self.model_file)

            # Build a SynthesisConfig if the installed version supports it
            try:
                from piper.voice import SynthesisConfig  # type: ignore
                self._syn_config = SynthesisConfig(speaker_id=self.speaker_id)
                log.info("Piper: using SynthesisConfig (new API)")
            except ImportError:
                self._syn_config = None
                log.info("Piper: SynthesisConfig not available, using legacy API")

            log.info("Piper TTS model loaded.")
        except Exception as exc:
            log.error(f"Failed to load Piper TTS model: {exc}")
            self._voice = None

    def _synthesize_sync(self, text: str) -> bytes:
        """Blocking synthesis — called from the thread pool executor."""
        voice = self._voice
        # Leading silence padding (matches original implementation)
        audio = b"\x00" * 11026

        if hasattr(voice, "synthesize_stream_raw"):
            # Legacy Piper API
            for block in voice.synthesize_stream_raw(text, speaker_id=self.speaker_id):
                audio += block
        elif hasattr(voice, "synthesize"):
            # New Piper API
            for chunk in voice.synthesize(text, syn_config=self._syn_config):
                audio += chunk.audio_int16_bytes
        else:
            raise RuntimeError(
                "Piper voice object has neither 'synthesize_stream_raw' "
                "nor 'synthesize' — unsupported Piper version."
            )

        return audio
