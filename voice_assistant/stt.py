# voice_assistant/stt.py
import asyncio
import logging
import numpy as np
import os
import sherpa_onnx
from concurrent.futures import ThreadPoolExecutor


log = logging.getLogger(__name__)


class STTEngine:
    """Sherpa-onnx streaming ASR engine.

    Owns the recognizer, the NN thread pool, and the batch-decode queue.
    Call ``await engine.start()`` once the asyncio event loop is running
    to launch the background consumer task(s).
    """

    def __init__(self, stt_cfg: dict, endpoint_cfg: dict, server_cfg: dict) -> None:
        self.stt_cfg    = stt_cfg
        self.endpoint_cfg = endpoint_cfg

        self.nn_pool_size = server_cfg.get("nn_pool_size", 1)
        self.max_batch    = server_cfg.get("max_batch_size", 3)
        self.max_wait_ms  = server_cfg.get("max_wait_ms", 10)

        self._nn_pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.nn_pool_size, thread_name_prefix="assistant-nn"
        )
        self._queue: asyncio.Queue | None = None
        self._consumer_tasks: list[asyncio.Task] = []

        log.info("Loading STT model...")
        self.recognizer = self._create_recognizer()
        self.sample_rate: int = int(self.recognizer.config.feat_config.sampling_rate)
        log.info(f"STT model loaded. Sample rate: {self.sample_rate} Hz")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background batch-decode consumer tasks."""
        self._queue = asyncio.Queue()
        for _ in range(self.nn_pool_size):
            task = asyncio.create_task(self._consumer())
            self._consumer_tasks.append(task)
        log.info(f"STT: {self.nn_pool_size} consumer task(s) started.")

    def stop(self) -> None:
        """Cancel consumer tasks and shut down thread pool."""
        for task in self._consumer_tasks:
            task.cancel()
        self._consumer_tasks.clear()
        self._nn_pool.shutdown(wait=False)
        log.info("STT engine stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_stream(self) -> sherpa_onnx.OnlineStream:
        return self.recognizer.create_stream()

    def is_ready(self, stream: sherpa_onnx.OnlineStream) -> bool:
        return self.recognizer.is_ready(stream)

    def is_endpoint(self, stream: sherpa_onnx.OnlineStream) -> bool:
        return self.recognizer.is_endpoint(stream)

    def get_result(self, stream: sherpa_onnx.OnlineStream) -> str:
        return self.recognizer.get_result(stream)

    def reset(self, stream: sherpa_onnx.OnlineStream) -> None:
        self.recognizer.reset(stream)

    async def decode(self, stream: sherpa_onnx.OnlineStream) -> None:
        """Queue a stream for batch decoding and await the result."""
        if self._queue is None:
            raise RuntimeError("STTEngine.start() has not been called.")
        loop   = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((stream, future))
        await future

    @staticmethod
    def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Linear interpolation resample (pure numpy, no torch dependency)."""
        if orig_sr == target_sr:
            return samples
        new_length = max(1, int(len(samples) * target_sr / orig_sr))
        old_idx    = np.linspace(0, len(samples) - 1, new_length)
        return np.interp(old_idx, np.arange(len(samples)), samples).astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _consumer(self) -> None:
        """Drain the queue, batch streams, run decode in the thread pool."""
        q = self._queue
        while True:
            if q.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue
            batch: list[tuple] = []
            try:
                while len(batch) < self.max_batch:
                    item = q.get_nowait()
                    if self.recognizer.is_ready(item[0]):
                        batch.append(item)
            except asyncio.QueueEmpty:
                pass
            if not batch:
                continue
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._nn_pool,
                self.recognizer.decode_streams,
                [b[0] for b in batch],
            )
            for _, future in batch:
                q.task_done()
                future.set_result(None)

    def _create_recognizer(self) -> sherpa_onnx.OnlineRecognizer:
        cfg = self.stt_cfg
        ep  = self.endpoint_cfg
        model_type = cfg.get("type", "transducer")
        common = dict(
            tokens=os.path.expanduser(cfg["tokens"]),
            num_threads=cfg.get("num_threads", 2),
            sample_rate=cfg.get("sample_rate", 16000),
            feature_dim=cfg.get("feat_dim", 80),
            decoding_method=cfg.get("decoding_method", "greedy_search"),
            enable_endpoint_detection=ep.get("enabled", True),
            rule1_min_trailing_silence=ep.get("rule1_min_trailing_silence", 2.4),
            rule2_min_trailing_silence=ep.get("rule2_min_trailing_silence", 1.2),
            rule3_min_utterance_length=ep.get("rule3_min_utterance_length", 20),
            provider=cfg.get("provider", "cpu")
        )
        if model_type == "transducer":
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=os.path.expanduser(cfg["encoder"]),
                decoder=os.path.expanduser(cfg["decoder"]),
                joiner=os.path.expanduser(cfg["joiner"]),
                max_active_paths=cfg.get("num_active_paths", 4),
                hotwords_file=os.path.expanduser(cfg.get("hotwords_file", "")),
                hotwords_score=cfg.get("hotwords_score", 1.5),
                blank_penalty=cfg.get("blank_penalty", 0.0),
                modeling_unit=cfg.get("modeling_unit", "bpe"),
                bpe_vocab=os.path.expanduser(cfg.get("bpe_vocab", "")),
                **common
            )
        elif model_type == "paraformer":
            return sherpa_onnx.OnlineRecognizer.from_paraformer(
                encoder=os.path.expanduser(cfg["paraformer_encoder"]),
                decoder=os.path.expanduser(cfg["paraformer_decoder"]),
                **common
            )
        elif model_type == "zipformer2_ctc":
            return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=os.path.expanduser(cfg["zipformer2_ctc"]),
                **common
            )
        elif model_type == "wenet_ctc":
            return sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
                model=os.path.expanduser(cfg["wenet_ctc"]),
                **common
            )
        else:
            raise ValueError(f"Unknown STT model type: {model_type}")
