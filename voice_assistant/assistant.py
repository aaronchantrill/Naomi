# orion/assistant.py
import asyncio
import json
import logging
import os
import signal
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from .stt import STTEngine
from .tts import TTSEngine
from .punctuation import PunctuationEngine
from .llm import LLMClient

log = logging.getLogger(__name__)
SHUTDOWN_PHRASES = {"shut down", "shutdown", "goodbye", "turn yourself off", "shut yourself down"}


class VoiceAssistant:
    """Top-level orchestrator.

    Owns the FastAPI application, SSL certificate management, and
    WebSocket handlers. Delegates ASR to ``STTEngine`` and TTS to
    ``TTSEngine``.
    """

    def __init__(self, config_path: Path = Path("~/.config/Naomi/config.json")) -> None:
        self.config_path = config_path
        self.cfg         = self._load_config()

        self.srv_cfg = self.cfg["server"]
        self.host    = self.srv_cfg.get("host", "0.0.0.0")
        self.port    = self.srv_cfg.get("port", 8443)
        self.cert    = Path(self.srv_cfg.get("cert_file", "cert.pem")).expanduser().resolve()
        self.key     = Path(self.srv_cfg.get("key_file",  "key.pem")).expanduser().resolve()

        self._ensure_cert()

        self.stt = STTEngine(
            stt_cfg=self.cfg["stt"],
            endpoint_cfg=self.cfg["endpoint"],
            server_cfg=self.srv_cfg,
        )
        self.punctuation = PunctuationEngine(punctuation_cfg=self.cfg.get("punctuation", {}))
        self.llm = LLMClient(llm_cfg=self.cfg.get("llm", {}), tts_cfg=self.cfg.get("tts", {}))

        self.app = self._build_app()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background tasks (call once the event loop is running)."""
        await self.stt.start()
        log.info(f"ASR ready")
        log.info(f"Voice assistant ready — https://{self.host}:{self.port}")

    def stop(self) -> None:
        self.llm.save_conversation()
        self.stt.stop()
        self.llm.tts.stop()

    def run(self) -> None:
        """Blocking entry point — starts uvicorn."""
        try:
          uvicorn.run(
              self.app,
              host=self.host,
              port=self.port,
              ssl_certfile=self.cert,
              ssl_keyfile=self.key,
              log_level="info",
          )
        except KeyboardInterrupt:
          print("Exiting")

    # ------------------------------------------------------------------
    # FastAPI app factory
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        assistant = self  # closure reference

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await assistant.start()
            yield
            await assistant.stop()

        app = FastAPI(lifespan=lifespan)

        # Read the index.html file each time it is served, so changes don't
        # require a restart
        with open(Path(__file__).parent.parent / "index.html", "r") as f:
            index_html = f.read().replace(
                "__TTS_SAMPLE_RATE__",
                str(self.llm.tts.sample_rate)
            )

        @app.get("/", response_class=HTMLResponse)
        async def index():
            # Read the index.html file each time it is served, so changes don't
            # require a restart
            with open(Path(__file__).parent.parent / "index.html", "r") as f:
                index_html = f.read().replace(
                    "__TTS_SAMPLE_RATE__",
                    str(self.llm.tts.sample_rate)
                )
            return HTMLResponse(index_html)

        @app.websocket("/ws")
        async def websocket_asr(websocket: WebSocket):
            await assistant.handle_asr(websocket)

        @app.websocket("/tts")
        async def websocket_tts(websocket: WebSocket):
            await assistant.handle_tts(websocket)

        return app

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    async def handle_asr(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client     = websocket.client
        log.info(f"ASR client connected: {client}")

        asr_stream = self.stt.create_stream()
        browser_sr = self.stt.sample_rate
        segment    = 0

        async def send(obj: dict) -> None:
            try:
                await websocket.send_text(json.dumps(obj))
            except Exception:
                pass

        try:
            while True:
                msg = await websocket.receive()

                if "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "init":
                            browser_sr = int(data.get("sampleRate", self.stt.sample_rate))
                            log.info(f"ASR client sample rate: {browser_sr} Hz")
                        elif data.get("type") == "done":
                            break
                        elif data.get("type") == "text":
                            text = data.get('text')
                            await self._respond(text, websocket)
                        elif data.get("type") == "PING":
                            # if it's just a ping, don't do anything
                            pass
                        else:
                            log.info(f"Type: {data.get('type')}")
                    except Exception as e:
                        log.error(e)

                elif "bytes" in msg:
                    samples = np.frombuffer(msg["bytes"], dtype=np.float32).copy()
                    if browser_sr != self.stt.sample_rate:
                        samples = self.stt.resample(samples, browser_sr, self.stt.sample_rate)

                    asr_stream.accept_waveform(
                        sample_rate=self.stt.sample_rate,
                        waveform=samples,
                    )

                    while self.stt.is_ready(asr_stream):
                        await self.stt.decode(asr_stream)
                        result = self.stt.get_result(asr_stream)
                        if result:
                            if self.stt.is_endpoint(asr_stream):
                                if result:
                                    log.info("Running through punctuation")
                                    result = self.punctuation.add_punctuation(result)
                                    log.info(f"\x1b[36m{result}\x1b[0m")
                                    await send({"type": "final", "text": result, "segment": segment})
                                    self.stt.reset(asr_stream)
                                    # asr_stream = self.stt.create_stream()
                                    segment += 1
                                    if result.strip():
                                        asyncio.create_task(self._respond(result.strip(), websocket))
                            else:
                                await send({"type": "partial", "text": result, "segment": segment})

        except Exception as e:
            log.info(f"ASR connection closed: {e}")
        finally:
            # Flush tail padding
            try:
                tail = np.zeros(int(self.stt.sample_rate * 0.3), dtype=np.float32)
                asr_stream.accept_waveform(sample_rate=self.stt.sample_rate, waveform=tail)
                asr_stream.input_finished()
                while self.stt.is_ready(asr_stream):
                    await self.stt.decode(asr_stream)
                result = self.stt.get_result(asr_stream)
                # Run it through the punctuation model
                log.info("Flushing tail padding")
                result = self.punctuation.add_punctuation(result)
                if result:
                    await send({"type": "final", "text": result, "segment": segment})
            except Exception:
                pass
            log.info(f"ASR client disconnected: {client}")

    async def _respond(self, user_text: str, websocket: WebSocket) -> None:
        """Get LLM response, convert text to audio, and send it to the frontend."""
        try:
            log.info("Starting _respond()")
            # Send text to front end
            await self.llm.handle("Aaron", user_text, websocket)
            if any(phrase in user_text.lower() for phrase in SHUTDOWN_PHRASES):
                asyncio.create_task(self._delayed_shutdown())
        except Exception as e:
            log.error(f"LLM/TTS error: {e}")

    async def handle_tts(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client = websocket.client
        log.info(f"TTS client connected: {client}")

        if not self.tts.available:
            await websocket.send_text(json.dumps({
                "type": "tts_error", "detail": "TTS model not loaded"
            }))
            log.warn("TTS Error - closing websocket")
            await websocket.close()
            return

        try:
            msg = await websocket.receive()
            print(f"msg: {msg}")
            if "text" not in msg:
                return
            data = json.loads(msg["text"])
            text = data.get("text", "").strip()
            if not text:
                await websocket.send_text(json.dumps({
                    "type": "tts_error", "detail": "empty text"
                }))
                return

            log.info(f"TTS synthesising: {text!r}")
            audio_bytes = await self.tts.synthesize(text)
            log.info("TTS synthesis complete.")
            await websocket.send_text(json.dumps({"type": "tts_text", "detail": text}))
            log.info("Sent text")
            await websocket.send_bytes(audio_bytes)
            log.info("Sent audio")
            await websocket.send_text(json.dumps({"type": "tts_done"}))
            log.info("TTS completed")
        except Exception as e:
            log.error(f"TTS error: {e}")
            try:
                await websocket.send_text(json.dumps({
                    "type": "tts_error", "detail": str(e)
                }))
            except Exception:
                pass
        finally:
            log.info(f"TTS client disconnected: {client}")

    def shutdown(self) -> None:
        """Gracefully shut down the server."""
        self.stop()
        log.info("Shutting down — goodbye!")
        os.kill(os.getpid(), signal.SIGTERM)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    async def _delayed_shutdown(self) -> None:
        await asyncio.sleep(3)
        self.shutdown()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path) as f:
            return json.load(f)

    def _ensure_cert(self) -> None:
        if os.path.exists(self.cert) and os.path.exists(self.key):
            return
        log.info("Generating self-signed SSL certificate...")
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(self.key, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"TestState"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"TestCity"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Naomi"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(socket.getfqdn())]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        with open(self.cert, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        log.info("SSL certificate generated.")
