"""FastAPI application exposing an OpenAI-compatible interface."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

if __package__ in {None, ""}:  # pragma: no cover - runtime convenience
    import pathlib
    import sys

    package_root = pathlib.Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from voxcpm.server.audio import AudioEncoder, apply_speed  # type: ignore[import-not-found]
    from voxcpm.server.config import ServerSettings  # type: ignore[import-not-found]
    from voxcpm.server.schemas import AudioSpeechRequest  # type: ignore[import-not-found]
    from voxcpm.server.tts import VoxCPMTTSManager  # type: ignore[import-not-found]
    from voxcpm.server.voices import VoiceLibrary  # type: ignore[import-not-found]
else:  # pragma: no cover - exercised via package imports
    from .audio import AudioEncoder, apply_speed
    from .config import ServerSettings
    from .schemas import AudioSpeechRequest
    from .tts import VoxCPMTTSManager
    from .voices import VoiceLibrary

LOGGER = logging.getLogger("voxcpm.server.app")


def _error_payload(message: str, error_type: str = "invalid_request_error", code: Optional[str] = None) -> Dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    settings = settings or ServerSettings()

    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(
        title="VoxCPM OpenAI-Compatible API",
        version="1.0.0",
        summary="Serve VoxCPM text-to-speech through OpenAI-compatible endpoints",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        allow_credentials=settings.cors_allow_credentials,
    )

    voice_library = VoiceLibrary(settings.voices_dir)
    tts_manager = VoxCPMTTSManager(settings)
    audio_encoder = AudioEncoder(chunk_size=settings.stream_chunk_size)

    app.state.settings = settings
    app.state.voice_library = voice_library
    app.state.tts_manager = tts_manager
    app.state.audio_encoder = audio_encoder

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:  # noqa: D401 - FastAPI signature
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        return JSONResponse(status_code=exc.status_code, content=_error_payload(detail, code=str(exc.status_code)))

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:  # noqa: D401 - FastAPI signature
        errors = exc.errors()
        if errors:
            first = errors[0]
            if first.get("type") == "json_invalid":
                reason = first.get("ctx", {}).get("error", "unknown error")
                message = f"Request body was not valid JSON: {reason}"
                return JSONResponse(status_code=400, content=_error_payload(message, code="400"))

            location = first.get("loc", [])
            field = next((str(item) for item in location if item not in {"body"}), "request")
            message = first.get("msg", "Invalid request parameter")
            detail = f"{field}: {message}"
            return JSONResponse(status_code=422, content=_error_payload(detail, code="422"))

        return JSONResponse(status_code=422, content=_error_payload("Invalid request payload", code="422"))

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - requires runtime environment
        try:
            LOGGER.info("Preparing VoxCPM model")
            await tts_manager.ensure_ready()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to prepare VoxCPM: %s", exc)
            raise

    @app.get("/health", tags=["system"])
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "model_loaded": tts_manager.is_ready}

    @app.get("/v1/models", tags=["openai"])
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.openai_model_name,
                    "object": "model",
                    "owned_by": "VoxCPM",
                    "root": settings.model_id,
                }
            ],
        }

    @app.get("/v1/models/{model_id}", tags=["openai"])
    async def get_model(model_id: str) -> Dict[str, Any]:
        if model_id not in {settings.openai_model_name, settings.model_id}:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return {
            "id": settings.openai_model_name,
            "object": "model",
            "owned_by": "VoxCPM",
            "root": settings.model_id,
        }

    @app.get("/v1/voices", tags=["openai"])
    async def list_voices() -> Dict[str, Any]:
        return voice_library.as_response()

    @app.options("/v1/audio/speech", include_in_schema=False)
    async def audio_speech_options() -> Response:
        return Response(status_code=204, headers={"Allow": "OPTIONS, POST"})

    @app.post("/v1/audio/speech", tags=["openai"])
    async def audio_speech(payload: AudioSpeechRequest) -> Response:
        model_name = payload.model or settings.openai_model_name
        if model_name not in {settings.openai_model_name, settings.model_id}:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not available")

        voice_name = payload.voice or settings.default_voice
        voice = voice_library.get(voice_name)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

        try:
            generation = await tts_manager.generate(
                payload.input,
                voice,
                cfg_scale=payload.cfg_scale,
                inference_steps=payload.inference_steps,
                normalize=payload.normalize,
                denoise=payload.denoise,
                max_length=payload.max_length,
                prompt_override=payload.prompt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("Synthesis failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to synthesize speech") from exc

        waveform = generation.audio
        if payload.speed and abs(payload.speed - 1.0) >= 1e-3:
            try:
                waveform = apply_speed(waveform, payload.speed, generation.sample_rate)
            except Exception as exc:  # pragma: no cover - runtime safety
                LOGGER.exception("Failed to apply speed %.2f: %s", payload.speed, exc)
                raise HTTPException(status_code=500, detail="Unable to apply speed adjustment") from exc

        try:
            encoded = audio_encoder.encode(waveform, generation.sample_rate, payload.response_format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        request_id = f"resp_{uuid.uuid4().hex}"
        headers = {
            "X-Request-ID": request_id,
            "X-Model": settings.openai_model_name,
            "X-Voice-Name": voice.name,
        }

        if payload.stream:
            if not settings.allow_streaming:
                raise HTTPException(status_code=400, detail="Streaming responses are disabled on this server")

            async def event_stream() -> AsyncGenerator[bytes, None]:
                created = int(time.time())
                metadata = {
                    "id": request_id,
                    "object": "audio.speech",
                    "created": created,
                    "model": settings.openai_model_name,
                    "voice": voice.name,
                    "format": encoded.format,
                }
                yield _encode_sse("response", metadata)

                index = 0
                for chunk in audio_encoder.iter_base64_chunks(encoded.data):
                    payload_chunk = {
                        "id": request_id,
                        "index": index,
                        "model": settings.openai_model_name,
                        "voice": voice.name,
                        "data": chunk,
                    }
                    index += 1
                    yield _encode_sse("audio", payload_chunk)
                    await asyncio.sleep(0)

                yield _encode_sse("done", {"id": request_id})

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={**headers, "Cache-Control": "no-cache"},
            )

        disposition = f'inline; filename="speech.{encoded.extension}"'
        return Response(
            content=encoded.data,
            media_type=encoded.media_type,
            headers={**headers, "Content-Disposition": disposition},
        )

    return app


def _encode_sse(event: str, data: Dict[str, Any]) -> bytes:
    message = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {message}\n\n".encode("utf-8")


def main() -> None:  # pragma: no cover - CLI shim
    """Entry-point for running the server directly via ``python app.py``."""

    if __package__ in {None, ""}:  # pragma: no cover - runtime convenience
        # ``voxcpm.server.main`` relies on the same path bootstrap performed above.
        from voxcpm.server.main import run as _run  # type: ignore[import-not-found]
    else:  # pragma: no cover - exercised via package imports
        from .main import run as _run

    _run()


if __name__ == "__main__":  # pragma: no cover - runtime convenience
    main()


__all__ = ["create_app", "main"]
