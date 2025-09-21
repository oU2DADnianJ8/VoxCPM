"""Utilities for transcribing prompt audio into text."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, Optional

import torch

LOGGER = logging.getLogger("voxcpm.server.transcription")


class PromptTranscriber:
    """Lazy-loading ASR helper used to obtain prompt transcripts."""

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._model_lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        self._cache_lock = asyncio.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            LOGGER.info("Loading prompt ASR model '%s' on %s", self.model_id, self.device)
            from funasr import AutoModel  # Imported lazily to avoid startup cost

            self._model = AutoModel(
                model=self.model_id,
                disable_update=True,
                log_level="WARNING",
                device=self.device,
            )

    def _transcribe_sync(self, audio_path: str) -> str:
        self._ensure_model()
        if self._model is None:  # pragma: no cover - defensive safety
            raise RuntimeError("Prompt ASR model failed to initialise")

        result = self._model.generate(input=audio_path, language="auto", use_itn=True)
        text = ""
        if result and isinstance(result, list):
            candidate = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            text = str(candidate).split("|>")[-1].strip()
        return text

    async def transcribe(self, audio_path: str) -> str:
        """Return the transcript for ``audio_path`` (cached per path)."""

        async with self._cache_lock:
            cached = self._cache.get(audio_path)
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._transcribe_sync, audio_path)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            raise RuntimeError(f"Failed to transcribe prompt audio '{audio_path}': {exc}") from exc

        cleaned = text.strip()
        if not cleaned:
            raise RuntimeError(f"Transcription for '{audio_path}' returned empty text")

        async with self._cache_lock:
            self._cache[audio_path] = cleaned
        return cleaned

    def clear_cache(self) -> None:  # pragma: no cover - maintenance helper
        """Clear cached transcripts (mainly for testing)."""

        self._cache.clear()


__all__ = ["PromptTranscriber"]
