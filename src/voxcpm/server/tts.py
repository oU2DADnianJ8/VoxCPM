"""Runtime management of the VoxCPM text-to-speech model."""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import VoxCPM
from .config import ServerSettings
from .schemas import PromptOverride
from .voices import VoiceProfile


LOGGER = logging.getLogger("voxcpm.server.tts")


@dataclass
class TTSResult:
    """Represents a synthesized waveform."""

    audio: np.ndarray
    sample_rate: int
    voice: str


class VoxCPMTTSManager:
    """High level orchestrator for model loading and inference."""

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._model: Optional[VoxCPM] = None
        self._sample_rate: int = 16000
        self._init_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        """Ensure the underlying model is loaded."""

        if self._model is not None:
            return

        async with self._init_lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        LOGGER.info("Loading VoxCPM model (id=%s, path=%s)", self.settings.model_id, self.settings.model_path)
        if self.settings.model_path:
            model = VoxCPM(
                voxcpm_model_path=self.settings.model_path,
                zipenhancer_model_path=self.settings.zipenhancer_model_id if self.settings.load_denoiser else None,
                enable_denoiser=self.settings.load_denoiser,
            )
        else:
            model = VoxCPM.from_pretrained(
                hf_model_id=self.settings.model_id,
                load_denoiser=self.settings.load_denoiser,
                zipenhancer_model_id=self.settings.zipenhancer_model_id,
                cache_dir=self.settings.cache_dir,
                local_files_only=self.settings.local_files_only,
            )
        self._model = model
        self._sample_rate = getattr(model.tts_model, "sample_rate", 16000)
        LOGGER.info("VoxCPM model ready (sample_rate=%s)", self._sample_rate)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    async def generate(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        cfg_scale: Optional[float] = None,
        inference_steps: Optional[int] = None,
        normalize: Optional[bool] = None,
        denoise: Optional[bool] = None,
        max_length: Optional[int] = None,
        prompt_override: Optional[PromptOverride] = None,
    ) -> TTSResult:
        """Generate speech for ``text`` using the specified ``voice``."""

        await self.ensure_ready()
        if self._model is None:  # pragma: no cover - defensive
            raise RuntimeError("VoxCPM model failed to initialize")

        prompt_path: Optional[str] = voice.prompt_audio.as_posix() if voice.prompt_audio else None
        prompt_text: Optional[str] = voice.prompt_text
        temp_prompt: Optional[tempfile.NamedTemporaryFile] = None

        if prompt_override:
            if prompt_override.audio_base64:
                temp_prompt = self._decode_prompt(prompt_override.audio_base64, prompt_override.audio_format)
                prompt_path = temp_prompt.name
            if prompt_override.text:
                prompt_text = prompt_override.text

        if prompt_path and not prompt_text:
            raise ValueError("A prompt_text must be provided when prompt audio is supplied")

        cfg_value = cfg_scale if cfg_scale is not None else self.settings.inference_cfg_value
        inference_timesteps = inference_steps if inference_steps is not None else self.settings.inference_timesteps
        normalize_text = normalize if normalize is not None else self.settings.normalize_text
        denoise_prompts = denoise if denoise is not None else self.settings.denoise_prompts
        max_len = max_length if max_length is not None else self.settings.max_length

        loop = asyncio.get_running_loop()

        async with self._inference_lock:
            audio = await loop.run_in_executor(
                None,
                lambda: self._model.generate(
                    text=text,
                    prompt_wav_path=prompt_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_length=max_len,
                    normalize=normalize_text,
                    denoise=denoise_prompts,
                    retry_badcase=self.settings.retry_badcase,
                    retry_badcase_max_times=self.settings.retry_badcase_max_times,
                    retry_badcase_ratio_threshold=self.settings.retry_badcase_ratio_threshold,
                ),
            )

        if temp_prompt is not None:
            prompt_file = Path(temp_prompt.name)
            try:
                temp_prompt.close()
            except OSError:  # pragma: no cover - defensive
                pass
            try:
                prompt_file.unlink()
            except OSError:  # pragma: no cover - defensive
                pass

        return TTSResult(audio=audio, sample_rate=self.sample_rate, voice=voice.name)

    def _decode_prompt(self, audio_base64: str, audio_format: Optional[str]) -> tempfile.NamedTemporaryFile:
        """Decode inline prompt audio into a temporary file."""

        try:
            payload = base64.b64decode(audio_base64)
        except (ValueError, TypeError) as exc:  # pragma: no cover - fast fail
            raise ValueError("Failed to decode prompt audio; base64 data is invalid") from exc

        suffix = f".{audio_format.lower()}" if audio_format else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(payload)
        temp_file.flush()
        return temp_file


__all__ = ["VoxCPMTTSManager", "TTSResult"]
