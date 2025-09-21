"""Audio encoding helpers for the API server."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio


@dataclass
class EncodedAudio:
    """Container holding encoded audio bytes and metadata."""

    data: bytes
    format: str
    media_type: str
    extension: str


class AudioEncoder:
    """Utility to convert VoxCPM waveforms into encoded audio payloads."""

    _SUPPORTED_FORMATS: Dict[str, Dict[str, str]] = {
        "mp3": {"media_type": "audio/mpeg", "extension": "mp3", "ffmpeg_format": "mp3", "codec": "libmp3lame"},
        "wav": {"media_type": "audio/wav", "extension": "wav", "subtype": "PCM_16"},
        "flac": {"media_type": "audio/flac", "extension": "flac", "subtype": None},
        "aac": {"media_type": "audio/aac", "extension": "aac", "ffmpeg_format": "adts", "codec": "aac"},
        "opus": {"media_type": "audio/ogg", "extension": "opus", "ffmpeg_format": "ogg", "codec": "libopus"},
        "pcm": {"media_type": "audio/L16", "extension": "pcm"},
    }

    def __init__(self, chunk_size: int = 32768):
        self.chunk_size = chunk_size

    @property
    def supported_formats(self) -> Iterable[str]:  # pragma: no cover - trivial
        return self._SUPPORTED_FORMATS.keys()

    def encode(self, waveform: np.ndarray, sample_rate: int, target_format: str) -> EncodedAudio:
        fmt = target_format.lower()
        if fmt not in self._SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported response_format '{target_format}'. Supported: {', '.join(self._SUPPORTED_FORMATS)}")

        spec = self._SUPPORTED_FORMATS[fmt]
        waveform = np.asarray(waveform, dtype=np.float32)

        if fmt == "pcm":
            clipped = np.clip(waveform, -1.0, 1.0)
            pcm16 = (clipped * 32767.0).astype("<i2")
            return EncodedAudio(data=pcm16.tobytes(), format=fmt, media_type=spec["media_type"], extension=spec["extension"])

        if "ffmpeg_format" in spec:
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError("ffmpeg is required to encode to mp3/aac/opus formats but was not found in PATH")

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
            wav_buffer.seek(0)

            cmd = [
                ffmpeg_path,
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-vn",
            ]
            if spec.get("codec"):
                cmd.extend(["-c:a", spec["codec"]])
            cmd.extend(["-f", spec["ffmpeg_format"], "pipe:1"])

            result = subprocess.run(cmd, input=wav_buffer.read(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed to encode audio ({fmt}): {result.stderr.decode('utf-8', errors='ignore') or 'unknown error'}"
                )

            return EncodedAudio(data=result.stdout, format=fmt, media_type=spec["media_type"], extension=spec["extension"])

        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format=fmt.upper(), subtype=spec.get("subtype"))
        buffer.seek(0)
        return EncodedAudio(data=buffer.read(), format=fmt, media_type=spec["media_type"], extension=spec["extension"])

    def chunk_bytes(self, payload: bytes) -> Iterable[bytes]:
        """Split *payload* into streaming-sized chunks."""

        for index in range(0, len(payload), self.chunk_size):
            yield payload[index : index + self.chunk_size]

    def iter_base64_chunks(self, payload: bytes) -> Iterable[str]:
        """Yield base64 encoded chunks of the payload."""

        for chunk in self.chunk_bytes(payload):
            yield base64.b64encode(chunk).decode("ascii")


def apply_speed(waveform: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """Apply a playback speed multiplier to ``waveform``."""

    if abs(speed - 1.0) < 1e-3:
        return waveform

    target_rate = max(1, int(round(sample_rate / speed)))
    tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, sample_rate, target_rate)
    return resampled.squeeze(0).cpu().numpy()


__all__ = ["AudioEncoder", "EncodedAudio", "apply_speed"]
