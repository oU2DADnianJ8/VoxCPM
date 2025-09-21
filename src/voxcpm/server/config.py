"""Configuration helpers for the VoxCPM OpenAI-compatible server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


def _default_voices_dir() -> Path:
    """Return the directory that should be used for bundled/custom voices."""

    package_root = Path(__file__).resolve().parents[3]
    candidates = [
        package_root / "voices",
        package_root / "assets" / "voices",
        Path.cwd() / "voices",
        Path.cwd() / "assets" / "voices",
        Path(__file__).resolve().parent / "voices",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _coerce_bool(value: Optional[object], default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Optional[object], default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _coerce_float(value: Optional[object], default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _coerce_str_list(value: Optional[object], default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return str(Path(value).expanduser().resolve())


@dataclass
class ServerSettings:
    """Environment-driven configuration for the API server."""

    model_id: str
    model_path: Optional[str]
    cache_dir: Optional[str]
    local_files_only: bool
    load_denoiser: bool
    zipenhancer_model_id: str

    openai_model_name: str

    voices_dir: Path
    default_voice: str

    stream_chunk_size: int
    allow_streaming: bool

    cors_allow_origins: List[str]
    cors_allow_methods: List[str]
    cors_allow_headers: List[str]
    cors_allow_credentials: bool

    inference_cfg_value: float
    inference_timesteps: int
    max_length: int
    normalize_text: bool
    denoise_prompts: bool
    retry_badcase: bool
    retry_badcase_max_times: int
    retry_badcase_ratio_threshold: float

    log_level: str

    def __init__(self, **overrides: object) -> None:
        env = os.environ

        self.model_id = str(overrides.get("model_id") or env.get("VOXCPM_MODEL_ID") or "openbmb/VoxCPM-0.5B")
        self.model_path = _coerce_path(overrides.get("model_path") or env.get("VOXCPM_MODEL_PATH"))
        self.cache_dir = _coerce_path(overrides.get("cache_dir") or env.get("VOXCPM_CACHE_DIR"))
        self.local_files_only = _coerce_bool(overrides.get("local_files_only") or env.get("VOXCPM_LOCAL_FILES_ONLY"), False)
        self.load_denoiser = _coerce_bool(overrides.get("load_denoiser") or env.get("VOXCPM_ENABLE_DENOISER"), True)
        self.zipenhancer_model_id = str(
            overrides.get("zipenhancer_model_id")
            or env.get("VOXCPM_ZIPENHANCER_ID")
            or "iic/speech_zipenhancer_ans_multiloss_16k_base"
        )

        self.openai_model_name = str(
            overrides.get("openai_model_name") or env.get("VOXCPM_OPENAI_MODEL_NAME") or "voxcpm-0.5b"
        )

        voice_dir_value = overrides.get("voices_dir") or env.get("VOXCPM_VOICES_DIR")
        if voice_dir_value:
            self.voices_dir = Path(str(voice_dir_value)).expanduser().resolve()
        else:
            self.voices_dir = _default_voices_dir()
        self.default_voice = str(overrides.get("default_voice") or env.get("VOXCPM_DEFAULT_VOICE") or "default")

        self.stream_chunk_size = _coerce_int(
            overrides.get("stream_chunk_size") or env.get("VOXCPM_STREAM_CHUNK_SIZE"), 32768
        )
        self.allow_streaming = _coerce_bool(overrides.get("allow_streaming") or env.get("VOXCPM_ALLOW_STREAMING"), True)

        self.cors_allow_origins = _coerce_str_list(
            overrides.get("cors_allow_origins") or env.get("VOXCPM_CORS_ALLOW_ORIGINS"), ["*"]
        )
        self.cors_allow_methods = _coerce_str_list(
            overrides.get("cors_allow_methods") or env.get("VOXCPM_CORS_ALLOW_METHODS"), ["GET", "POST", "OPTIONS"]
        )
        self.cors_allow_headers = _coerce_str_list(
            overrides.get("cors_allow_headers") or env.get("VOXCPM_CORS_ALLOW_HEADERS"), ["*"]
        )
        self.cors_allow_credentials = _coerce_bool(
            overrides.get("cors_allow_credentials") or env.get("VOXCPM_CORS_ALLOW_CREDENTIALS"), False
        )

        self.inference_cfg_value = _coerce_float(overrides.get("inference_cfg_value") or env.get("VOXCPM_CFG_VALUE"), 2.0)
        self.inference_timesteps = _coerce_int(
            overrides.get("inference_timesteps") or env.get("VOXCPM_INFERENCE_TIMESTEPS"), 10
        )
        self.max_length = _coerce_int(overrides.get("max_length") or env.get("VOXCPM_MAX_LENGTH"), 4096)
        self.normalize_text = _coerce_bool(overrides.get("normalize_text") or env.get("VOXCPM_NORMALIZE_TEXT"), True)
        self.denoise_prompts = _coerce_bool(overrides.get("denoise_prompts") or env.get("VOXCPM_DENOISE_PROMPTS"), True)
        self.retry_badcase = _coerce_bool(overrides.get("retry_badcase") or env.get("VOXCPM_RETRY_BADCASE"), True)
        self.retry_badcase_max_times = _coerce_int(
            overrides.get("retry_badcase_max_times") or env.get("VOXCPM_RETRY_BADCASE_MAX_TIMES"), 3
        )
        self.retry_badcase_ratio_threshold = _coerce_float(
            overrides.get("retry_badcase_ratio_threshold") or env.get("VOXCPM_RETRY_BADCASE_RATIO"), 6.0
        )

        self.log_level = str(overrides.get("log_level") or env.get("VOXCPM_LOG_LEVEL") or "info")


__all__ = ["ServerSettings"]
