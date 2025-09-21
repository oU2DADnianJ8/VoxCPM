"""Pydantic request/response models for the API server."""

from __future__ import annotations

from typing import Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - compatibility fallback
    from pydantic import BaseModel, Field  # type: ignore
    from pydantic import validator as field_validator  # type: ignore

    ConfigDict = None  # type: ignore
    _PYDANTIC_V2 = False


class PromptOverride(BaseModel):
    """Optional inline prompt definition for zero-shot voice cloning."""

    audio_base64: Optional[str] = Field(None, alias="audio")
    audio_format: Optional[str] = Field(None, alias="format")
    text: Optional[str] = None

    if ConfigDict is not None:  # pragma: no branch - compatibility
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - pydantic v1 compatibility

        class Config:
            allow_population_by_field_name = True


class AudioSpeechRequest(BaseModel):
    """Schema for the OpenAI-compatible ``/v1/audio/speech`` endpoint."""

    model: str
    voice: Optional[str] = Field(None, description="Voice preset identifier")
    input: str = Field(..., description="Text to synthesize")
    response_format: str = Field("mp3", alias="response_format")
    speed: float = Field(1.0, description="Playback speed multiplier")
    stream: bool = Field(False, description="Enable server-sent events streaming")
    cfg_scale: Optional[float] = Field(None, alias="cfg_scale")
    inference_steps: Optional[int] = Field(None, alias="inference_steps")
    normalize: Optional[bool] = None
    denoise: Optional[bool] = None
    max_length: Optional[int] = Field(None, alias="max_length")
    prompt: Optional[PromptOverride] = None

    if ConfigDict is not None:  # pragma: no branch - compatibility
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - pydantic v1 compatibility

        class Config:
            allow_population_by_field_name = True

    if _PYDANTIC_V2:

        @field_validator("input", mode="before")
        def _validate_input(cls, value: str) -> str:
            if not value or not str(value).strip():
                raise ValueError("input text must be a non-empty string")
            return value

        @field_validator("speed", mode="before")
        def _validate_speed(cls, value: float) -> float:
            numeric = float(value)
            if numeric <= 0:
                raise ValueError("speed must be greater than 0")
            if numeric > 4:
                raise ValueError("speed must be less than or equal to 4.0")
            return numeric

    else:  # pragma: no cover - pydantic v1 compatibility

        @field_validator("input", pre=True)
        def _validate_input(cls, value: str) -> str:
            if not value or not str(value).strip():
                raise ValueError("input text must be a non-empty string")
            return value

        @field_validator("speed", pre=True)
        def _validate_speed(cls, value: float) -> float:
            numeric = float(value)
            if numeric <= 0:
                raise ValueError("speed must be greater than 0")
            if numeric > 4:
                raise ValueError("speed must be less than or equal to 4.0")
            return numeric


__all__ = ["AudioSpeechRequest", "PromptOverride"]
