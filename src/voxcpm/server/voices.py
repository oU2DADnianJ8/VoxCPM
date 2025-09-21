"""Voice library utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ValidationError


class VoiceConfig(BaseModel):
    """Schema describing a voice definition file."""

    name: str
    description: Optional[str] = None
    prompt_audio: Optional[str] = None
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass(frozen=True)
class VoiceProfile:
    """Runtime representation of a voice option."""

    name: str
    description: Optional[str] = None
    prompt_audio: Optional[Path] = None
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: str = "manifest"

    def to_response(self) -> Dict[str, object]:
        """Serialize the profile for API responses."""

        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "tags": self.tags,
            "has_prompt": self.prompt_audio is not None,
            "requires_prompt_text": self.requires_prompt_text,
            "source": self.source,
        }

    @property
    def requires_prompt_text(self) -> bool:
        return self.prompt_audio is not None and self.prompt_text is None


LOGGER = logging.getLogger("voxcpm.server.voices")


class VoiceLibrary:
    """Loads and manages :class:`VoiceProfile` definitions."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._voices: Dict[str, VoiceProfile] = {}
        self.reload()

    def reload(self) -> None:
        """Reload voice definitions from disk."""

        voices: Dict[str, VoiceProfile] = {}
        metadata_by_stem: Dict[str, Dict[str, Any]] = {}
        manifests: List[Tuple[Path, VoiceConfig]] = []

        if not self.directory.exists():
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - filesystem safety
                LOGGER.warning("Failed to create voices directory %s: %s", self.directory, exc)
                self._voices = {
                    "default": VoiceProfile(name="default", description="Unconditioned VoxCPM voice", source="builtin")
                }
                return

        for json_file in sorted(self.directory.glob("*.json")):
            try:
                with json_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Failed to parse %s: %s", json_file, exc)
                continue

            if "name" in data:
                try:
                    config = VoiceConfig(**data)
                except ValidationError as exc:
                    LOGGER.warning("Invalid voice manifest %s: %s", json_file, exc)
                    continue
                manifests.append((json_file, config))
            else:
                metadata_by_stem[json_file.stem.lower()] = data

        for json_file, config in manifests:
            prompt_audio_path: Optional[Path] = None
            if config.prompt_audio:
                prompt_audio_path = (json_file.parent / config.prompt_audio).resolve()

            profile = VoiceProfile(
                name=config.name,
                description=config.description,
                prompt_audio=prompt_audio_path,
                prompt_text=config.prompt_text,
                language=config.language,
                tags=config.tags or [],
                source="manifest",
            )
            voices[config.name.lower()] = profile

        for wav_file in sorted(self.directory.glob("*.wav")):
            voice_key = wav_file.stem.lower()
            if voice_key in voices:
                # Manifests take precedence when both exist.
                continue

            metadata = metadata_by_stem.get(voice_key, {})
            description = metadata.get("description") if isinstance(metadata, dict) else None
            language = metadata.get("language") if isinstance(metadata, dict) else None
            tags = metadata.get("tags") if isinstance(metadata, dict) else None
            prompt_text = None

            transcript_path = wav_file.with_suffix(".txt")
            if transcript_path.exists():
                try:
                    prompt_text_candidate = transcript_path.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    LOGGER.warning("Failed to read transcript %s: %s", transcript_path, exc)
                else:
                    if prompt_text_candidate:
                        prompt_text = prompt_text_candidate

            if prompt_text is None and isinstance(metadata, dict):
                metadata_prompt_text = metadata.get("prompt_text")
                if metadata_prompt_text:
                    prompt_text = str(metadata_prompt_text)

            profile = VoiceProfile(
                name=wav_file.stem,
                description=(description or f"Voice preset derived from {wav_file.name}"),
                prompt_audio=wav_file.resolve(),
                prompt_text=prompt_text,
                language=str(language) if language else None,
                tags=list(tags) if isinstance(tags, list) else [],
                source="filesystem",
            )
            voices[voice_key] = profile

        # Always ensure the default voice is available.
        if "default" not in voices:
            voices["default"] = VoiceProfile(
                name="default",
                description="Unconditioned VoxCPM voice",
                source="builtin",
            )

        self._voices = voices

    def get(self, name: Optional[str]) -> Optional[VoiceProfile]:
        """Return the voice profile associated with ``name``."""

        if name is None:
            return None
        return self._voices.get(name.lower())

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name.lower() in self._voices

    def list_profiles(self) -> Iterable[VoiceProfile]:  # pragma: no cover - trivial
        return sorted(self._voices.values(), key=lambda profile: profile.name.lower())

    def as_response(self) -> Dict[str, List[Dict[str, object]]]:
        """Return a serializable structure with all voices."""

        return {
            "data": [profile.to_response() for profile in self.list_profiles()],
            "object": "list",
        }


__all__ = ["VoiceLibrary", "VoiceProfile"]
