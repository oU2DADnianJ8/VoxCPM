"""Voice library utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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

    def to_response(self) -> Dict[str, Optional[str]]:
        """Serialize the profile for API responses."""

        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "tags": self.tags,
            "has_prompt": self.prompt_audio is not None,
        }


class VoiceLibrary:
    """Loads and manages :class:`VoiceProfile` definitions."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._voices: Dict[str, VoiceProfile] = {}
        self.reload()

    def reload(self) -> None:
        """Reload voice definitions from disk."""

        voices: Dict[str, VoiceProfile] = {}
        if self.directory.exists():
            for json_file in sorted(self.directory.glob("*.json")):
                try:
                    with json_file.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    config = VoiceConfig(**data)
                except (OSError, json.JSONDecodeError, ValidationError) as exc:
                    # Skip malformed definitions while preserving others.
                    print(f"[VoiceLibrary] Failed to load {json_file}: {exc}")
                    continue

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
                )
                voices[config.name.lower()] = profile

        # Always ensure the default voice is available.
        if "default" not in voices:
            voices["default"] = VoiceProfile(name="default", description="Unconditioned VoxCPM voice")

        self._voices = voices

    def get(self, name: Optional[str]) -> Optional[VoiceProfile]:
        """Return the voice profile associated with ``name``."""

        if name is None:
            return None
        return self._voices.get(name.lower())

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name.lower() in self._voices

    def list_profiles(self) -> Iterable[VoiceProfile]:  # pragma: no cover - trivial
        return self._voices.values()

    def as_response(self) -> Dict[str, List[Dict[str, Optional[str]]]]:
        """Return a serializable structure with all voices."""

        return {
            "data": [profile.to_response() for profile in self._voices.values()],
            "object": "list",
        }


__all__ = ["VoiceLibrary", "VoiceProfile"]
