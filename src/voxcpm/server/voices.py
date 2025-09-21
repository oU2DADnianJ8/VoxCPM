"""Voice preset discovery utilities for the VoxCPM API server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

LOGGER = logging.getLogger("voxcpm.server.voices")


class VoiceConfig(BaseModel):
    """Schema describing a voice definition file."""

    name: str
    description: Optional[str] = None
    prompt_audio: Optional[str] = None
    prompt_text: Optional[str] = None
    transcript_file: Optional[str] = None
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

    def to_response(self) -> Dict[str, object]:
        """Serialize the profile for API responses."""

        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "tags": list(self.tags),
            "has_prompt_audio": self.prompt_audio is not None,
            "has_prompt_text": bool(self.prompt_text),
        }


def _resolve_path(base_dir: Path, value: Optional[str], *, label: str) -> Optional[Path]:
    """Resolve ``value`` relative to ``base_dir`` and ensure it exists."""

    if not value:
        return None

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        LOGGER.warning("%s '%s' defined under %s does not exist", label, candidate, base_dir)
        return None
    return candidate


def _read_transcript_file(path: Path, *, warn_if_missing: bool) -> Optional[str]:
    """Return cleaned transcript text from ``path`` if available."""

    if not path.exists():
        if warn_if_missing:
            LOGGER.warning("Transcript file '%s' could not be found", path)
        return None
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        LOGGER.warning("Failed to read transcript file '%s': %s", path, exc)
        return None
    return contents or None


class VoiceLibrary:
    """Loads and manages :class:`VoiceProfile` definitions."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._voices: Dict[str, VoiceProfile] = {}
        self.reload()

    def reload(self) -> None:
        """Reload voice definitions from disk."""

        voices: Dict[str, VoiceProfile] = {}
        metadata: Dict[str, Tuple[VoiceConfig, Path]] = {}

        if self.directory.exists():
            for json_file in sorted(self.directory.glob("*.json")):
                try:
                    with json_file.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    config = VoiceConfig(**data)
                except (OSError, json.JSONDecodeError, ValidationError) as exc:
                    LOGGER.warning("Failed to load voice definition '%s': %s", json_file, exc)
                    continue

                metadata[config.name.lower()] = (config, json_file.parent)

            wav_files = sorted({path for pattern in ("*.wav", "*.WAV") for path in self.directory.glob(pattern)})
            for wav_file in wav_files:
                voice_key = wav_file.stem.lower()
                config_entry = metadata.pop(voice_key, None)
                config = config_entry[0] if config_entry else None
                base_dir = config_entry[1] if config_entry else wav_file.parent

                transcript_text: Optional[str] = None
                if config:
                    if config.prompt_text:
                        transcript_text = config.prompt_text.strip() or None
                    elif config.transcript_file:
                        transcript_path = _resolve_path(base_dir, config.transcript_file, label="Transcript file")
                        if transcript_path:
                            transcript_text = _read_transcript_file(transcript_path, warn_if_missing=True)
                if transcript_text is None:
                    transcript_candidate = wav_file.with_suffix(".txt")
                    transcript_text = _read_transcript_file(transcript_candidate, warn_if_missing=False)

                name = wav_file.stem
                if config and config.name.lower() != voice_key:
                    LOGGER.info(
                        "Voice definition '%s' overrides name to '%s'; using filename '%s' instead",
                        wav_file.name,
                        config.name,
                        name,
                    )
                description = config.description if config else f"Voice derived from {wav_file.name}"
                language = config.language if config else None
                tags = list(config.tags or []) if config else []

                profile = VoiceProfile(
                    name=name,
                    description=description,
                    prompt_audio=wav_file.resolve(),
                    prompt_text=transcript_text,
                    language=language,
                    tags=tags,
                )
                voices[name.lower()] = profile

        for voice_key, (config, base_dir) in metadata.items():
            prompt_audio_path = _resolve_path(base_dir, config.prompt_audio, label="Prompt audio")
            transcript_text = config.prompt_text.strip() if config.prompt_text else None
            if not transcript_text and config.transcript_file:
                transcript_path = _resolve_path(base_dir, config.transcript_file, label="Transcript file")
                if transcript_path:
                    transcript_text = _read_transcript_file(transcript_path, warn_if_missing=True)

            profile = VoiceProfile(
                name=config.name,
                description=config.description,
                prompt_audio=prompt_audio_path,
                prompt_text=transcript_text,
                language=config.language,
                tags=list(config.tags or []),
            )
            voices[voice_key] = profile

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

    def as_response(self) -> Dict[str, List[Dict[str, object]]]:
        """Return a serializable structure with all voices."""

        return {
            "data": [profile.to_response() for profile in self._voices.values()],
            "object": "list",
        }


__all__ = ["VoiceLibrary", "VoiceProfile"]
