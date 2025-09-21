"""OpenAI-compatible API server for VoxCPM."""

from .app import create_app
from .config import ServerSettings

__all__ = ["create_app", "ServerSettings"]
