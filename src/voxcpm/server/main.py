"""Command line entry-point for running the API server."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import uvicorn

from .app import create_app
from .config import ServerSettings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VoxCPM OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development only)")
    parser.add_argument("--model-id", dest="model_id", help="Hugging Face model identifier to load")
    parser.add_argument("--model-path", dest="model_path", help="Local path to a VoxCPM checkpoint directory")
    parser.add_argument("--cache-dir", dest="cache_dir", help="Custom cache directory for model downloads")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download models from remote hubs")
    parser.add_argument("--voices-dir", dest="voices_dir", help="Directory containing voice definitions")
    parser.add_argument("--no-denoiser", action="store_true", help="Disable the ZipEnhancer denoiser pipeline")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming responses")
    parser.add_argument("--log-level", dest="log_level", help="Logging level (debug, info, warning, ...)")
    return parser


def run() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    for field in ["model_id", "model_path", "cache_dir", "voices_dir", "log_level"]:
        value = getattr(args, field)
        if value:
            overrides[field] = value

    if args.local_files_only:
        overrides["local_files_only"] = True
    if args.no_denoiser:
        overrides["load_denoiser"] = False
        overrides["denoise_prompts"] = False
    if args.no_stream:
        overrides["allow_streaming"] = False

    settings = ServerSettings(**overrides)
    app = create_app(settings)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
