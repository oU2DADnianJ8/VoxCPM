import sys

import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

sys.path.append("src")

import voxcpm.server.app as server_app
from voxcpm.server.config import ServerSettings


class DummyTranscriber:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        DummyTranscriber.instances.append(self)

    async def transcribe(self, audio_path: str) -> str:
        self.calls.append(audio_path)
        return "auto transcript"


def _patch_model_loading():
    def fake_load_model(self):
        class _DummyModel:
            def __init__(self):
                self.tts_model = type("tts", (), {"sample_rate": 16000})()

            def generate(self, **kwargs):
                return np.zeros(16000, dtype=np.float32)

        self._model = _DummyModel()
        self._sample_rate = 16000

    return patch("voxcpm.server.tts.VoxCPMTTSManager._load_model", fake_load_model)


def test_voice_prompt_auto_transcription(tmp_path):
    voice_wav = tmp_path / "myvoice.wav"
    voice_wav.write_bytes(b"fake")

    settings = ServerSettings(voices_dir=str(tmp_path), output_dir=str(tmp_path / "generated"))
    DummyTranscriber.instances = []

    with patch("voxcpm.server.app.PromptTranscriber", DummyTranscriber), _patch_model_loading():
        app = server_app.create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": settings.openai_model_name,
                    "voice": "myvoice",
                    "input": "Hello there",
                    "response_format": "pcm",
                },
            )

            assert response.status_code == 200
            saved_files = list(settings.output_dir.glob("*.wav"))
            assert saved_files
    assert DummyTranscriber.instances
    assert DummyTranscriber.instances[0].calls


def test_voice_prompt_requires_text_when_asr_disabled(tmp_path):
    voice_wav = tmp_path / "voice.wav"
    voice_wav.write_bytes(b"fake")

    settings = ServerSettings(
        voices_dir=str(tmp_path), enable_prompt_asr=False, output_dir=str(tmp_path / "generated")
    )
    DummyTranscriber.instances = []

    with _patch_model_loading():
        app = server_app.create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": settings.openai_model_name,
                    "voice": "voice",
                    "input": "Testing",
                    "response_format": "pcm",
                },
            )

    assert response.status_code == 400
    assert "requires an accompanying transcript" in response.json()["error"]["message"]
