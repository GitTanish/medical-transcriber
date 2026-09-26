"""Integration tests for POST /api/transcribe/ endpoint."""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.asr import ASRResult, ASRServiceError
from tests.helpers import create_test_wav_bytes

client = TestClient(app)


def test_transcribe_endpoint():
    """Verify POST /api/transcribe/ accepts valid WAV uploads and returns TranscriptionResponse."""
    wav_bytes = create_test_wav_bytes(duration_s=2.0, sample_rate=16000)
    files = {"file": ("test_chunk.wav", wav_bytes, "audio/wav")}

    with patch("backend.api.transcription.extract_speech_segments", return_value=[]):
        response = client.post("/api/transcribe/", files=files, data={"language": "en"})
    assert response.status_code == 200

    data = response.json()
    assert "text" in data
    assert "duration" in data
    assert data["duration"] == 2.0
    assert "segments" in data
    assert isinstance(data["segments"], list)


def test_transcribe_endpoint_empty_file():
    """Verify POST /api/transcribe/ rejects empty uploads with HTTP 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/transcribe/", files=files, data={"language": "en"})
    assert response.status_code == 400
    assert "Empty audio file" in response.json()["detail"]


def test_transcribe_endpoint_with_mocked_speech_segments():
    """Verify speech segments returned by VAD are transcribed and accumulated in response."""
    wav_bytes = create_test_wav_bytes(duration_s=2.0, sample_rate=16000)
    files = {"file": ("test_chunk.wav", wav_bytes, "audio/wav")}

    mock_segments = [
        {
            "id": 0,
            "start": 0.2,
            "end": 1.8,
            "audio": torch.zeros(16000, dtype=torch.float32),
        }
    ]

    with patch(
        "backend.api.transcription.extract_speech_segments",
        return_value=mock_segments,
    ), patch(
        "backend.api.transcription.transcribe_audio_result",
        return_value=ASRResult(text="Patient has headache", language="en"),
    ):
        response = client.post("/api/transcribe/", files=files, data={"language": "en"})
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Patient has headache"
        assert len(data["segments"]) == 1
        assert data["segments"][0]["id"] == 0
        assert data["segments"][0]["text"] == "Patient has headache"


def test_transcribe_endpoint_auto_detects_indian_language_and_forwards_context():
    wav_bytes = create_test_wav_bytes(duration_s=2.0, sample_rate=16000)
    files = {"file": ("hindi.wav", wav_bytes, "audio/wav")}
    mock_segments = [
        {
            "id": 0,
            "start": 0.2,
            "end": 1.8,
            "audio": torch.zeros(16000, dtype=torch.float32),
        }
    ]
    with patch(
        "backend.api.transcription.extract_speech_segments",
        return_value=mock_segments,
    ), patch(
        "backend.api.transcription.transcribe_audio_result",
        return_value=ASRResult(text="नमस्ते डॉक्टर", language="hi"),
    ) as mock_transcribe:
        response = client.post(
            "/api/transcribe/",
            files=files,
            data={"language": "auto", "context": "पिछला वाक्य"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "hi"
    assert data["detected_language"] == "hi"
    assert data["segments"][0]["language"] == "hi"
    call_kwargs = mock_transcribe.call_args.kwargs
    assert call_kwargs["language"] is None
    assert call_kwargs["context"] == "पिछला वाक्य"
    assert call_kwargs["response_format"] == "verbose_json"


def test_transcribe_endpoint_rejects_invalid_audio():
    files = {"file": ("not-audio.wav", b"not audio", "audio/wav")}
    response = client.post("/api/transcribe/", files=files)
    assert response.status_code == 400
    assert "decoded" in response.json()["detail"]


def test_transcribe_endpoint_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr("backend.api.transcription.settings.max_audio_bytes", 8)
    files = {"file": ("large.wav", b"123456789", "audio/wav")}
    response = client.post("/api/transcribe/", files=files)
    assert response.status_code == 413
    assert "maximum upload size" in response.json()["detail"]


def test_transcribe_endpoint_rejects_invalid_language():
    files = {"file": ("test.wav", create_test_wav_bytes(), "audio/wav")}
    response = client.post("/api/transcribe/", files=files, data={"language": "not a language"})
    assert response.status_code == 422


def test_transcribe_endpoint_hides_provider_errors():
    wav_bytes = create_test_wav_bytes(duration_s=2.0, sample_rate=16000)
    files = {"file": ("test_chunk.wav", wav_bytes, "audio/wav")}
    mock_segments = [
        {
            "id": 0,
            "start": 0.2,
            "end": 1.8,
            "audio": torch.zeros(16000, dtype=torch.float32),
        }
    ]
    with patch(
        "backend.api.transcription.extract_speech_segments",
        return_value=mock_segments,
    ), patch(
        "backend.api.transcription.transcribe_audio_result",
        side_effect=ASRServiceError("provider secret details"),
    ):
        response = client.post("/api/transcribe/", files=files)
    assert response.status_code == 503
    assert "secret details" not in response.text
    assert response.json()["detail"] == "Transcription service unavailable"


def test_transcribe_endpoint_caps_context_before_provider(monkeypatch):
    """Verify context forwarded to the ASR provider is trimmed to the prompt budget."""
    monkeypatch.setattr("backend.api.transcription.settings.asr_prompt_max_chars", 200)
    wav_bytes = create_test_wav_bytes(duration_s=2.0, sample_rate=16000)
    files = {"file": ("long_context.wav", wav_bytes, "audio/wav")}
    mock_segments = [
        {
            "id": 0,
            "start": 0.2,
            "end": 1.8,
            "audio": torch.zeros(16000, dtype=torch.float32),
        }
    ]
    long_context = " ".join(["doctor asked about fever"] * 100)

    with patch(
        "backend.api.transcription.extract_speech_segments",
        return_value=mock_segments,
    ), patch(
        "backend.api.transcription.transcribe_audio_result",
        return_value=ASRResult(text="no fever reported", language="en"),
    ) as mock_transcribe:
        response = client.post(
            "/api/transcribe/",
            files=files,
            data={"language": "en", "context": long_context},
        )

    assert response.status_code == 200
    forwarded_context = mock_transcribe.call_args.kwargs["context"]
    assert forwarded_context
    assert len(forwarded_context) <= 200


if __name__ == "__main__":
    test_transcribe_endpoint()
    test_transcribe_endpoint_empty_file()
    test_transcribe_endpoint_with_mocked_speech_segments()
    test_transcribe_endpoint_caps_context_before_provider()
    print("All transcription API tests passed successfully!")
