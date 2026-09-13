"""Integration tests for POST /api/transcribe/ endpoint."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
from fastapi.testclient import TestClient
from backend.main import app
from tests.helpers import create_test_wav_bytes

client = TestClient(app)


def test_transcribe_endpoint():
    """Verify POST /api/transcribe/ accepts valid WAV uploads and returns TranscriptionResponse."""
    wav_bytes = create_test_wav_bytes(duration_s=2.0, sample_rate=16000)
    files = {"file": ("test_chunk.wav", wav_bytes, "audio/wav")}

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

    with patch("backend.api.transcription.extract_speech_segments", return_value=mock_segments):
        with patch("backend.api.transcription.transcribe_audio", return_value="Patient has headache"):
            response = client.post("/api/transcribe/", files=files, data={"language": "en"})
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "Patient has headache"
            assert len(data["segments"]) == 1
            assert data["segments"][0]["id"] == 0
            assert data["segments"][0]["text"] == "Patient has headache"


if __name__ == "__main__":
    test_transcribe_endpoint()
    test_transcribe_endpoint_empty_file()
    test_transcribe_endpoint_with_mocked_speech_segments()
    print("All transcription API tests passed successfully!")
