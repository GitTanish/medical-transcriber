"""Tests for Whisper speech-to-text (ASR) service."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pytest
import soundfile as sf
import torch

from backend.services.asr import audio_tensor_to_wav_bytes, transcribe_audio
from tests.helpers import create_temp_wav_file, generate_synthetic_audio


def test_audio_tensor_to_wav_bytes_conversion():
    """Verify audio_tensor_to_wav_bytes converts tensor into a valid 16kHz PCM WAV byte stream."""
    sample_rate = 16000
    audio_data = generate_synthetic_audio(duration_s=1.0, sample_rate=sample_rate)
    tensor = torch.from_numpy(audio_data)

    wav_bytes = audio_tensor_to_wav_bytes(tensor, sample_rate=sample_rate)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 44  # Valid WAV header is 44 bytes

    # Verify byte stream can be decoded by soundfile
    import io
    data, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == sample_rate
    assert np.allclose(data, audio_data, atol=1e-3)


def test_transcribe_audio_with_mocked_groq_response():
    """Verify transcribe_audio dispatches payload to Groq Whisper and returns text."""
    expected_text = "Doctor hello Rahul good to see you"
    mock_response = MagicMock()
    mock_response.text = expected_text

    with patch("backend.services.asr.client.audio.transcriptions.create", return_value=mock_response) as mock_create:
        temp_wav = create_temp_wav_file(duration_s=1.0)
        try:
            result = transcribe_audio(str(temp_wav), language="en")
            assert result == expected_text
            assert mock_create.called
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "whisper-large-v3-turbo"
            assert call_kwargs["language"] == "en"
        finally:
            temp_wav.unlink(missing_ok=True)


def test_transcribe_audio_from_tensor_and_bytes():
    """Verify transcribe_audio accepts in-memory torch.Tensor and raw bytes."""
    mock_response = MagicMock()
    mock_response.text = "In-memory audio segment"

    with patch("backend.services.asr.client.audio.transcriptions.create", return_value=mock_response):
        # 1. Test with torch.Tensor
        tensor = torch.zeros(16000, dtype=torch.float32)
        text_from_tensor = transcribe_audio(tensor, sample_rate=16000)
        assert text_from_tensor == "In-memory audio segment"

        # 2. Test with raw bytes
        raw_bytes = b"RIFF....WAVEfmt ...."
        text_from_bytes = transcribe_audio(raw_bytes)
        assert text_from_bytes == "In-memory audio segment"


def test_transcribe_audio_invalid_input_type():
    """Verify unsupported audio types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported audio input type"):
        transcribe_audio(12345)


def test_transcribe_audio_graceful_error_handling():
    """Verify that external API errors fail forward and return an empty string."""
    with patch("backend.services.asr.client.audio.transcriptions.create", side_effect=Exception("API connection timeout")):
        result = transcribe_audio(torch.zeros(16000, dtype=torch.float32))
        assert result == ""


if __name__ == "__main__":
    test_audio_tensor_to_wav_bytes_conversion()
    test_transcribe_audio_with_mocked_groq_response()
    test_transcribe_audio_from_tensor_and_bytes()
    test_transcribe_audio_invalid_input_type()
    test_transcribe_audio_graceful_error_handling()
    print("All ASR unit tests passed successfully!")