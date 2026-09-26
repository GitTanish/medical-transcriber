"""Tests for Whisper speech-to-text (ASR) service."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pytest
import soundfile as sf
import torch

from backend.config import settings
from backend.services.asr import (
    audio_tensor_to_wav_bytes,
    prompt_char_budget,
    transcribe_audio,
    trim_prompt_context,
)
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


def test_prompt_context_capped_to_provider_limit():
    """Verify the Whisper prompt never exceeds the provider's documented token cap."""
    mock_response = MagicMock()
    mock_response.text = "Patient reports chest pain"
    long_context = " ".join(["patient reports chest pain"] * 200)
    assert len(long_context) > settings.asr_prompt_max_chars

    with patch(
        "backend.services.asr.client.audio.transcriptions.create",
        return_value=mock_response,
    ) as mock_create:
        transcribe_audio(torch.zeros(16000, dtype=torch.float32), context=long_context)

    prompt = mock_create.call_args.kwargs["prompt"]
    assert prompt
    assert len(prompt) <= settings.asr_prompt_max_chars
    # The most recent speech is retained, and the trim snaps to a word boundary.
    assert long_context.endswith(prompt)
    assert not prompt.startswith(" ")


def test_prompt_char_budget_follows_settings(monkeypatch):
    """Verify the prompt budget is the tighter of context window and provider cap."""
    monkeypatch.setattr("backend.services.asr.settings.asr_context_chars", 120)
    assert prompt_char_budget() == 120

    monkeypatch.setattr("backend.services.asr.settings.asr_context_chars", 10_000)
    assert prompt_char_budget() == settings.asr_prompt_max_chars

    monkeypatch.setattr("backend.services.asr.settings.asr_prompt_max_chars", 0)
    assert prompt_char_budget() == 0
    assert trim_prompt_context("recent consultation context") is None


if __name__ == "__main__":
    test_audio_tensor_to_wav_bytes_conversion()
    test_transcribe_audio_with_mocked_groq_response()
    test_transcribe_audio_from_tensor_and_bytes()
    test_transcribe_audio_invalid_input_type()
    test_transcribe_audio_graceful_error_handling()
    test_prompt_context_capped_to_provider_limit()
    test_prompt_char_budget_follows_settings()
    print("All ASR unit tests passed successfully!")