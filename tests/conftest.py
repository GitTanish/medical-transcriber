"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pytest
import torch

from tests.helpers import (
    create_temp_wav_file,
    create_test_wav_bytes,
    generate_synthetic_audio,
)


@pytest.fixture
def synthetic_audio_array():
    """Returns a deterministic 2.0s 16kHz float32 audio numpy array."""
    return generate_synthetic_audio(duration_s=2.0, sample_rate=16000)


@pytest.fixture
def synthetic_audio_tensor(synthetic_audio_array):
    """Returns a 1D torch.Tensor representation of synthetic audio."""
    return torch.from_numpy(synthetic_audio_array)


@pytest.fixture
def synthetic_wav_bytes():
    """Returns in-memory PCM WAV bytes of synthetic audio."""
    return create_test_wav_bytes(duration_s=2.0, sample_rate=16000)


@pytest.fixture
def temp_wav_file():
    """Yields a temporary WAV file on disk and cleans it up after test completion."""
    path = create_temp_wav_file(duration_s=2.0, sample_rate=16000)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
