"""Reusable test audio generators and fixtures for testing."""

import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def make_llm_completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def generate_synthetic_audio(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    include_silence: bool = True,
    include_low_freq_rumble: bool = True,
) -> np.ndarray:
    """
    Generate a deterministic synthetic 1D float32 audio signal.
    
    Structure:
    - Initial silence (0.4s)
    - Active multi-frequency signal in human voice band (300 Hz, 600 Hz)
    - Optional low-frequency rumble (40 Hz) below the 80 Hz high-pass cutoff
    - Trailing silence (0.4s)
    """
    total_samples = int(sample_rate * duration_s)
    audio = np.zeros(total_samples, dtype=np.float32)

    if include_silence and duration_s > 0.8:
        start_idx = int(sample_rate * 0.4)
        end_idx = int(sample_rate * (duration_s - 0.4))
    else:
        start_idx = 0
        end_idx = total_samples

    active_len = end_idx - start_idx
    if active_len > 0:
        t = np.arange(active_len, dtype=np.float32) / sample_rate
        # Primary voice-band harmonics
        voice_signal = (
            0.5 * np.sin(2 * np.pi * 300.0 * t)
            + 0.3 * np.sin(2 * np.pi * 600.0 * t)
        )
        # Low-frequency rumble for testing high-pass filtering (40 Hz)
        if include_low_freq_rumble:
            rumble = 0.25 * np.sin(2 * np.pi * 40.0 * t)
            voice_signal = voice_signal + rumble

        # Normalize amplitude to prevent clipping
        max_val = np.max(np.abs(voice_signal))
        if max_val > 0:
            voice_signal = (voice_signal / max_val) * 0.7

        audio[start_idx:end_idx] = voice_signal

    return audio.astype(np.float32)


def create_test_wav_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    include_low_freq_rumble: bool = True,
) -> bytes:
    """Generate in-memory 16-bit PCM WAV bytes."""
    samples = generate_synthetic_audio(
        duration_s=duration_s,
        sample_rate=sample_rate,
        include_low_freq_rumble=include_low_freq_rumble,
    )
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def create_temp_wav_file(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    include_low_freq_rumble: bool = True,
) -> Path:
    """Create a temporary WAV file on disk, returning its Path."""
    wav_bytes = create_test_wav_bytes(
        duration_s=duration_s,
        sample_rate=sample_rate,
        include_low_freq_rumble=include_low_freq_rumble,
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(wav_bytes)
        temp.flush()
        return Path(temp.name)
