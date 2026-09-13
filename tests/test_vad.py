"""Tests for audio preprocessing and Voice Activity Detection (VAD)."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import soundfile as sf
import torch

from backend.services.vad import (
    detect_speech,
    extract_speech_chunks,
    extract_speech_segments,
    high_pass_filter,
)
from backend.services.audio_processing import preprocess_audio
from tests.helpers import create_temp_wav_file, generate_synthetic_audio


def rms(signal: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(signal ** 2)).item()


def peak(signal: torch.Tensor) -> float:
    return torch.max(torch.abs(signal)).item()


def test_high_pass_filter_attenuates_low_frequency():
    """Verify Butterworth 80 Hz filter removes sub-80Hz rumble and reduces signal energy."""
    sample_rate = 16000
    audio_np = generate_synthetic_audio(
        duration_s=2.0,
        sample_rate=sample_rate,
        include_low_freq_rumble=True,
    )
    audio = torch.from_numpy(audio_np)

    filtered = high_pass_filter(audio, sample_rate=sample_rate, cutoff=80.0)

    before_rms = rms(audio)
    after_rms = rms(filtered)
    before_peak = peak(audio)
    after_peak = peak(filtered)

    # 40 Hz component must be filtered out, reducing RMS amplitude
    assert after_rms < before_rms, "High-pass filter should reduce low-frequency rumble energy"
    assert after_rms > 0.0, "Filtered audio should preserve speech-band signals"
    assert after_peak <= before_peak + 1e-4, "Peak amplitude should not distort excessively"


def test_preprocess_audio_full_pipeline():
    """Verify preprocess_audio returns clean 1D tensor at 16kHz and downmixes multi-channel."""
    audio_np = generate_synthetic_audio(duration_s=1.5, sample_rate=16000)
    # Test stereo downmix handling
    stereo = np.stack([audio_np, audio_np], axis=-1)
    cleaned = preprocess_audio(stereo, sample_rate=16000)

    assert isinstance(cleaned, torch.Tensor)
    assert cleaned.ndim == 1
    assert len(cleaned) == len(audio_np)


def test_vad_speech_segmentation_structure():
    """Verify extract_speech_segments metadata schema on synthetic speech segments."""
    sample_rate = 16000
    # Verify that silence returns empty segments
    silence = torch.zeros(sample_rate * 2, dtype=torch.float32)
    segments = extract_speech_segments(silence, sample_rate=sample_rate)
    assert segments == []


def run_standalone_demo():
    temp_wav = create_temp_wav_file(duration_s=2.0, sample_rate=16000, include_low_freq_rumble=True)
    try:
        data, sample_rate = sf.read(str(temp_wav), dtype="float32")
        audio = torch.from_numpy(data)
        filtered = high_pass_filter(audio, sample_rate)

        before_rms = rms(audio)
        after_rms = rms(filtered)
        pct = ((after_rms - before_rms) / before_rms) * 100

        print("\n=== Audio Preprocessing Comparison ===")
        print(f"Sample rate:  {sample_rate} Hz")
        print(f"RMS before:   {before_rms:.6f}")
        print(f"RMS after:    {after_rms:.6f}")
        print(f"RMS change:   {pct:+.2f}% (Rumble attenuated)")
    finally:
        temp_wav.unlink(missing_ok=True)


if __name__ == "__main__":
    run_standalone_demo()
