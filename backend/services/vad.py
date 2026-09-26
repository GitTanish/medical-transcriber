from typing import Any

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from backend.config import settings
from backend.services.audio_processing import high_pass_filter

__all__ = [
    "detect_speech",
    "extract_speech_chunks",
    "extract_speech_segments",
    "high_pass_filter",
]

model = load_silero_vad()


def _prepare_audio(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports only 8000 Hz or 16000 Hz audio")
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio, dtype=torch.float32)
    if audio.ndim != 1 or audio.numel() == 0:
        raise ValueError("VAD audio must be a non-empty one-dimensional tensor")
    return audio.to(dtype=torch.float32).contiguous()


def detect_speech(
    audio: torch.Tensor,
    sample_rate: int = 16000,
) -> list[dict[str, Any]]:
    """Detect speech regions in an audio signal."""
    prepared = _prepare_audio(audio, sample_rate)
    timestamps = get_speech_timestamps(
        prepared,
        model,
        threshold=settings.vad_threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=settings.vad_min_speech_duration_ms,
        min_silence_duration_ms=settings.vad_min_silence_duration_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )
    return [
        {
            "start": max(0, min(int(segment["start"]), prepared.numel())),
            "end": max(0, min(int(segment["end"]), prepared.numel())),
        }
        for segment in timestamps
    ]


def extract_speech_chunks(
    audio: torch.Tensor,
    sample_rate: int = 16000,
) -> list[torch.Tensor]:
    """Return only the portions of the audio where speech was detected."""
    prepared = _prepare_audio(audio, sample_rate)
    return [
        prepared[segment["start"] : segment["end"]]
        for segment in detect_speech(prepared, sample_rate)
        if segment["end"] > segment["start"]
    ]


def extract_speech_segments(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    min_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Return speech segments with metadata and their audio tensors."""
    if min_duration is None:
        min_duration = settings.vad_min_segment_seconds
    if min_duration <= 0:
        raise ValueError("min_duration must be positive")
    prepared = _prepare_audio(audio, sample_rate)
    min_samples = int(sample_rate * min_duration)
    segments: list[dict[str, Any]] = []

    for segment in detect_speech(prepared, sample_rate):
        start = segment["start"]
        end = segment["end"]
        if end - start < min_samples:
            continue
        segments.append(
            {
                "id": len(segments),
                "start": round(start / sample_rate, 2),
                "end": round(end / sample_rate, 2),
                "audio": prepared[start:end],
            }
        )
    return segments
