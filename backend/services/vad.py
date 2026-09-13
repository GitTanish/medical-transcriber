import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from backend.services.audio_processing import high_pass_filter


model = load_silero_vad()


def detect_speech(audio: torch.Tensor, sample_rate: int = 16000):
    """
    Detect speech regions in an audio signal.
    """
    return get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
    )


def extract_speech_chunks(
    audio: torch.Tensor,
    sample_rate: int = 16000,
):
    """
    Return only the portions of the audio where speech was detected.
    """
    speech_timestamps = detect_speech(audio, sample_rate)

    chunks = []
    for segment in speech_timestamps:
        start = segment["start"]
        end = segment["end"]
        chunks.append(audio[start:end])

    return chunks


def extract_speech_segments(
    audio: torch.Tensor,
    sample_rate: int = 16000,
):
    """
    Return segments with metadata (seconds and sample indices) and audio tensors.
    """
    speech_timestamps = detect_speech(audio, sample_rate)
    segments = []
    min_samples = int(sample_rate * 0.25)  # Require at least 250ms of audio

    for i, seg in enumerate(speech_timestamps):
        start = seg["start"]
        end = seg["end"]
        if (end - start) < min_samples:
            continue

        segments.append({
            "id": i,
            "start": round(start / sample_rate, 2),
            "end": round(end / sample_rate, 2),
            "audio": audio[start:end],
        })
    return segments
