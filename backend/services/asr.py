import io
import os
from pathlib import Path
from typing import Optional, Union

from groq import Groq
import numpy as np
import soundfile as sf
import torch

from backend.config import settings

client = Groq(api_key=settings.groq_api_key)


def audio_tensor_to_wav_bytes(audio: Union[torch.Tensor, np.ndarray], sample_rate: int = 16000) -> bytes:
    """Convert a 1D audio tensor or numpy array to in-memory WAV bytes."""
    if isinstance(audio, torch.Tensor):
        data = audio.detach().cpu().numpy()
    else:
        data = np.asarray(audio, dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def transcribe_audio(
    audio: Union[str, Path, bytes, torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    language: Optional[str] = "en",
    model: str = "whisper-large-v3-turbo",
) -> str:
    """
    Transcribe audio from a file path, raw bytes, or a torch.Tensor / np.ndarray speech chunk.
    """
    if isinstance(audio, (str, Path)):
        with open(audio, "rb") as f:
            file_payload = (Path(audio).name, f.read(), "audio/wav")
    elif isinstance(audio, (torch.Tensor, np.ndarray)):
        wav_bytes = audio_tensor_to_wav_bytes(audio, sample_rate=sample_rate)
        file_payload = ("chunk.wav", wav_bytes, "audio/wav")
    elif isinstance(audio, bytes):
        file_payload = ("audio.wav", audio, "audio/wav")
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio)}")

    kwargs = {
        "file": file_payload,
        "model": model,
        "response_format": "text",
    }
    if language:
        kwargs["language"] = language

    try:
        transcription = client.audio.transcriptions.create(**kwargs)
        return transcription if isinstance(transcription, str) else transcription.text
    except Exception as e:
        print(f"[ASR Warning] Transcription failed for audio segment: {e}")
        return ""