import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from groq import Groq

from backend.config import settings

logger = logging.getLogger(__name__)
client = Groq(api_key=settings.groq_api_key or "not-configured")
LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}$")
INDIAN_LANGUAGES = {
    "hi": "Hindi",
    "ur": "Urdu",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "as": "Assamese",
    "ne": "Nepali",
    "si": "Sinhala",
    "sd": "Sindhi",
}


class ASRServiceError(RuntimeError):
    """Raised when the speech-to-text provider cannot complete a request."""


class UnsupportedAudioTypeError(ValueError, TypeError):
    """Raised when an ASR input has an unsupported runtime type."""


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)


def normalize_language_code(language: str | None) -> str | None:
    """Normalize a language tag to the base code expected by Whisper."""
    if language is None:
        return None
    value = language.strip().lower().replace("_", "-")
    if not value or value == "auto":
        return None
    base_code = value.split("-", 1)[0]
    if not LANGUAGE_PATTERN.fullmatch(base_code):
        raise ValueError("language must be a valid two- or three-letter language code")
    return base_code


def prompt_char_budget() -> int:
    """Return the character budget allowed for the Whisper prompt parameter.

    Groq documents a 224-token ceiling on transcription prompts and rejects
    requests that exceed it. The rolling transcript context is therefore capped
    by both the configured context window and this provider-side ceiling.
    """
    return max(0, min(settings.asr_context_chars, settings.asr_prompt_max_chars))


def trim_prompt_context(text: str | None) -> str | None:
    """Trim rolling transcript context to a length the ASR provider accepts.

    The most recent speech is preserved, and the cut snaps forward to the next
    word boundary so the prompt never begins mid-word.
    """
    if not text:
        return None
    budget = prompt_char_budget()
    if budget <= 0:
        return None
    trimmed = text.strip()
    if len(trimmed) > budget:
        trimmed = trimmed[-budget:]
        space_index = trimmed.find(" ")
        if 0 <= space_index < len(trimmed) - 1:
            trimmed = trimmed[space_index + 1 :]
    return trimmed or None


def _mono_float32(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        data = audio.detach().cpu().numpy()
    else:
        data = np.asarray(audio)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 0:
        data = data.reshape(1)
    elif data.ndim == 2:
        axis = 0 if data.shape[0] < data.shape[-1] else -1
        data = data.mean(axis=axis)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1).mean(axis=1)
    if data.size == 0:
        raise ValueError("Audio must contain at least one sample")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    return np.clip(data, -1.0, 1.0)


def audio_tensor_to_wav_bytes(
    audio: torch.Tensor | np.ndarray,
    sample_rate: int = 16000,
) -> bytes:
    """Convert a mono audio tensor or array to an in-memory PCM WAV payload."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    data = _mono_float32(audio)
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _audio_file_payload(
    audio: str | Path | bytes | torch.Tensor | np.ndarray,
    sample_rate: int,
) -> tuple[str, bytes, str]:
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        return path.name, path.read_bytes(), "audio/octet-stream"
    if isinstance(audio, (torch.Tensor, np.ndarray)):
        return (
            "chunk.wav",
            audio_tensor_to_wav_bytes(audio, sample_rate=sample_rate),
            "audio/wav",
        )
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("Audio bytes must not be empty")
        return "audio.wav", audio, "audio/octet-stream"
    raise UnsupportedAudioTypeError(f"Unsupported audio input type: {type(audio)}")


def _response_field(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _parse_segments(raw_segments: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, (list, tuple)):
        return []
    segments: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        if isinstance(raw_segment, dict):
            start = raw_segment.get("start")
            end = raw_segment.get("end")
            text = raw_segment.get("text")
        else:
            start = getattr(raw_segment, "start", None)
            end = getattr(raw_segment, "end", None)
            text = getattr(raw_segment, "text", None)
        if not isinstance(text, str):
            continue
        segments.append(
            {
                "start": float(start) if isinstance(start, (int, float)) else None,
                "end": float(end) if isinstance(end, (int, float)) else None,
                "text": text.strip(),
            }
        )
    return segments


def _parse_result(response: Any, fallback_language: str | None) -> ASRResult:
    raw_text = response if isinstance(response, str) else _response_field(response, "text", "")
    text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
    raw_language = _response_field(response, "language")
    detected_language: str | None = None
    if isinstance(raw_language, str):
        try:
            detected_language = normalize_language_code(raw_language)
        except ValueError:
            detected_language = None
    return ASRResult(
        text=text.strip(),
        language=detected_language or fallback_language,
        segments=_parse_segments(_response_field(response, "segments")),
    )


def transcribe_audio_result(
    audio: str | Path | bytes | torch.Tensor | np.ndarray,
    sample_rate: int = 16000,
    language: str | None = "auto",
    model: str | None = None,
    context: str | None = None,
    response_format: str = "text",
    raise_on_error: bool = False,
) -> ASRResult:
    """Transcribe audio and return text, detected language, and optional segments."""
    normalized_language = normalize_language_code(language)
    file_payload = _audio_file_payload(audio, sample_rate)
    kwargs: dict[str, object] = {
        "file": file_payload,
        "model": model or settings.asr_model,
        "response_format": response_format,
        "temperature": 0.0,
        "timeout": settings.provider_timeout_seconds,
    }
    if normalized_language:
        kwargs["language"] = normalized_language
    context_text = trim_prompt_context(context)
    if context_text:
        kwargs["prompt"] = context_text
    if response_format == "verbose_json":
        kwargs["timestamp_granularities"] = ["segment"]

    try:
        response = client.audio.transcriptions.create(**kwargs)
        return _parse_result(response, normalized_language)
    except Exception as exc:
        if raise_on_error:
            raise ASRServiceError("ASR provider request failed") from exc
        logger.warning(
            "ASR transcription failed",
            extra={"error_type": type(exc).__name__},
        )
        return ASRResult(text="")


def transcribe_audio(
    audio: str | Path | bytes | torch.Tensor | np.ndarray,
    sample_rate: int = 16000,
    language: str | None = "auto",
    model: str | None = None,
    raise_on_error: bool = False,
    context: str | None = None,
    response_format: str = "text",
) -> str:
    """Transcribe supported audio input through the configured ASR provider."""
    return transcribe_audio_result(
        audio,
        sample_rate=sample_rate,
        language=language,
        model=model,
        context=context,
        response_format=response_format,
        raise_on_error=raise_on_error,
    ).text
