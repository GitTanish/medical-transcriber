import io
import logging
from pathlib import PurePosixPath
from typing import Annotated

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.config import settings
from backend.schemas.transcription import Segment, TranscriptionResponse
from backend.services.asr import (
    ASRServiceError,
    normalize_language_code,
    transcribe_audio_result,
    trim_prompt_context,
)
from backend.services.audio_processing import preprocess_audio
from backend.services.vad import extract_speech_segments

logger = logging.getLogger(__name__)
router = APIRouter()


class InvalidAudioError(ValueError):
    """Raised when an uploaded payload is not usable audio."""


def _safe_filename(filename: str | None) -> str:
    value = (filename or "audio.wav").replace("\\", "/")
    value = PurePosixPath(value).name.strip()
    return (value or "audio.wav")[:255]


def _normalize_language(language: str | None) -> str | None:
    if language is not None and len(language) > settings.max_language_chars:
        raise HTTPException(
            status_code=422,
            detail="language must be a valid two- or three-letter language code",
        )
    try:
        return normalize_language_code(language)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="language must be a valid two- or three-letter language code",
        ) from exc


def _mono_audio(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2:
        axis = 0 if values.shape[0] <= 8 and values.shape[1] > 8 else -1
        values = values.mean(axis=axis)
    elif values.ndim > 2:
        raise InvalidAudioError("Audio must contain one or two channels")
    if values.size == 0:
        raise InvalidAudioError("Audio file contains no samples")
    if not np.isfinite(values).all():
        raise InvalidAudioError("Audio contains non-finite samples")
    return np.ascontiguousarray(values.reshape(-1), dtype=np.float32)


def _transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str,
    language: str | None,
    context: str | None,
) -> TranscriptionResponse:
    try:
        data, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise InvalidAudioError("The uploaded file could not be decoded as audio") from exc

    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise InvalidAudioError("Audio has an invalid sample rate")
    if sample_rate > settings.max_input_sample_rate:
        raise InvalidAudioError("Audio sample rate is too high")

    audio_array = _mono_audio(data)
    if audio_array.size == 0:
        raise InvalidAudioError("Audio file contains no samples")

    duration_seconds = audio_array.size / float(sample_rate)
    if duration_seconds > settings.max_audio_duration_seconds:
        raise InvalidAudioError(
            f"Audio duration exceeds the {settings.max_audio_duration_seconds:g}-second limit"
        )
    total_duration = round(duration_seconds, 2)

    audio = torch.from_numpy(audio_array)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(
            orig_freq=int(sample_rate),
            new_freq=16000,
        )(audio)
        sample_rate = 16000

    cleaned_audio = preprocess_audio(
        audio,
        sample_rate=sample_rate,
        apply_high_pass=True,
        apply_noise_reduction=True,
        cutoff=80.0,
        prop_decrease=settings.noise_reduction_strength,
    )
    speech_segments = extract_speech_segments(cleaned_audio, sample_rate=sample_rate)
    transcript_parts: list[str] = []
    segments: list[Segment] = []
    detected_languages: list[str] = []
    rolling_context = context or ""

    for speech_segment in speech_segments:
        try:
            segment_audio = speech_segment["audio"]
            segment_start = float(speech_segment["start"])
            segment_end = float(speech_segment["end"])
            segment_id = int(speech_segment["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidAudioError("VAD returned an invalid speech segment") from exc

        asr_result = transcribe_audio_result(
            segment_audio,
            sample_rate=sample_rate,
            language=language,
            context=rolling_context,
            response_format="verbose_json",
            raise_on_error=True,
        )
        segment_text = asr_result.text.strip()
        if asr_result.language:
            detected_languages.append(asr_result.language)
        if segment_text:
            transcript_parts.append(segment_text)
            segments.append(
                Segment(
                    id=segment_id,
                    start=segment_start,
                    end=segment_end,
                    text=segment_text,
                    language=asr_result.language or language,
                )
            )
            rolling_context = trim_prompt_context(
                f"{rolling_context} {segment_text}"
            ) or ""

    detected_language = detected_languages[0] if language is None and detected_languages else language
    response_language = detected_language or language
    return TranscriptionResponse(
        text=" ".join(transcript_parts),
        filename=filename,
        language=response_language,
        detected_language=detected_language,
        duration=total_duration,
        segments=segments,
    )


@router.post("/", response_model=TranscriptionResponse)
async def create_transcription(
    file: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = "auto",
    context: Annotated[str | None, Form()] = None,
):
    """Decode, preprocess, detect speech, and transcribe an uploaded chunk."""
    normalized_language = _normalize_language(language)
    normalized_context = trim_prompt_context(context)
    try:
        audio_bytes = await file.read(settings.max_audio_bytes + 1)
    finally:
        await file.close()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
    if len(audio_bytes) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds the maximum upload size.",
        )

    try:
        return await run_in_threadpool(
            _transcribe_audio_bytes,
            audio_bytes,
            _safe_filename(file.filename),
            normalized_language,
            normalized_context,
        )
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Audio processing rejected the upload")
        raise HTTPException(status_code=400, detail="Audio could not be processed") from exc
    except ASRServiceError as exc:
        logger.error(
            "Transcription provider unavailable",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=503, detail="Transcription service unavailable") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected transcription failure",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
