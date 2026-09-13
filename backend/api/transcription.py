"""Audio/transcription API endpoints."""

import io
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import soundfile as sf
import torch
import torchaudio

from backend.schemas.transcription import Segment, TranscriptionResponse
from backend.services.audio_processing import preprocess_audio
from backend.services.vad import extract_speech_segments
from backend.services.asr import transcribe_audio

router = APIRouter()


@router.post("/", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
):
    """
    Full pipeline:
    Upload -> Decode -> Preprocessing (High-Pass + Noise Reduction) -> Silero VAD -> Whisper ASR
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file provided.")

        # Decode audio into float32 samples (supports WAV, MP3, OGG, FLAC)
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        audio = torch.from_numpy(data)

        # Stereo -> Mono downmix
        if audio.ndim > 1:
            audio = audio.mean(dim=-1)

        total_duration = round(len(audio) / sample_rate, 2)

        # Resample to 16,000 Hz if required
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
            sample_rate = 16000

        # Preprocessing: 80Hz High-Pass + Moderate Noise Reduction
        cleaned_audio = preprocess_audio(
            audio,
            sample_rate=sample_rate,
            apply_high_pass=True,
            apply_noise_reduction=True,
            cutoff=80.0,
            prop_decrease=0.75,
        )

        # VAD speech segmentation
        speech_segments = extract_speech_segments(cleaned_audio, sample_rate=sample_rate)

        if not speech_segments:
            return TranscriptionResponse(
                text="",
                filename=file.filename or "audio.wav",
                language=language,
                duration=total_duration,
                segments=[],
            )

        # Transcribe speech segments
        segments = []
        transcript_parts = []

        for seg in speech_segments:
            seg_text = transcribe_audio(
                seg["audio"],
                sample_rate=sample_rate,
                language=language if language else None,
            ).strip()

            if seg_text:
                transcript_parts.append(seg_text)
                segments.append(
                    Segment(
                        id=seg["id"],
                        start=seg["start"],
                        end=seg["end"],
                        text=seg_text,
                    )
                )

        full_text = " ".join(transcript_parts)

        return TranscriptionResponse(
            text=full_text,
            filename=file.filename or "audio.wav",
            language=language,
            duration=total_duration,
            segments=segments,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
