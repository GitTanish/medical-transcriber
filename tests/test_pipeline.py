"""Tests for the end-to-end audio processing and transcription pipeline."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import soundfile as sf
import torch
import torchaudio

from backend.services.audio_processing import preprocess_audio
from backend.services.vad import extract_speech_chunks
from backend.services.asr import transcribe_audio
from tests.helpers import create_temp_wav_file


def test_end_to_end_pipeline():
    """Verify audio loading, preprocessing, VAD segmentation, and ASR pipeline execution."""
    sample_rate = 16000
    temp_wav = create_temp_wav_file(duration_s=2.0, sample_rate=sample_rate, include_low_freq_rumble=True)

    try:
        # 1. Load audio
        data, sr = sf.read(str(temp_wav), dtype="float32")
        audio = torch.from_numpy(data)

        if audio.ndim > 1:
            audio = audio.mean(dim=-1)

        if sr != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
            sr = 16000

        # 2. Preprocess: High-Pass + Noise Reduction
        cleaned_audio = preprocess_audio(
            audio,
            sample_rate=sr,
            apply_high_pass=True,
            apply_noise_reduction=True,
            cutoff=80.0,
            prop_decrease=0.75,
        )
        assert isinstance(cleaned_audio, torch.Tensor)
        assert len(cleaned_audio) == len(audio)

        # 3. VAD segmentation
        speech_chunks = extract_speech_chunks(cleaned_audio, sample_rate=sr)
        assert isinstance(speech_chunks, list)

        # 4. Transcription with mocked Whisper
        mock_response = MagicMock()
        mock_response.text = "Sample consultation dialogue"
        with patch("backend.services.asr.client.audio.transcriptions.create", return_value=mock_response):
            transcript = transcribe_audio(cleaned_audio, sample_rate=sr)
            assert transcript == "Sample consultation dialogue"

    finally:
        temp_wav.unlink(missing_ok=True)


if __name__ == "__main__":
    test_end_to_end_pipeline()
    print("End-to-end pipeline test completed successfully!")
