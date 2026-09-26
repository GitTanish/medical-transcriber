from typing import TypeAlias

import noisereduce as nr
import numpy as np
import torch
from scipy.signal import butter, sosfilt

AudioInput: TypeAlias = np.ndarray | torch.Tensor


def _to_mono(audio: AudioInput) -> np.ndarray:
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
        raise ValueError("Audio input must contain at least one sample")
    return np.clip(
        np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0),
        -1.0,
        1.0,
    ).reshape(-1)


def high_pass_filter(
    audio: AudioInput,
    sample_rate: int = 16000,
    cutoff: float = 80.0,
    order: int = 6,
) -> AudioInput:
    """Apply a Butterworth high-pass filter to remove low-frequency noise."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 < cutoff < sample_rate / 2:
        raise ValueError("cutoff must be between 0 and the Nyquist frequency")
    if order <= 0:
        raise ValueError("order must be positive")

    is_torch = isinstance(audio, torch.Tensor)
    data = _to_mono(audio)
    sos = butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, data).astype(np.float32, copy=False)
    filtered = np.ascontiguousarray(filtered)
    return torch.from_numpy(filtered.copy()) if is_torch else filtered


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    prop_decrease: float = 0.75,
) -> np.ndarray:
    """Apply stationary noise reduction while preserving speech transients."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 <= prop_decrease <= 1:
        raise ValueError("prop_decrease must be between 0 and 1")

    data = _to_mono(audio)
    if data.size < 1024 or np.max(np.abs(data)) < 1e-5:
        return data.astype(np.float32, copy=False)

    try:
        cleaned = nr.reduce_noise(y=data, sr=sample_rate, prop_decrease=prop_decrease)
        return np.nan_to_num(
            np.asarray(cleaned, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
    except (FloatingPointError, RuntimeError, TypeError, ValueError):
        return data.astype(np.float32, copy=False)


def preprocess_audio(
    audio: AudioInput,
    sample_rate: int = 16000,
    apply_high_pass: bool = True,
    apply_noise_reduction: bool = True,
    cutoff: float = 80.0,
    prop_decrease: float = 0.75,
) -> torch.Tensor:
    """Run high-pass filtering and noise reduction, returning mono float32 audio."""
    data = _to_mono(audio)
    if apply_high_pass:
        data = high_pass_filter(data, sample_rate=sample_rate, cutoff=cutoff)
    if apply_noise_reduction:
        data = reduce_noise(data, sample_rate=sample_rate, prop_decrease=prop_decrease)
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
