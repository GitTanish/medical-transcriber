"""Audio preprocessing: filtering and noise reduction."""

from typing import Union
import numpy as np
from scipy.signal import butter, sosfilt
import noisereduce as nr
import torch


def high_pass_filter(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    cutoff: float = 80.0,
    order: int = 6,
) -> Union[np.ndarray, torch.Tensor]:
    """Apply a Butterworth high-pass filter to remove low-frequency rumble and mic handling noise."""
    is_torch = isinstance(audio, torch.Tensor)
    if is_torch:
        data = audio.detach().cpu().numpy()
    else:
        data = np.asarray(audio, dtype=np.float32)

    sos = butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, data).astype(np.float32)

    if is_torch:
        return torch.from_numpy(filtered)
    return filtered


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    prop_decrease: float = 0.75,
) -> np.ndarray:
    """Apply moderate stationary noise reduction without clipping speech consonants."""
    if len(audio) < 1024 or np.max(np.abs(audio)) < 1e-5:
        return audio.astype(np.float32)

    try:
        cleaned = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=prop_decrease,
        )
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned.astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def preprocess_audio(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    apply_high_pass: bool = True,
    apply_noise_reduction: bool = True,
    cutoff: float = 80.0,
    prop_decrease: float = 0.75,
) -> torch.Tensor:
    """Full preprocessing pipeline: high-pass -> noise reduction -> 1D torch.Tensor."""
    is_torch = isinstance(audio, torch.Tensor)
    if is_torch:
        data = audio.detach().cpu().numpy()
    else:
        data = np.asarray(audio, dtype=np.float32)

    # Downmix to mono if multi-channel
    if data.ndim > 1:
        data = data.mean(axis=-1 if data.shape[-1] < data.shape[0] else 0)

    data = data.squeeze().astype(np.float32)

    if apply_high_pass:
        data = high_pass_filter(data, sample_rate=sample_rate, cutoff=cutoff)

    if apply_noise_reduction:
        data = reduce_noise(data, sample_rate=sample_rate, prop_decrease=prop_decrease)

    return torch.from_numpy(data)
