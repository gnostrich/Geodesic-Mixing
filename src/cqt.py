"""CQT computation and disk caching."""

import hashlib
import os

import librosa
import numpy as np
import torch

# Default CQT parameters: 10 octaves from ~20 Hz to ~20 kHz
SR = 44100
HOP_LENGTH = 512
FMIN = 20.0
N_BINS = 120
BINS_PER_OCTAVE = 12


def _cache_key(audio_path: str, sr: int, hop_length: int,
               fmin: float, n_bins: int, bins_per_octave: int) -> str:
    stat = os.stat(audio_path)
    key_str = (f"{os.path.abspath(audio_path)}|{stat.st_mtime}|{stat.st_size}"
               f"|{sr}|{hop_length}|{fmin}|{n_bins}|{bins_per_octave}")
    return hashlib.sha256(key_str.encode()).hexdigest()


def compute_cqt(audio_path: str, sr: int = SR, hop_length: int = HOP_LENGTH,
                fmin: float = FMIN, n_bins: int = N_BINS,
                bins_per_octave: int = BINS_PER_OCTAVE,
                cache_dir: str = ".geodesic_cache") -> torch.Tensor:
    """Compute the CQT for an audio file with disk caching.

    Returns a complex-valued tensor of shape ``[n_bins, n_frames]``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(audio_path, sr, hop_length, fmin, n_bins, bins_per_octave)
    cache_path = os.path.join(cache_dir, f"{key}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        cqt_complex = data["real"] + 1j * data["imag"]
        return torch.from_numpy(cqt_complex.astype(np.complex64))

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    C = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin,
                    n_bins=n_bins, bins_per_octave=bins_per_octave)

    np.savez(cache_path,
             real=C.real.astype(np.float32),
             imag=C.imag.astype(np.float32))

    return torch.from_numpy(C.astype(np.complex64))


def load_audio(audio_path: str, sr: int = SR):
    """Load an audio waveform as a float32 tensor."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return torch.from_numpy(y.astype(np.float32)), sr


def estimate_bpm(audio_path: str, sr: int = SR) -> float:
    """Estimate BPM using librosa beat tracker."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
