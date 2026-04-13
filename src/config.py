"""Configuration vector: offsets, warps, gains.  Initialisation and clamping."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .cqt import estimate_bpm

N_WARP_KNOTS = 10
N_GAIN_KNOTS = 20
MIN_ACTIVE_S = 30.0
MAX_OVERLAP_S = 60.0


# ---------------------------------------------------------------------------
# Interpolation helpers (fully differentiable)
# ---------------------------------------------------------------------------

def _linear_interp(x: torch.Tensor, xp: torch.Tensor,
                   yp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation, differentiable w.r.t. *yp*.

    ``x``  – query points  [N]
    ``xp`` – knot positions [K]  (sorted, ascending)
    ``yp`` – knot values    [K]
    Returns interpolated values [N].
    """
    idx = torch.searchsorted(xp[1:].contiguous(), x.contiguous()).clamp(0, len(xp) - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = yp[idx], yp[idx + 1]
    t = ((x - x0) / (x1 - x0).clamp(min=1e-8)).clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


def _cubic_hermite_interp(x: torch.Tensor, xp: torch.Tensor,
                          yp: torch.Tensor) -> torch.Tensor:
    """Cubic Hermite (Catmull-Rom) interpolation — C1 smooth.

    Same signature as :func:`_linear_interp`.
    """
    K = xp.shape[0]

    # Tangents via finite differences (Catmull-Rom)
    m = torch.zeros_like(yp)
    m[1:-1] = (yp[2:] - yp[:-2]) / (xp[2:] - xp[:-2]).clamp(min=1e-8)
    m[0] = (yp[1] - yp[0]) / (xp[1] - xp[0]).clamp(min=1e-8)
    m[-1] = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]).clamp(min=1e-8)

    idx = torch.searchsorted(xp[1:].contiguous(), x.contiguous()).clamp(0, K - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = yp[idx], yp[idx + 1]
    m0, m1 = m[idx], m[idx + 1]
    h = (x1 - x0).clamp(min=1e-8)
    t = ((x - x0) / h).clamp(0.0, 1.0)

    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


# ---------------------------------------------------------------------------
# MixConfig
# ---------------------------------------------------------------------------

class MixConfig(nn.Module):
    """Differentiable mix configuration for *N* tracks.

    Parameters live on a shared device and are optimised jointly via
    ``E.backward()`` / Adam.
    """

    def __init__(self, n_tracks: int, cqt_lengths: list[int],
                 hop_length: int = 512, sr: int = 44100):
        super().__init__()
        self.n_tracks = n_tracks
        self.cqt_lengths = list(cqt_lengths)
        self.hop_length = hop_length
        self.sr = sr
        self.frame_dur = hop_length / sr          # seconds per CQT frame

        # Offsets in seconds (one scalar per track)
        self.offsets = nn.Parameter(torch.zeros(n_tracks))

        # Warp: log-rate at each knot.  exp(0) = 1 → identity warp.
        self.warp_log_rates = nn.Parameter(torch.zeros(n_tracks, N_WARP_KNOTS))

        # Gain envelope values at each knot (will be clamped to [0, 1]).
        self.gain_knots = nn.Parameter(torch.ones(n_tracks, N_GAIN_KNOTS))

    # ---- helpers ----------------------------------------------------------

    def offset_frames(self) -> torch.Tensor:
        """Offsets converted to CQT-frame units.  Shape ``[n_tracks]``."""
        return self.offsets / self.frame_dur

    def warp_mapping(self, i: int):
        """Return ``(knot_pos, warp_vals)`` for track *i* (both in frames).

        *knot_pos*  — uniform grid in *nominal* local time.
        *warp_vals* — corresponding positions in *warped* local time,
        obtained by integrating ``exp(warp_log_rates[i])``.
        """
        T = self.cqt_lengths[i]
        rates = torch.exp(self.warp_log_rates[i])               # [K]
        knots = torch.linspace(0, T - 1, N_WARP_KNOTS,
                               device=rates.device, dtype=rates.dtype)
        spacing = knots[1] - knots[0]
        avg = (rates[:-1] + rates[1:]) / 2.0
        increments = spacing * avg
        warp_vals = torch.cat([torch.zeros(1, device=rates.device),
                               torch.cumsum(increments, dim=0)])
        return knots, warp_vals

    def warp_frames(self, i: int, nominal: torch.Tensor) -> torch.Tensor:
        """Map *nominal* local frames → warped local frames for track *i*."""
        knots, warp_vals = self.warp_mapping(i)
        return _linear_interp(nominal, knots, warp_vals)

    def gain_at(self, i: int, local_frames: torch.Tensor) -> torch.Tensor:
        """Evaluate (C1) gain envelope for track *i* at *local_frames*."""
        T = self.cqt_lengths[i]
        gv = torch.clamp(self.gain_knots[i], 0.0, 1.0)
        knots = torch.linspace(0, T - 1, N_GAIN_KNOTS,
                               device=gv.device, dtype=gv.dtype)
        return _cubic_hermite_interp(local_frames, knots, gv)

    def output_length(self) -> int:
        """Total output length in CQT frames (detached)."""
        off = self.offset_frames()
        ends = []
        for i in range(self.n_tracks):
            _, wv = self.warp_mapping(i)
            ends.append(off[i] + wv[-1])
        return int(torch.stack(ends).max().detach().item()) + 1

    def clamp_gains(self):
        """Project gain knots back into [0, 1] (call after each step)."""
        with torch.no_grad():
            self.gain_knots.data.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Initialisers
# ---------------------------------------------------------------------------

def init_from_bpm(audio_paths: list[str], cqt_lengths: list[int],
                  hop_length: int = 512, sr: int = 44100) -> tuple[MixConfig, list[int]]:
    """BPM-ordered initialisation (deterministic seed config).

    Returns ``(config, track_order)`` where *track_order* is the BPM-sorted
    index permutation.
    """
    bpms = [estimate_bpm(p, sr=sr) for p in audio_paths]
    order = list(np.argsort(bpms))
    n = len(audio_paths)
    cfg = MixConfig(n, cqt_lengths, hop_length=hop_length, sr=sr)

    with torch.no_grad():
        cum = 0.0
        for track_idx in order:
            cfg.offsets.data[track_idx] = cum
            dur = cqt_lengths[track_idx] * cfg.frame_dur
            cum += max(dur - MAX_OVERLAP_S / 2, MIN_ACTIVE_S)
    return cfg, order


def init_random(n_tracks: int, cqt_lengths: list[int],
                track_order: list[int], hop_length: int = 512,
                sr: int = 44100, jitter_s: float = 2.0) -> MixConfig:
    """Random initialisation for basin discovery."""
    cfg = MixConfig(n_tracks, cqt_lengths, hop_length=hop_length, sr=sr)

    with torch.no_grad():
        cum = 0.0
        for track_idx in track_order:
            jitter = (torch.rand(1).item() - 0.5) * 2 * jitter_s
            cfg.offsets.data[track_idx] = cum + jitter
            dur = cqt_lengths[track_idx] * cfg.frame_dur
            cum += max(dur - MAX_OVERLAP_S / 2, MIN_ACTIVE_S)

        # Random gain seeds — interior knots uniform in [0.4, 1.0],
        # first/last fade to near-zero.
        cfg.gain_knots.data.uniform_(0.4, 1.0)
        cfg.gain_knots.data[:, 0].uniform_(0.0, 0.15)
        cfg.gain_knots.data[:, -1].uniform_(0.0, 0.15)

        # Small warp noise (±3 % tempo)
        cfg.warp_log_rates.data.normal_(0.0, 0.03)

    return cfg
