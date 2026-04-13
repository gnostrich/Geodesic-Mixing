"""Forward pass and energy computation (flat and adaptive measures)."""

from __future__ import annotations

import torch
import numpy as np

from .config import MixConfig, _linear_interp, _cubic_hermite_interp, N_GAIN_KNOTS

# Maximum lag for inter-scale coupling (frames).
# ~500 frames at hop=512 / sr=44100 ≈ 5.8 s — phrase-length upper bound.
TAU_MAX = 500


# ---------------------------------------------------------------------------
# Core forward pass
# ---------------------------------------------------------------------------

def _assemble_mix(config: MixConfig,
                  cqts: list[torch.Tensor]) -> torch.Tensor:
    """Build the mixed magnitude spectrogram from the current config.

    1. Warp each track's CQT along its warped time axis.
    2. Multiply by interpolated gain.
    3. Sum complex contributions, *then* take magnitude.

    Returns ``P`` of shape ``[B, T_out]`` (magnitude).
    """
    T_out = config.output_length()
    B = cqts[0].shape[0]
    device = config.offsets.device

    out_frames = torch.arange(T_out, device=device, dtype=torch.float32)
    off = config.offset_frames()                          # [n_tracks]

    mix = torch.zeros(B, T_out, device=device, dtype=cqts[0].dtype)

    for i in range(config.n_tracks):
        T_i = cqts[i].shape[1]

        # Nominal local frame for every output frame
        nominal = out_frames - off[i]

        # Warp → warped local frame
        warped = config.warp_frames(i, nominal)

        # Soft validity mask — smooth sigmoid at both boundaries.
        # Avoids hard cutoffs that break gradient flow at track edges.
        warped_c = warped.clamp(0, T_i - 1 - 1e-4)
        mask = (torch.sigmoid((warped + 0.5) * 30.0)
                * torch.sigmoid((T_i - 0.5 - warped) * 30.0))

        # Linear interpolation of complex CQT along time axis
        lo = warped_c.long()
        hi = (lo + 1).clamp(max=T_i - 1)
        frac = (warped_c - lo.float()).unsqueeze(0)       # [1, T_out]

        track_cqt = cqts[i][:, lo] * (1 - frac) + cqts[i][:, hi] * frac

        # Gain (C1 envelope, evaluated at warped local position)
        gain_vals = torch.clamp(config.gain_knots[i], 0.0, 1.0)
        gain_knot_x = torch.linspace(0, T_i - 1, N_GAIN_KNOTS,
                                     device=device, dtype=torch.float32)
        gain = _cubic_hermite_interp(warped_c, gain_knot_x, gain_vals)
        gain = gain * mask                                # soft fade at boundaries
        gain = gain.clamp(0.0, 1.0)

        mix = mix + track_cqt * gain.unsqueeze(0)

    P = torch.abs(mix)                                    # magnitude AFTER sum
    return P


def _spectral_flux(P: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame magnitude difference.  ``[B, T-1]``."""
    return P[:, 1:] - P[:, :-1]


# ---------------------------------------------------------------------------
# Adaptive coupling weights
# ---------------------------------------------------------------------------

def compute_coupling_weights(delta: torch.Tensor,
                             tau_max: int = TAU_MAX) -> torch.Tensor:
    """Compute per-band adaptive weights omega_k from spectral flux.

    For each ordered pair ``(j, k)`` with ``j != k`` compute the normalised
    lagged cross-correlation of flux, select the peak lag via argmax (treated
    as fixed for the backward pass — straight-through), and accumulate the
    incoming coupling.  Then ``omega_k = exp(mean incoming C)``.

    Parameters
    ----------
    delta : Tensor [B, T]
        Spectral flux (frame differences of magnitude spectrogram).
    tau_max : int
        Maximum positive lag to consider.

    Returns
    -------
    omega : Tensor [B]
        Adaptive weights, bounded in ``[1/e, e]`` by construction.
    """
    B, T = delta.shape
    device = delta.device

    # FFT length for cross-correlation (next power of two)
    n_fft = 1 << int(np.ceil(np.log2(T + tau_max)))

    # Pre-compute FFT of each band's flux
    Delta_fft = torch.fft.rfft(delta, n=n_fft, dim=1)    # [B, F]
    norms = delta.norm(dim=1).clamp(min=1e-8)             # [B]

    mu = torch.zeros(B, device=device)

    for k in range(B):
        # C_{j→k}(τ) = Σ_n δ_{j,n} · δ_{k,n+τ}
        # = IFFT( conj(FFT(δ_j)) · FFT(δ_k) )[τ]  for positive τ
        cc_fft = Delta_fft.conj() * Delta_fft[k : k + 1]  # [B, F]
        cc = torch.fft.irfft(cc_fft, n=n_fft, dim=1)      # [B, n_fft]

        # Normalise by ||delta_j|| * ||delta_k||
        norm_jk = (norms * norms[k]).clamp(min=1e-8)       # [B]
        cc = cc / norm_jk.unsqueeze(1)

        # Positive lags 1 … tau_max
        cc_lags = cc[:, 1 : tau_max + 1]                   # [B, tau_max]

        # Peak lag per source band (not differentiable — straight-through)
        abs_cc = torch.abs(cc_lags)
        tau_star = torch.argmax(abs_cc, dim=1)              # [B]

        # Gather correlation value at selected lag
        c_vals = cc_lags[torch.arange(B, device=device), tau_star]  # [B]

        # Zero out self-coupling (j == k)
        c_vals = c_vals.clone()
        c_vals[k] = 0.0

        mu[k] = c_vals.sum() / (B - 1)

    omega = torch.exp(mu)
    return omega


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def compute_energy(config: MixConfig, cqts: list[torch.Tensor],
                   adaptive: bool = False) -> torch.Tensor:
    """Full forward pass → scalar energy *E*.

    Parameters
    ----------
    config : MixConfig
        Current (differentiable) configuration.
    cqts : list of Tensor [B, T_i]
        Pre-computed complex CQTs, one per track.
    adaptive : bool
        If True, use directed inter-scale coupling weights.

    Returns
    -------
    E : scalar Tensor
    """
    P = _assemble_mix(config, cqts)
    delta = _spectral_flux(P)

    if adaptive:
        omega = compute_coupling_weights(delta)
        E = (omega.unsqueeze(1) * delta ** 2).sum()
    else:
        E = (delta ** 2).sum()
    return E


def compute_energy_with_penalty(config: MixConfig, cqts: list[torch.Tensor],
                                adaptive: bool = False,
                                lam: float = 1.0) -> torch.Tensor:
    """Energy + soft constraint penalties (overlap / active-duration)."""
    E = compute_energy(config, cqts, adaptive=adaptive)

    off = config.offset_frames()
    frame_dur = config.frame_dur
    penalty = torch.tensor(0.0, device=E.device)

    for i in range(config.n_tracks):
        _, wv_i = config.warp_mapping(i)
        dur_i = wv_i[-1] * frame_dur
        # Penalise if active duration < 30 s
        shortfall = torch.relu(MIN_ACTIVE_S - dur_i)
        penalty = penalty + shortfall ** 2

        for j in range(i + 1, config.n_tracks):
            _, wv_j = config.warp_mapping(j)
            start_i, end_i = off[i], off[i] + wv_i[-1]
            start_j, end_j = off[j], off[j] + wv_j[-1]
            overlap = torch.relu(torch.min(end_i, end_j) - torch.max(start_i, start_j))
            max_overlap_frames = MAX_OVERLAP_S / frame_dur
            excess = torch.relu(overlap - max_overlap_frames)
            penalty = penalty + excess ** 2

    return E + lam * penalty


# Convenience re-exports for direct access from energy module
MIN_ACTIVE_S = 30.0
MAX_OVERLAP_S = 60.0
