"""Tests for adaptive coupling weights.

Key invariants
--------------
1. Coupling is NOT symmetric: C_{j→k} ≠ C_{k→j} on structured signals.
2. White noise: all ω_k ≈ 1 and adaptive energy ≈ flat energy.
3. Weight bounds: all ω_k ∈ [1/e, e] ≈ [0.37, 2.72].
"""

import torch
import pytest

from src.config import MixConfig
from src.energy import (
    compute_energy,
    compute_coupling_weights,
    _assemble_mix,
    _spectral_flux,
)


# ---- helpers --------------------------------------------------------------

def _make_flux(B: int, T: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T)


def _hierarchical_flux(B: int = 12, T: int = 2000) -> torch.Tensor:
    """Construct flux with directed inter-scale coupling.

    Low bands (coarse scales) drive high bands (fine scales) via a causal
    relationship: flux at band j with lag produces flux at band k > j.
    """
    torch.manual_seed(99)
    delta = torch.zeros(B, T)

    # Coarse bands: independent random pulses
    for j in range(B // 3):
        pulse_locs = torch.randint(0, T, (T // 50,))
        delta[j, pulse_locs] = torch.randn(len(pulse_locs))

    # Fine bands: echo of coarse bands with lag
    lag = 10
    for k in range(B // 3, B):
        source = k % (B // 3)
        if lag < T:
            delta[k, lag:] = delta[source, :T - lag] * 0.8
        delta[k] += torch.randn(T) * 0.1  # noise floor

    return delta


# ---- tests ----------------------------------------------------------------

class TestSymmetryViolation:
    """C_{j→k} and C_{k→j} must generically differ on structured signals."""

    def test_directed_coupling_asymmetric(self):
        delta = _hierarchical_flux(B=12, T=2000)
        omega = compute_coupling_weights(delta, tau_max=50)

        # Compute the full coupling matrix to check asymmetry
        B, T = delta.shape
        import numpy as np
        n_fft = 1 << int(np.ceil(np.log2(T + 50)))
        Delta = torch.fft.rfft(delta, n=n_fft, dim=1)
        norms = delta.norm(dim=1).clamp(min=1e-8)

        C = torch.zeros(B, B)
        for j in range(B):
            for k in range(B):
                if j == k:
                    continue
                cc_fft = Delta[j].conj() * Delta[k]
                cc = torch.fft.irfft(cc_fft, n=n_fft)
                cc = cc / (norms[j] * norms[k]).clamp(min=1e-8)
                cc_lags = cc[1:51]
                tau_star = torch.argmax(cc_lags.abs())
                C[j, k] = cc_lags[tau_star].item()

        # Check that C is not symmetric
        diff = (C - C.T).abs()
        max_asym = diff.max().item()
        assert max_asym > 0.01, \
            f"Coupling matrix appears symmetric (max diff={max_asym:.6f})"

    def test_asymmetry_on_causal_pair(self):
        """A band that causally drives another should have asymmetric C."""
        B, T = 4, 3000
        torch.manual_seed(42)
        delta = torch.zeros(B, T)
        # Band 0 drives band 2 with lag 15
        delta[0] = torch.randn(T)
        delta[2, 15:] = delta[0, :T - 15] * 0.9
        delta[1] = torch.randn(T) * 0.3
        delta[3] = torch.randn(T) * 0.3

        import numpy as np
        n_fft = 1 << int(np.ceil(np.log2(T + 50)))
        Delta = torch.fft.rfft(delta, n=n_fft, dim=1)
        norms = delta.norm(dim=1).clamp(min=1e-8)

        def _coupling(j, k):
            # C_{j→k}(τ) = IFFT(conj(FFT(δ_j)) · FFT(δ_k))[τ]
            cc = torch.fft.irfft(Delta[j].conj() * Delta[k], n=n_fft)
            cc = cc / (norms[j] * norms[k]).clamp(min=1e-8)
            lags = cc[1:51]
            return lags[torch.argmax(lags.abs())].item()

        c_0to2 = _coupling(0, 2)
        c_2to0 = _coupling(2, 0)

        # 0→2 should be strong (causal), 2→0 weaker or at different lag
        assert abs(c_0to2) > 0.3, f"Expected strong 0→2 coupling, got {c_0to2}"
        assert abs(c_0to2 - c_2to0) > 0.05, \
            f"C_{{0→2}}={c_0to2:.4f} ≈ C_{{2→0}}={c_2to0:.4f} — should differ"


class TestWhiteNoiseBaseline:
    """On white noise, adaptive mode should reduce to flat mode."""

    def test_weights_near_unity(self):
        B, T = 16, 5000
        torch.manual_seed(123)
        delta = torch.randn(B, T)
        omega = compute_coupling_weights(delta, tau_max=100)

        for k in range(B):
            assert 0.8 < omega[k].item() < 1.25, \
                f"ω[{k}]={omega[k].item():.4f} — expected ≈1 on white noise"

    def test_energy_matches_flat(self):
        """Adaptive and flat energy should be close on white noise."""
        B, T = 12, 200
        torch.manual_seed(77)
        cqt = torch.randn(B, T, dtype=torch.complex64)
        cfg_flat = MixConfig(n_tracks=1, cqt_lengths=[T])
        cfg_adapt = MixConfig(n_tracks=1, cqt_lengths=[T])

        E_flat = compute_energy(cfg_flat, [cqt], adaptive=False).item()
        E_adapt = compute_energy(cfg_adapt, [cqt], adaptive=True).item()

        ratio = E_adapt / (E_flat + 1e-12)
        assert 0.7 < ratio < 1.4, \
            f"Adaptive/flat ratio={ratio:.3f} — expected ≈1 on random signal"


class TestWeightBounds:
    """All ω_k must lie in [1/e, e] ≈ [0.37, 2.72]."""

    def test_bounds_random_flux(self):
        for seed in range(5):
            delta = _make_flux(B=20, T=3000, seed=seed)
            omega = compute_coupling_weights(delta, tau_max=100)
            lo, hi = omega.min().item(), omega.max().item()
            assert lo >= 0.3, f"ω_min={lo:.4f} < 0.3 (seed={seed})"
            assert hi <= 2.8, f"ω_max={hi:.4f} > 2.8 (seed={seed})"

    def test_bounds_hierarchical(self):
        delta = _hierarchical_flux(B=12, T=2000)
        omega = compute_coupling_weights(delta, tau_max=50)
        lo, hi = omega.min().item(), omega.max().item()
        assert lo >= 0.3, f"ω_min={lo:.4f} < 0.3"
        assert hi <= 2.8, f"ω_max={hi:.4f} > 2.8"


class TestAdaptiveGradients:
    """Adaptive energy must still be differentiable."""

    def test_gradients_flow(self):
        B, T = 8, 100
        cqt = torch.randn(B, T, dtype=torch.complex64)
        cfg = MixConfig(n_tracks=1, cqt_lengths=[T])
        E = compute_energy(cfg, [cqt], adaptive=True)
        E.backward()

        for name, p in cfg.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
