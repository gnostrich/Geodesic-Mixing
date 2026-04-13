"""Tests for the energy computation.

Key invariants
--------------
1. E = 0 for a single track with time-invariant |CQT| (stationarity).
2. Gradients flow through the full forward pass (gradient check).
3. Higher-flux configurations have higher energy (magnitude ordering).
"""

import torch
import pytest

from src.config import MixConfig
from src.energy import compute_energy, _assemble_mix, _spectral_flux


# ---- helpers --------------------------------------------------------------

def _constant_cqt(B: int = 12, T: int = 200, mag: float = 1.0) -> torch.Tensor:
    """Complex CQT with constant magnitude (random but fixed phase)."""
    phase = torch.randn(B, T)
    return mag * torch.exp(1j * phase)


def _single_track_config(T: int, B: int = 12) -> MixConfig:
    cfg = MixConfig(n_tracks=1, cqt_lengths=[T], hop_length=512, sr=44100)
    return cfg


# ---- tests ----------------------------------------------------------------

class TestStationarity:
    """E must be zero for a spectrally-stationary signal."""

    def test_constant_magnitude_single_track(self):
        B, T = 12, 200
        cqt = _constant_cqt(B, T, mag=1.0)
        cfg = _single_track_config(T, B)

        E = compute_energy(cfg, [cqt], adaptive=False)
        assert E.item() == pytest.approx(0.0, abs=1e-5), \
            f"E should be 0 for constant |CQT|, got {E.item()}"

    def test_constant_magnitude_different_amplitudes(self):
        """Stationarity holds regardless of amplitude."""
        for mag in [0.1, 1.0, 10.0]:
            B, T = 12, 150
            cqt = _constant_cqt(B, T, mag=mag)
            cfg = _single_track_config(T, B)
            E = compute_energy(cfg, [cqt], adaptive=False)
            assert E.item() == pytest.approx(0.0, abs=1e-4), \
                f"mag={mag}: E={E.item()}"


class TestGradientFlow:
    """Gradients must reach all config parameters."""

    def test_gradients_exist_flat(self):
        B, T = 8, 100
        cqt = torch.randn(B, T, dtype=torch.complex64)
        cfg = MixConfig(n_tracks=1, cqt_lengths=[T])
        E = compute_energy(cfg, [cqt], adaptive=False)
        E.backward()

        for name, p in cfg.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), \
                f"All-zero gradient for {name}"

    def test_gradients_exist_two_tracks(self):
        B, T1, T2 = 8, 100, 120
        cqts = [torch.randn(B, T1, dtype=torch.complex64),
                torch.randn(B, T2, dtype=torch.complex64)]
        cfg = MixConfig(n_tracks=2, cqt_lengths=[T1, T2])
        with torch.no_grad():
            cfg.offsets.data[1] = 50 * cfg.frame_dur  # offset track 2
        E = compute_energy(cfg, cqts, adaptive=False)
        E.backward()

        for name, p in cfg.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_numerical_gradient(self):
        """Finite-difference check on gain knots (flat mode).

        We test gain knots rather than offsets because offsets go through
        a division by frame_dur and boundary effects that complicate
        finite-difference checks.  Gain knots affect energy smoothly.
        """
        B, T = 8, 200
        torch.manual_seed(42)
        cqt = torch.randn(B, T, dtype=torch.complex64)
        cfg = MixConfig(n_tracks=1, cqt_lengths=[T])
        # Move gain knots away from the [0,1] clamp boundaries
        # so both ±eps perturbations are in the differentiable interior.
        with torch.no_grad():
            cfg.gain_knots.data.fill_(0.8)

        E = compute_energy(cfg, [cqt], adaptive=False)
        E.backward()
        # Pick a middle gain knot (index 10 out of 20)
        analytic = cfg.gain_knots.grad[0, 10].item()

        eps = 1e-4
        with torch.no_grad():
            cfg.gain_knots.data[0, 10] += eps
        E_plus = compute_energy(cfg, [cqt], adaptive=False).item()
        with torch.no_grad():
            cfg.gain_knots.data[0, 10] -= 2 * eps
        E_minus = compute_energy(cfg, [cqt], adaptive=False).item()
        with torch.no_grad():
            cfg.gain_knots.data[0, 10] += eps  # restore

        numerical = (E_plus - E_minus) / (2 * eps)
        rel_err = abs(analytic - numerical) / (abs(numerical) + 1e-8)
        assert rel_err < 0.2, \
            f"Gradient mismatch: analytic={analytic:.6f}, numerical={numerical:.6f}"


class TestMagnitudeOrdering:
    """Configurations with more spectral change must have higher E."""

    def test_noisy_vs_smooth(self):
        B, T = 12, 200
        # Smooth: constant magnitude
        smooth_cqt = _constant_cqt(B, T)
        # Noisy: random fluctuating magnitude
        noisy_cqt = torch.randn(B, T, dtype=torch.complex64) * 5

        cfg_smooth = _single_track_config(T, B)
        cfg_noisy = _single_track_config(T, B)

        E_smooth = compute_energy(cfg_smooth, [smooth_cqt]).item()
        E_noisy = compute_energy(cfg_noisy, [noisy_cqt]).item()

        assert E_noisy > E_smooth, \
            f"Noisy E ({E_noisy}) should exceed smooth E ({E_smooth})"

    def test_scaled_noise(self):
        """Doubling flux amplitude should quadruple energy (squared)."""
        B, T = 12, 200
        torch.manual_seed(7)
        cqt1 = torch.randn(B, T, dtype=torch.complex64)
        cqt2 = cqt1 * 2

        cfg1 = _single_track_config(T, B)
        cfg2 = _single_track_config(T, B)

        E1 = compute_energy(cfg1, [cqt1]).item()
        E2 = compute_energy(cfg2, [cqt2]).item()

        ratio = E2 / (E1 + 1e-12)
        # Magnitude doubles → flux doubles → squared flux quadruples
        assert 3.0 < ratio < 5.0, \
            f"Expected ~4x scaling, got {ratio:.2f}"


class TestComplexSumBeforeMagnitude:
    """The mix must sum complex CQTs before taking magnitude."""

    def test_cancellation(self):
        """Two tracks with opposite phase should cancel → low E."""
        B, T = 8, 100
        cqt = torch.randn(B, T, dtype=torch.complex64)
        cqt_neg = -cqt  # perfect antiphase

        cfg = MixConfig(n_tracks=2, cqt_lengths=[T, T])
        # Both at same offset, full gain
        E_cancel = compute_energy(cfg, [cqt, cqt_neg]).item()

        # Single track reference
        cfg_single = MixConfig(n_tracks=1, cqt_lengths=[T])
        E_single = compute_energy(cfg_single, [cqt]).item()

        # Cancellation should produce near-zero energy
        assert E_cancel < E_single * 0.01, \
            f"Phase cancellation failed: E_cancel={E_cancel}, E_single={E_single}"
