"""Gradient descent loop and basin discovery."""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass, field

import torch

from .config import MixConfig, init_random, N_WARP_KNOTS
from .energy import compute_energy, compute_energy_with_penalty


@dataclass
class Basin:
    """A converged local minimum (one valid mix)."""
    config: MixConfig
    energy: float
    track_order: list[int]
    transition_points: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        order_str = " → ".join(str(i) for i in self.track_order)
        trans = ", ".join(f"{t:.1f}s" for t in self.transition_points)
        return f"Basin(E={self.energy:.2f}, order=[{order_str}], transitions=[{trans}])"


# ---------------------------------------------------------------------------
# Single descent
# ---------------------------------------------------------------------------

def descend(config: MixConfig, cqts: list[torch.Tensor],
            adaptive: bool = False, steps: int = 300,
            lr: float = 1e-3, penalty_weight: float = 1.0,
            log_every: int = 10, verbose: bool = True) -> float:
    """Run Adam on the energy from the current *config* in-place.

    Returns the final energy value.
    """
    optimiser = torch.optim.Adam(config.parameters(), lr=lr)

    for step in range(1, steps + 1):
        optimiser.zero_grad()
        E = compute_energy_with_penalty(config, cqts, adaptive=adaptive,
                                        lam=penalty_weight)
        E.backward()
        optimiser.step()
        config.clamp_gains()

        if verbose and step % log_every == 0:
            print(f"  step {step:4d}  E = {E.item():.4f}")

    # Final energy without penalty for ranking
    with torch.no_grad():
        E_final = compute_energy(config, cqts, adaptive=adaptive).item()
    return E_final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track_order_from_config(config: MixConfig) -> list[int]:
    """Derive the track ordering by ascending offset."""
    with torch.no_grad():
        offsets = config.offsets.detach().cpu().tolist()
    return sorted(range(config.n_tracks), key=lambda i: offsets[i])


def _transition_points(config: MixConfig) -> list[float]:
    """Estimate transition midpoints (seconds) between consecutive tracks."""
    order = _track_order_from_config(config)
    pts: list[float] = []
    with torch.no_grad():
        off = config.offsets.detach().cpu()
        for a, b in zip(order, order[1:]):
            _, wv_a = config.warp_mapping(a)
            end_a = (off[a] + wv_a[-1].cpu()) * config.frame_dur
            start_b = off[b].item() * 1.0  # already in seconds
            mid = (end_a.item() + start_b) / 2.0
            pts.append(mid)
    return pts


def _configs_equivalent(a: MixConfig, b: MixConfig,
                        energy_tol: float = 0.5,
                        offset_tol_s: float = 1.0) -> bool:
    """Rough deduplication: same track ordering and similar energy/offsets."""
    if _track_order_from_config(a) != _track_order_from_config(b):
        return False
    with torch.no_grad():
        diff = (a.offsets - b.offsets).abs().max().item()
    return diff < offset_tol_s


# ---------------------------------------------------------------------------
# Basin discovery
# ---------------------------------------------------------------------------

def discover_basins(cqts: list[torch.Tensor], cqt_lengths: list[int],
                    audio_paths: list[str],
                    hop_length: int = 512, sr: int = 44100,
                    adaptive: bool = False, restarts: int = 20,
                    steps: int = 300, lr: float = 1e-3,
                    verbose: bool = True) -> list[Basin]:
    """Run *restarts* descents from random inits.  Return deduplicated basins
    ranked by energy (lowest first).
    """
    from .config import init_from_bpm

    n = len(audio_paths)

    # Seed config gives us the BPM-derived ordering
    seed_cfg, seed_order = init_from_bpm(audio_paths, cqt_lengths,
                                         hop_length=hop_length, sr=sr)

    # Generate shuffled orderings
    orderings: list[list[int]] = [seed_order]
    all_perms = list(itertools.permutations(range(n)))
    random.shuffle(all_perms)
    for perm in all_perms:
        if list(perm) not in orderings:
            orderings.append(list(perm))
        if len(orderings) >= restarts:
            break
    # Pad with random shuffles if not enough permutations
    while len(orderings) < restarts:
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    basins: list[Basin] = []

    for r, order in enumerate(orderings[:restarts]):
        if verbose:
            print(f"\n=== restart {r + 1}/{restarts}  order={order} ===")

        cfg = init_random(n, cqt_lengths, order,
                          hop_length=hop_length, sr=sr)
        E = descend(cfg, cqts, adaptive=adaptive, steps=steps,
                    lr=lr, verbose=verbose)

        # Dedup
        dup = False
        for existing in basins:
            if _configs_equivalent(cfg, existing.config):
                if E < existing.energy:
                    existing.config = copy.deepcopy(cfg)
                    existing.energy = E
                    existing.track_order = _track_order_from_config(cfg)
                    existing.transition_points = _transition_points(cfg)
                dup = True
                break

        if not dup:
            basins.append(Basin(
                config=copy.deepcopy(cfg),
                energy=E,
                track_order=_track_order_from_config(cfg),
                transition_points=_transition_points(cfg),
            ))

    basins.sort(key=lambda b: b.energy)

    if verbose:
        print(f"\nDiscovered {len(basins)} distinct basins:")
        for i, b in enumerate(basins):
            print(f"  #{i + 1}  {b}")

    return basins
