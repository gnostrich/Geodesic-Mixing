#!/usr/bin/env python3
"""CLI entry point for geodesic mixing.

Usage
-----
    python -m src.mix path/to/tracks/ --output mix.wav [--adaptive] \
        [--restarts 20] [--steps 300] [--basin N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch

from .cqt import compute_cqt, SR, HOP_LENGTH
from .config import init_from_bpm
from .energy import compute_energy
from .optimise import descend, discover_basins
from .render import render_mix

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}


def _find_tracks(folder: str) -> list[str]:
    paths: list[str] = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
            paths.append(os.path.join(folder, f))
    if not paths:
        sys.exit(f"No audio files found in {folder}")
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Geodesic Mixing — multiscale spectral-flux DJ tool")
    ap.add_argument("tracks", help="Folder of audio files")
    ap.add_argument("--output", "-o", default="mix.wav",
                    help="Output WAV path (default: mix.wav)")
    ap.add_argument("--adaptive", action="store_true",
                    help="Enable directed inter-scale coupling weights")
    ap.add_argument("--restarts", type=int, default=20,
                    help="Number of random restarts for basin discovery")
    ap.add_argument("--steps", type=int, default=300,
                    help="Gradient descent steps per restart")
    ap.add_argument("--basin", type=int, default=1,
                    help="Which basin to render (1 = lowest energy)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam learning rate")
    ap.add_argument("--sr", type=int, default=SR,
                    help="Sample rate")
    ap.add_argument("--hop", type=int, default=HOP_LENGTH,
                    help="CQT hop length")
    args = ap.parse_args(argv)

    tracks = _find_tracks(args.tracks)
    n = len(tracks)
    print(f"Found {n} tracks:")
    for t in tracks:
        print(f"  {os.path.basename(t)}")

    # ------------------------------------------------------------------
    # Step 1 — CQT precompute
    # ------------------------------------------------------------------
    print("\n[1/6] Computing CQTs …")
    t0 = time.time()
    cqts = []
    cqt_lengths = []
    for path in tracks:
        C = compute_cqt(path, sr=args.sr, hop_length=args.hop)
        cqts.append(C)
        cqt_lengths.append(C.shape[1])
        print(f"  {os.path.basename(path)}: {C.shape[1]} frames "
              f"({C.shape[1] * args.hop / args.sr:.1f}s)")
    print(f"  done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Steps 2–5 — Basin discovery (init → forward → backward → repeat)
    # ------------------------------------------------------------------
    mode = "adaptive" if args.adaptive else "flat"
    print(f"\n[2-5] Basin discovery ({mode}, {args.restarts} restarts, "
          f"{args.steps} steps) …")
    t0 = time.time()

    basins = discover_basins(
        cqts, cqt_lengths, tracks,
        hop_length=args.hop, sr=args.sr,
        adaptive=args.adaptive, restarts=args.restarts,
        steps=args.steps, lr=args.lr,
    )
    print(f"  discovery took {time.time() - t0:.1f}s")

    if not basins:
        sys.exit("No basins discovered.")

    # ------------------------------------------------------------------
    # Print basins
    # ------------------------------------------------------------------
    print(f"\nDiscovered {len(basins)} basins (ranked by energy):")
    for i, b in enumerate(basins, 1):
        marker = " ←" if i == args.basin else ""
        order_names = [os.path.basename(tracks[j]) for j in b.track_order]
        order_str = " → ".join(order_names)
        trans_str = ", ".join(f"{t:.1f}s" for t in b.transition_points)
        print(f"  #{i}  E={b.energy:.2f}  [{order_str}]  "
              f"transitions=[{trans_str}]{marker}")

    # ------------------------------------------------------------------
    # Step 6 — Render
    # ------------------------------------------------------------------
    idx = args.basin - 1
    if idx < 0 or idx >= len(basins):
        sys.exit(f"--basin {args.basin} out of range (1–{len(basins)})")

    chosen = basins[idx]
    print(f"\n[6] Rendering basin #{args.basin} (E={chosen.energy:.2f}) …")
    render_mix(chosen.config, tracks, args.output, sr=args.sr)
    print("Done.")


if __name__ == "__main__":
    main()
