# Geodesic Mixing

Automatic DJ mixing by minimising **multiscale spectral flux** — the integrated squared rate of change of the combined signal's spectral-magnitude profile across all timescales simultaneously.

Based on *Geodesic Mixing: Multiscale Spectral Flux as the Energy Functional for Audio Transition Landscapes* (v8, April 2026).

## Quick start

```bash
pip install torch librosa soundfile numpy
python -m src.mix path/to/tracks/ --output mix.wav
```

## Usage

```
python -m src.mix path/to/tracks/ --output mix.wav [--adaptive] [--restarts 20] [--steps 300] [--basin N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `mix.wav` | Output WAV path |
| `--adaptive` | off | Enable directed inter-scale coupling weights |
| `--restarts` | 20 | Random restarts for basin discovery |
| `--steps` | 300 | Gradient descent steps per restart |
| `--basin N` | 1 | Render the Nth basin (1 = lowest energy) |
| `--lr` | 0.001 | Adam learning rate |
| `--sr` | 44100 | Sample rate |
| `--hop` | 512 | CQT hop length |

## Two modes

**Flat measure** (default): every octave weighted equally — `E = Σ δ²`.

**Adaptive measure** (`--adaptive`): weights each frequency band by how strongly other bands' flux predicts its flux. Bands participating in hierarchical structure (rhythm → phrase → energy arc) cost more to disrupt. `ω_k = exp(mean incoming coupling)`, bounded in [1/e, e].

## Pipeline

1. **CQT precompute** — constant-Q transform (~20 Hz – 20 kHz), cached to disk.
2. **Configuration vector** — per-track offset, monotone time warp (~10 knots), C1 gain envelope (~20 knots).
3. **Forward pass** — warp → gain → sum complex CQTs → magnitude → flux → energy.
4. **Backward pass** — `E.backward()`, Adam step, clamp gains to [0, 1].
5. **Basin discovery** — repeat from R random initialisations, deduplicate, rank by E.
6. **Audio render** — time-domain warp + gain + sum → WAV.

## Constraints

- All tracks must appear
- Each track active ≥ 30 s
- Maximum pairwise overlap ≤ 60 s
- Gain envelopes are C1 (cubic Hermite interpolation)

## Tests

```bash
pytest tests/ -v
```

## Project structure

```
src/
  cqt.py          # CQT computation + caching
  config.py       # Configuration vector, init, clamping
  energy.py       # Forward pass, E computation (flat + adaptive)
  optimise.py     # Descent loop, basin discovery
  render.py       # Audio rendering from converged config
  mix.py          # CLI entry point
tests/
  test_energy.py    # Gradient check, stationarity, magnitude ordering
  test_adaptive.py  # Symmetry violation, white noise baseline, weight bounds
paper/
  geodesic-mixing-v8.pdf
```
