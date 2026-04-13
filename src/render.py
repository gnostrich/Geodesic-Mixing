"""Audio rendering from a converged configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import MixConfig, _linear_interp, _cubic_hermite_interp, N_GAIN_KNOTS
from .cqt import load_audio


def render_mix(config: MixConfig, audio_paths: list[str],
               output_path: str, sr: int = 44100) -> None:
    """Render the final mix to a WAV file.

    For each output sample, sum the time-warped, gain-enveloped waveforms of
    all active tracks.  Writes 32-bit float WAV via *soundfile*.
    """
    hop = config.hop_length
    device = config.offsets.device

    # Load waveforms (always on CPU for rendering)
    waveforms: list[torch.Tensor] = []
    for p in audio_paths:
        wav, _ = load_audio(p, sr=sr)
        waveforms.append(wav)

    T_out_frames = config.output_length()
    T_out_samples = T_out_frames * hop + hop  # slight overallocation

    output = torch.zeros(T_out_samples)

    with torch.no_grad():
        off_sec = config.offsets.detach().cpu()

        for i in range(config.n_tracks):
            T_i_samples = len(waveforms[i])
            T_i_frames = config.cqt_lengths[i]

            offset_samples = int(round(off_sec[i].item() * sr))

            # Determine the output sample range this track can span.
            # The warped duration may differ from the original.
            _, wv = config.warp_mapping(i)
            wv = wv.detach().cpu()
            warped_dur_samples = int(round(wv[-1].item() * hop)) + hop
            start = max(0, offset_samples)
            end = min(T_out_samples, offset_samples + warped_dur_samples)
            if start >= end:
                continue

            out_idx = torch.arange(start, end, dtype=torch.float32)

            # Nominal local frame for each output sample
            nominal_frames = (out_idx - offset_samples) / hop

            # Warp → warped local frame
            knot_pos, warp_vals = config.warp_mapping(i)
            knot_pos = knot_pos.detach().cpu()
            warp_vals = warp_vals.detach().cpu()
            warped_frames = _linear_interp(nominal_frames, knot_pos, warp_vals)

            # Warped frame → sample index into original waveform
            warped_samples = warped_frames * hop

            # Validity mask
            valid = (warped_samples >= 0) & (warped_samples < T_i_samples - 1)
            ws_c = warped_samples.clamp(0, T_i_samples - 1 - 1e-3)

            # Linear interpolation of waveform
            lo = ws_c.long()
            hi = (lo + 1).clamp(max=T_i_samples - 1)
            frac = ws_c - lo.float()
            wav_interp = waveforms[i][lo] * (1 - frac) + waveforms[i][hi] * frac

            # Gain envelope (evaluated at warped *frame* positions)
            gain_vals = torch.clamp(config.gain_knots[i].detach().cpu(), 0.0, 1.0)
            gain_knot_x = torch.linspace(0, T_i_frames - 1, N_GAIN_KNOTS)
            gain = _cubic_hermite_interp(warped_frames, gain_knot_x, gain_vals)
            gain = gain.clamp(0.0, 1.0) * valid.float()

            wav_interp = wav_interp * gain

            # Accumulate
            length = min(end - start, T_out_samples - start)
            output[start : start + length] += wav_interp[:length]

    # Trim trailing silence
    nz = torch.nonzero(output.abs() > 1e-7)
    if len(nz) > 0:
        output = output[: nz[-1].item() + 1]

    # Normalise to prevent clipping
    peak = output.abs().max()
    if peak > 0:
        output = output / peak * 0.95

    sf.write(output_path, output.numpy(), sr, subtype="FLOAT")
    print(f"Rendered mix → {output_path}  "
          f"({len(output) / sr:.1f}s, {sr} Hz)")
