/* ═══════════════════════════════════════════════════════════
   Mix Engine: energy computation, coupling weights, optimizer
   ═══════════════════════════════════════════════════════════ */

'use strict';

const Engine = (() => {

const TAU_MAX = 200;   // max lag for coupling (frames)
const N_GAIN_KNOTS = 20;

// ─── Track object ─────────────────────────────────────────

class Track {
    constructor(name, audioBuffer, sr) {
        this.name = name;
        this.audioBuffer = audioBuffer;
        this.sr = sr || audioBuffer.sampleRate;
        this.samples = audioBuffer.getChannelData(0);
        this.duration = audioBuffer.duration;
        this.stft = null;      // raw STFT
        this.spec = null;      // log-frequency spectrogram
        this.offset = 0;       // seconds
        this.gainKnots = new Float32Array(N_GAIN_KNOTS).fill(1.0);
        this.color = '#6c5ce7';
    }

    computeSpectrogram() {
        this.stft = DSP.computeSTFT(this.samples, DSP.FFT_SIZE, DSP.HOP_SIZE);
        this.spec = DSP.stftToLogFreq(this.stft, DSP.FFT_SIZE, this.sr);
        return this;
    }

    get offsetFrames() {
        return this.offset * this.sr / DSP.HOP_SIZE;
    }
}

// ─── Mix computation ──────────────────────────────────────

function computeMix(tracks) {
    if (tracks.length === 0) return null;

    const nBins = DSP.N_CQ_BINS;
    const hopDur = DSP.HOP_SIZE / tracks[0].sr;

    // Determine output length
    let maxEnd = 0;
    for (const t of tracks) {
        if (!t.spec) continue;
        const end = t.offsetFrames + t.spec.nFrames;
        if (end > maxEnd) maxEnd = end;
    }
    const outFrames = Math.ceil(maxEnd) + 1;
    if (outFrames <= 0) return null;

    // Accumulate complex mix
    const mixRe = new Float32Array(nBins * outFrames);
    const mixIm = new Float32Array(nBins * outFrames);

    for (const track of tracks) {
        if (!track.spec) continue;
        const spec = track.spec;
        const off = track.offsetFrames;
        const gain = DSP.interpolateGain(track.gainKnots, spec.nFrames);

        for (let m = 0; m < outFrames; m++) {
            const src = m - off;
            if (src < 0 || src >= spec.nFrames - 1) continue;

            const lo = Math.floor(src);
            const hi = lo + 1;
            const frac = src - lo;
            const g = gain[Math.min(lo, gain.length - 1)];
            if (g < 1e-6) continue;

            for (let b = 0; b < nBins; b++) {
                const outIdx = b * outFrames + m;
                const sLo = b * spec.nFrames + lo;
                const sHi = b * spec.nFrames + hi;

                mixRe[outIdx] += (spec.re[sLo] * (1-frac) + spec.re[sHi] * frac) * g;
                mixIm[outIdx] += (spec.im[sLo] * (1-frac) + spec.im[sHi] * frac) * g;
            }
        }
    }

    // Magnitude (sum complex THEN magnitude — preserves phase interactions)
    const mag = new Float32Array(nBins * outFrames);
    for (let i = 0; i < mag.length; i++)
        mag[i] = Math.sqrt(mixRe[i] * mixRe[i] + mixIm[i] * mixIm[i]);

    return { re: mixRe, im: mixIm, mag, nBins, nFrames: outFrames };
}

// ─── Spectral flux ────────────────────────────────────────

function computeFlux(mag, nBins, nFrames) {
    const nFlux = nFrames - 1;
    const flux = new Float32Array(nBins * nFlux);
    for (let b = 0; b < nBins; b++) {
        for (let m = 0; m < nFlux; m++) {
            flux[b * nFlux + m] = mag[b * nFrames + m + 1] - mag[b * nFrames + m];
        }
    }
    return { data: flux, nBins, nFrames: nFlux };
}

// ─── Flat energy ──────────────────────────────────────────

function flatEnergy(flux) {
    let E = 0;
    for (let i = 0; i < flux.data.length; i++)
        E += flux.data[i] * flux.data[i];
    return E;
}

// ─── Adaptive coupling weights ────────────────────────────

function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

function computeCouplingWeights(flux) {
    const B = flux.nBins;
    const T = flux.nFrames;
    const tauMax = Math.min(TAU_MAX, T - 1);
    const nfft = nextPow2(T + tauMax);

    // Pre-compute FFTs of each band's flux
    const DeltaRe = new Array(B);
    const DeltaIm = new Array(B);
    const norms = new Float32Array(B);

    for (let b = 0; b < B; b++) {
        const re = new Float32Array(nfft);
        const im = new Float32Array(nfft);
        let norm2 = 0;
        for (let m = 0; m < T; m++) {
            const v = flux.data[b * T + m];
            re[m] = v;
            norm2 += v * v;
        }
        DSP.fft(re, im, false);
        DeltaRe[b] = re;
        DeltaIm[b] = im;
        norms[b] = Math.sqrt(norm2);
    }

    const mu = new Float32Array(B);

    for (let k = 0; k < B; k++) {
        let incomingSum = 0;
        for (let j = 0; j < B; j++) {
            if (j === k) continue;

            // C_{j→k}(τ) = IFFT(conj(Delta_j) * Delta_k)[τ]
            const ccRe = new Float32Array(nfft);
            const ccIm = new Float32Array(nfft);
            for (let i = 0; i < nfft; i++) {
                // conj(Delta_j) * Delta_k
                const aRe = DeltaRe[j][i], aIm = -DeltaIm[j][i]; // conjugate
                const bRe = DeltaRe[k][i], bIm = DeltaIm[k][i];
                ccRe[i] = aRe * bRe - aIm * bIm;
                ccIm[i] = aRe * bIm + aIm * bRe;
            }
            DSP.fft(ccRe, ccIm, true); // inverse FFT

            // Normalize
            const normJK = Math.max(1e-8, norms[j] * norms[k]);

            // Find peak positive lag
            let bestAbs = 0, bestVal = 0;
            for (let tau = 1; tau <= tauMax; tau++) {
                const val = ccRe[tau] / normJK;
                const absVal = Math.abs(val);
                if (absVal > bestAbs) {
                    bestAbs = absVal;
                    bestVal = val;
                }
            }
            incomingSum += bestVal;
        }
        mu[k] = incomingSum / (B - 1);
    }

    // ω_k = exp(μ_k)
    const omega = new Float32Array(B);
    for (let k = 0; k < B; k++)
        omega[k] = Math.exp(mu[k]);

    return omega;
}

// ─── Adaptive energy ──────────────────────────────────────

function adaptiveEnergy(flux, omega) {
    const B = flux.nBins, T = flux.nFrames;
    let E = 0;
    for (let b = 0; b < B; b++) {
        const w = omega[b];
        for (let m = 0; m < T; m++) {
            const d = flux.data[b * T + m];
            E += w * d * d;
        }
    }
    return E;
}

// ─── Full energy computation ──────────────────────────────

function computeEnergy(tracks, adaptive) {
    const mix = computeMix(tracks);
    if (!mix || mix.nFrames < 2) return { energy: 0, mix, flux: null, omega: null };

    const flux = computeFlux(mix.mag, mix.nBins, mix.nFrames);
    let energy, omega = null;

    if (adaptive) {
        omega = computeCouplingWeights(flux);
        energy = adaptiveEnergy(flux, omega);
    } else {
        energy = flatEnergy(flux);
    }

    return { energy, mix, flux, omega };
}

// ─── Optimizer ────────────────────────────────────────────

function optimizeSweep(tracks, adaptive, onProgress) {
    // For each pair of adjacent tracks (by current offset),
    // sweep the second track's offset to minimise transition energy.
    if (tracks.length < 2) return;

    const sorted = tracks.slice().sort((a, b) => a.offset - b.offset);
    const hopDur = DSP.HOP_SIZE / tracks[0].sr;
    let bestEnergy = Infinity;
    let step = 0;
    const totalSteps = (sorted.length - 1) * 20;

    for (let i = 0; i < sorted.length - 1; i++) {
        const prev = sorted[i];
        const next = sorted[i + 1];
        const prevEnd = prev.offset + prev.duration;

        // Sweep overlap from 5s to 55s in 20 steps
        const bestOffset = next.offset;
        let localBest = Infinity;
        let localBestOff = next.offset;

        for (let s = 0; s < 20; s++) {
            const overlap = 5 + s * 2.5;  // 5s to 52.5s
            const candidateOff = prevEnd - overlap;
            if (candidateOff < 0) continue;

            next.offset = candidateOff;
            const { energy } = computeEnergy(tracks, adaptive);
            if (energy < localBest) {
                localBest = energy;
                localBestOff = candidateOff;
            }

            step++;
            if (onProgress) onProgress(step / totalSteps, energy);
        }

        next.offset = localBestOff;
    }

    return computeEnergy(tracks, adaptive);
}

function optimizeGains(tracks, adaptive, onProgress) {
    // For each track, try fading gain at transitions
    if (tracks.length < 2) return computeEnergy(tracks, adaptive);

    const sorted = tracks.slice().sort((a, b) => a.offset - b.offset);
    let step = 0;
    const totalSteps = sorted.length * N_GAIN_KNOTS * 5;

    for (const track of sorted) {
        for (let ki = 0; ki < N_GAIN_KNOTS; ki++) {
            let bestVal = track.gainKnots[ki];
            let bestE = Infinity;

            for (let v = 0; v <= 4; v++) {
                const candidate = v * 0.25;
                track.gainKnots[ki] = candidate;
                const { energy } = computeEnergy(tracks, adaptive);
                if (energy < bestE) {
                    bestE = energy;
                    bestVal = candidate;
                }
                step++;
                if (onProgress) onProgress(step / totalSteps, energy);
            }
            track.gainKnots[ki] = bestVal;
        }
    }

    return computeEnergy(tracks, adaptive);
}

// ─── Public API ───────────────────────────────────────────

return {
    Track, N_GAIN_KNOTS, TAU_MAX,
    computeMix, computeFlux, computeEnergy,
    flatEnergy, computeCouplingWeights, adaptiveEnergy,
    optimizeSweep, optimizeGains,
};

})();
