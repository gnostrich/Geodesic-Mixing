/* ═══════════════════════════════════════════════════════════
   DSP: FFT, STFT, log-frequency mapping
   ═══════════════════════════════════════════════════════════ */

'use strict';

const DSP = (() => {

// ─── Constants ────────────────────────────────────────────
const FFT_SIZE  = 2048;
const HOP_SIZE  = 512;
const N_CQ_BINS = 60;       // 10 octaves × 6 bins/octave
const BINS_PER_OCTAVE = 6;
const FMIN = 27.5;           // A0

// ─── Radix-2 Cooley-Tukey FFT (in-place) ─────────────────

function fft(re, im, invert) {
    const N = re.length;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    // Butterfly stages
    for (let len = 2; len <= N; len <<= 1) {
        const half = len >> 1;
        const ang = (invert ? 1 : -1) * 2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let cRe = 1, cIm = 0;
            for (let j = 0; j < half; j++) {
                const a = i + j, b = a + half;
                const tRe = re[b] * cRe - im[b] * cIm;
                const tIm = re[b] * cIm + im[b] * cRe;
                re[b] = re[a] - tRe;  im[b] = im[a] - tIm;
                re[a] += tRe;         im[a] += tIm;
                const nRe = cRe * wRe - cIm * wIm;
                cIm = cRe * wIm + cIm * wRe;
                cRe = nRe;
            }
        }
    }
    if (invert) {
        for (let i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
    }
}

// ─── Hann window ──────────────────────────────────────────

function hannWindow(N) {
    const w = new Float32Array(N);
    for (let n = 0; n < N; n++)
        w[n] = 0.5 * (1 - Math.cos(2 * Math.PI * n / (N - 1)));
    return w;
}

// Cached windows
const _windows = {};
function getWindow(N) {
    if (!_windows[N]) _windows[N] = hannWindow(N);
    return _windows[N];
}

// ─── STFT (complex output) ────────────────────────────────
//
// Returns { re, im, nBins, nFrames } where re/im are
// Float32Arrays of shape [nBins × nFrames] in row-major order
// (frequency-major: re[b * nFrames + m]).

function computeSTFT(samples, fftSize, hopSize) {
    fftSize = fftSize || FFT_SIZE;
    hopSize = hopSize || HOP_SIZE;

    const win = getWindow(fftSize);
    const nFrames = Math.max(1, Math.floor((samples.length - fftSize) / hopSize) + 1);
    const nBins = (fftSize >> 1) + 1;

    const re = new Float32Array(nBins * nFrames);
    const im = new Float32Array(nBins * nFrames);

    const fRe = new Float32Array(fftSize);
    const fIm = new Float32Array(fftSize);

    for (let m = 0; m < nFrames; m++) {
        const off = m * hopSize;
        // Windowed frame
        for (let n = 0; n < fftSize; n++) {
            fRe[n] = (off + n < samples.length) ? samples[off + n] * win[n] : 0;
            fIm[n] = 0;
        }
        fft(fRe, fIm, false);
        // Store positive frequencies
        for (let k = 0; k < nBins; k++) {
            re[k * nFrames + m] = fRe[k];
            im[k * nFrames + m] = fIm[k];
        }
    }

    return { re, im, nBins, nFrames };
}

// ─── Log-frequency mapping (STFT → pseudo-CQT) ───────────
//
// Groups linear FFT bins into logarithmically-spaced bands,
// averaging the complex values.  Approximate CQT, fast to compute.

let _logFreqMap = null;
let _logFreqMapKey = '';

function buildLogFreqMap(fftSize, sr) {
    const key = `${fftSize}|${sr}`;
    if (_logFreqMapKey === key && _logFreqMap) return _logFreqMap;

    const nFftBins = (fftSize >> 1) + 1;
    const mapping = [];

    for (let b = 0; b < N_CQ_BINS; b++) {
        const fLo = FMIN * Math.pow(2, b / BINS_PER_OCTAVE);
        const fHi = FMIN * Math.pow(2, (b + 1) / BINS_PER_OCTAVE);
        const indices = [];
        for (let k = 0; k < nFftBins; k++) {
            const f = k * sr / fftSize;
            if (f >= fLo && f < fHi) indices.push(k);
        }
        // At very low frequencies, ensure at least one bin
        if (indices.length === 0) {
            const nearest = Math.round(fLo * fftSize / sr);
            if (nearest >= 0 && nearest < nFftBins) indices.push(nearest);
        }
        mapping.push(indices);
    }

    _logFreqMap = mapping;
    _logFreqMapKey = key;
    return mapping;
}

function stftToLogFreq(stft, fftSize, sr) {
    const mapping = buildLogFreqMap(fftSize, sr);
    const nFrames = stft.nFrames;
    const re = new Float32Array(N_CQ_BINS * nFrames);
    const im = new Float32Array(N_CQ_BINS * nFrames);

    for (let b = 0; b < N_CQ_BINS; b++) {
        const ids = mapping[b];
        if (ids.length === 0) continue;
        const invN = 1 / ids.length;
        for (let m = 0; m < nFrames; m++) {
            let sRe = 0, sIm = 0;
            for (let i = 0; i < ids.length; i++) {
                const k = ids[i];
                sRe += stft.re[k * nFrames + m];
                sIm += stft.im[k * nFrames + m];
            }
            re[b * nFrames + m] = sRe * invN;
            im[b * nFrames + m] = sIm * invN;
        }
    }

    return { re, im, nBins: N_CQ_BINS, nFrames };
}

// ─── Magnitude from complex spectrogram ───────────────────

function magnitude(spec) {
    const N = spec.re.length;
    const mag = new Float32Array(N);
    for (let i = 0; i < N; i++)
        mag[i] = Math.sqrt(spec.re[i] * spec.re[i] + spec.im[i] * spec.im[i]);
    return mag;
}

// ─── Cubic Hermite interpolation (C1) ─────────────────────

function cubicHermite(x, knots, values) {
    // x: query position in [0, 1]
    // knots: uniformly spaced, values: at each knot
    const K = values.length;
    const span = 1 / (K - 1);
    let idx = Math.floor(x / span);
    idx = Math.max(0, Math.min(idx, K - 2));

    const t = (x - idx * span) / span;
    const t2 = t * t, t3 = t2 * t;

    // Catmull-Rom tangents
    const y0 = values[idx], y1 = values[idx + 1];
    const m0 = idx > 0 ? (values[idx + 1] - values[idx - 1]) / 2 : (values[1] - values[0]);
    const m1 = idx < K - 2 ? (values[idx + 2] - values[idx]) / 2 : (values[K - 1] - values[K - 2]);

    const h00 = 2*t3 - 3*t2 + 1;
    const h10 = t3 - 2*t2 + t;
    const h01 = -2*t3 + 3*t2;
    const h11 = t3 - t2;

    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
}

// Interpolate a gain envelope with N_KNOTS control points
// over nFrames output frames.  Returns Float32Array[nFrames].
function interpolateGain(knots, nFrames) {
    const out = new Float32Array(nFrames);
    for (let m = 0; m < nFrames; m++) {
        const x = nFrames > 1 ? m / (nFrames - 1) : 0;
        out[m] = Math.max(0, Math.min(1, cubicHermite(x, null, knots)));
    }
    return out;
}

// ─── Public API ───────────────────────────────────────────

return {
    FFT_SIZE, HOP_SIZE, N_CQ_BINS, BINS_PER_OCTAVE, FMIN,
    fft, computeSTFT, stftToLogFreq, magnitude,
    cubicHermite, interpolateGain,
    buildLogFreqMap,
};

})();
