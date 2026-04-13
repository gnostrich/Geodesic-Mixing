/* ═══════════════════════════════════════════════════════════
   UI: Timeline, Gain Editor, Visualisations, Playback
   ═══════════════════════════════════════════════════════════ */

'use strict';

const UI = (() => {

const TRACK_COLORS = [
    '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
    '#3498db', '#9b59b6', '#e91e63', '#00bcd4',
];

// ─── Color maps ───────────────────────────────────────────

function magmaColor(t) {
    // Simplified magma-like colormap: t ∈ [0, 1]
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(255 * Math.min(1, t * 3));
    const g = Math.floor(255 * Math.max(0, Math.min(1, t * 3 - 1)));
    const b = Math.floor(255 * Math.max(0, Math.min(1, (t - 0.2) * 2)));
    return `rgb(${r},${g},${b})`;
}

function viridisColor(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(255 * (0.267 + t * (0.993 - 0.267)));
    const g = Math.floor(255 * (0.004 + t * (0.906 - 0.004)));
    const b = Math.floor(255 * (0.329 + t * (-0.143 - 0.329 + 1) * t));
    return `rgb(${Math.min(255,r)},${Math.min(255,g)},${Math.min(255,b)})`;
}

function fluxColor(t) {
    // Diverging: blue (negative) → black → red (positive)
    if (t >= 0) {
        const v = Math.min(1, t);
        return `rgb(${Math.floor(255*v)},${Math.floor(40*v)},${Math.floor(20*v)})`;
    } else {
        const v = Math.min(1, -t);
        return `rgb(${Math.floor(20*v)},${Math.floor(60*v)},${Math.floor(255*v)})`;
    }
}

// ─── Timeline ─────────────────────────────────────────────

class Timeline {
    constructor(canvas, app) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.app = app;
        this.pixelsPerSecond = 30;
        this.trackHeight = 50;
        this.headerHeight = 25;
        this.dragging = null; // { trackIndex, startX, startOffset }

        this._bindEvents();
    }

    _bindEvents() {
        const c = this.canvas;
        c.addEventListener('mousedown', e => this._onMouseDown(e));
        c.addEventListener('mousemove', e => this._onMouseMove(e));
        c.addEventListener('mouseup', e => this._onMouseUp(e));
        c.addEventListener('mouseleave', e => this._onMouseUp(e));
    }

    _trackAtY(y) {
        const idx = Math.floor((y - this.headerHeight) / this.trackHeight);
        if (idx >= 0 && idx < this.app.tracks.length) return idx;
        return -1;
    }

    _onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const idx = this._trackAtY(y);
        if (idx < 0) return;
        this.dragging = {
            trackIndex: idx,
            startX: x,
            startOffset: this.app.tracks[idx].offset,
        };
        this.canvas.style.cursor = 'grabbing';
    }

    _onMouseMove(e) {
        if (!this.dragging) {
            const rect = this.canvas.getBoundingClientRect();
            const y = e.clientY - rect.top;
            this.canvas.style.cursor = this._trackAtY(y) >= 0 ? 'grab' : 'default';
            return;
        }
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const dx = x - this.dragging.startX;
        const dt = dx / this.pixelsPerSecond;
        const track = this.app.tracks[this.dragging.trackIndex];
        track.offset = Math.max(0, this.dragging.startOffset + dt);
        this.render();
    }

    _onMouseUp(e) {
        if (this.dragging) {
            this.dragging = null;
            this.canvas.style.cursor = 'grab';
            this.app.onConfigChanged();
        }
    }

    render() {
        const tracks = this.app.tracks;
        if (tracks.length === 0) return;

        const pps = this.pixelsPerSecond;
        const th = this.trackHeight;

        // Canvas size
        let maxEnd = 0;
        for (const t of tracks) {
            const end = t.offset + t.duration;
            if (end > maxEnd) maxEnd = end;
        }
        const totalWidth = Math.max(600, (maxEnd + 5) * pps);
        const totalHeight = this.headerHeight + tracks.length * th + 10;

        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = totalWidth * dpr;
        this.canvas.height = totalHeight * dpr;
        this.canvas.style.width = totalWidth + 'px';
        this.canvas.style.height = totalHeight + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const ctx = this.ctx;
        ctx.clearRect(0, 0, totalWidth, totalHeight);

        // Time ruler
        ctx.fillStyle = '#333355';
        ctx.fillRect(0, 0, totalWidth, this.headerHeight);
        ctx.fillStyle = '#8888aa';
        ctx.font = '10px monospace';
        const step = pps >= 20 ? 5 : 10;
        for (let s = 0; s <= maxEnd + 5; s += step) {
            const x = s * pps;
            ctx.fillRect(x, 15, 1, 10);
            ctx.fillText(`${s}s`, x + 3, 12);
        }

        // Track rows
        for (let i = 0; i < tracks.length; i++) {
            const track = tracks[i];
            const y = this.headerHeight + i * th;
            const x = track.offset * pps;
            const w = track.duration * pps;

            // Background
            ctx.fillStyle = '#0a0a18';
            ctx.fillRect(0, y, totalWidth, th);

            // Track bar
            const color = track.color;
            ctx.globalAlpha = 0.2;
            ctx.fillStyle = color;
            ctx.fillRect(x, y + 2, w, th - 4);
            ctx.globalAlpha = 1;

            // Mini spectrogram
            if (track.spec) {
                this._drawMiniSpec(ctx, track, x, y + 2, w, th - 4);
            }

            // Border
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y + 2, w, th - 4);

            // Label
            ctx.fillStyle = '#fff';
            ctx.font = '11px sans-serif';
            ctx.fillText(track.name, x + 5, y + 15);

            // Duration label
            ctx.fillStyle = '#aaa';
            ctx.font = '9px monospace';
            ctx.fillText(`${track.duration.toFixed(1)}s`, x + 5, y + th - 8);
        }
    }

    _drawMiniSpec(ctx, track, x, y, w, h) {
        const spec = track.spec;
        const nBins = spec.nBins;
        const nFrames = spec.nFrames;
        if (nFrames === 0 || nBins === 0) return;

        // Pre-compute magnitude & max
        const mag = DSP.magnitude(spec);
        let maxMag = 0;
        for (let i = 0; i < mag.length; i++)
            if (mag[i] > maxMag) maxMag = mag[i];
        if (maxMag < 1e-10) return;

        // Downsample to canvas pixels
        const pw = Math.max(1, Math.floor(w));
        const ph = Math.max(1, Math.floor(h));
        const imgData = ctx.createImageData(pw, ph);

        for (let px = 0; px < pw; px++) {
            const frame = Math.floor(px / pw * nFrames);
            for (let py = 0; py < ph; py++) {
                const bin = Math.floor((1 - py / ph) * nBins);
                const val = mag[Math.min(bin, nBins-1) * nFrames + Math.min(frame, nFrames-1)] / maxMag;
                const idx = (py * pw + px) * 4;
                // Magma-like colormap
                const v = Math.pow(val, 0.4); // gamma
                imgData.data[idx]     = Math.floor(255 * Math.min(1, v * 2.5));
                imgData.data[idx + 1] = Math.floor(255 * Math.max(0, v * 2 - 0.5));
                imgData.data[idx + 2] = Math.floor(255 * Math.max(0, v - 0.3) * 2);
                imgData.data[idx + 3] = 180;
            }
        }

        ctx.putImageData(imgData, x, y);
    }

    zoom(factor) {
        this.pixelsPerSecond = Math.max(5, Math.min(200, this.pixelsPerSecond * factor));
        this.render();
    }
}

// ─── Gain Editor ──────────────────────────────────────────

class GainEditor {
    constructor(container, track, index, app) {
        this.track = track;
        this.app = app;
        this.index = index;

        this.row = document.createElement('div');
        this.row.className = 'gain-editor-row';

        const label = document.createElement('div');
        label.className = 'gain-editor-label';
        label.style.color = track.color;
        label.textContent = `Track ${index + 1}`;

        this.canvas = document.createElement('canvas');
        this.canvas.className = 'gain-editor-canvas';

        this.row.appendChild(label);
        this.row.appendChild(this.canvas);
        container.appendChild(this.row);

        this.dragging = -1;
        this._bindEvents();

        requestAnimationFrame(() => this.render());
    }

    _bindEvents() {
        this.canvas.addEventListener('mousedown', e => this._onDown(e));
        this.canvas.addEventListener('mousemove', e => this._onMove(e));
        this.canvas.addEventListener('mouseup', () => this._onUp());
        this.canvas.addEventListener('mouseleave', () => this._onUp());
    }

    _knotIndex(mx) {
        const K = this.track.gainKnots.length;
        const w = this.canvas.clientWidth;
        const knotSpacing = w / (K - 1);
        for (let i = 0; i < K; i++) {
            if (Math.abs(mx - i * knotSpacing) < 8) return i;
        }
        return -1;
    }

    _onDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        this.dragging = this._knotIndex(mx);
        if (this.dragging >= 0) this._updateKnot(e);
    }

    _onMove(e) {
        if (this.dragging < 0) return;
        this._updateKnot(e);
    }

    _updateKnot(e) {
        const rect = this.canvas.getBoundingClientRect();
        const my = e.clientY - rect.top;
        const h = this.canvas.clientHeight;
        const val = 1 - Math.max(0, Math.min(1, my / h));
        this.track.gainKnots[this.dragging] = val;
        this.render();
    }

    _onUp() {
        if (this.dragging >= 0) {
            this.dragging = -1;
            this.app.onConfigChanged();
        }
    }

    render() {
        const c = this.canvas;
        const w = c.clientWidth;
        const h = c.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        c.width = w * dpr;
        c.height = h * dpr;
        const ctx = c.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        const K = this.track.gainKnots.length;
        const knots = this.track.gainKnots;
        const color = this.track.color;

        // Interpolated curve
        ctx.beginPath();
        ctx.moveTo(0, h);
        const nPoints = w;
        for (let px = 0; px <= nPoints; px++) {
            const x = px / nPoints;
            const v = DSP.cubicHermite(x, null, knots);
            const y = h - Math.max(0, Math.min(1, v)) * h;
            if (px === 0) ctx.moveTo(px, y);
            else ctx.lineTo(px, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Fill below
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = color + '20';
        ctx.fill();

        // Control points
        for (let i = 0; i < K; i++) {
            const cx = i * w / (K - 1);
            const cy = h - knots[i] * h;
            ctx.beginPath();
            ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = i === this.dragging ? '#fff' : color;
            ctx.fill();
        }
    }
}

// ─── Visualiser ───────────────────────────────────────────

class Visualiser {
    constructor(specCanvas, fluxCanvas, weightsCanvas) {
        this.specCanvas = specCanvas;
        this.fluxCanvas = fluxCanvas;
        this.weightsCanvas = weightsCanvas;
    }

    drawSpectrogram(canvas, mag, nBins, nFrames) {
        if (!mag || nFrames < 2) return;

        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        let maxVal = 0;
        for (let i = 0; i < mag.length; i++)
            if (mag[i] > maxVal) maxVal = mag[i];
        if (maxVal < 1e-10) return;

        const imgData = ctx.createImageData(w, h);

        for (let px = 0; px < w; px++) {
            const frame = Math.floor(px / w * nFrames);
            for (let py = 0; py < h; py++) {
                const bin = Math.floor((1 - py / h) * nBins);
                const val = mag[Math.min(bin, nBins-1) * nFrames + Math.min(frame, nFrames-1)] / maxVal;
                const v = Math.pow(val, 0.35);
                const idx = (py * w + px) * 4;
                imgData.data[idx]     = Math.floor(255 * Math.min(1, v * 2.5));
                imgData.data[idx + 1] = Math.floor(255 * Math.max(0, v * 2.2 - 0.6));
                imgData.data[idx + 2] = Math.floor(200 * Math.max(0, v * 1.5 - 0.2));
                imgData.data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imgData, 0, 0);
    }

    drawFlux(canvas, flux) {
        if (!flux || flux.nFrames < 1) return;

        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const B = flux.nBins, T = flux.nFrames;
        let maxAbs = 0;
        for (let i = 0; i < flux.data.length; i++) {
            const a = Math.abs(flux.data[i]);
            if (a > maxAbs) maxAbs = a;
        }
        if (maxAbs < 1e-10) return;

        const imgData = ctx.createImageData(w, h);
        for (let px = 0; px < w; px++) {
            const frame = Math.floor(px / w * T);
            for (let py = 0; py < h; py++) {
                const bin = Math.floor((1 - py / h) * B);
                const val = flux.data[Math.min(bin, B-1) * T + Math.min(frame, T-1)] / maxAbs;
                const idx = (py * w + px) * 4;
                if (val >= 0) {
                    imgData.data[idx]     = Math.floor(255 * Math.min(1, val * 2));
                    imgData.data[idx + 1] = Math.floor(50 * val);
                    imgData.data[idx + 2] = Math.floor(30 * val);
                } else {
                    imgData.data[idx]     = Math.floor(30 * (-val));
                    imgData.data[idx + 1] = Math.floor(80 * (-val));
                    imgData.data[idx + 2] = Math.floor(255 * Math.min(1, (-val) * 2));
                }
                imgData.data[idx + 3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    drawWeights(canvas, omega) {
        if (!omega) return;

        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        const B = omega.length;
        const barW = w / B;
        const maxW = Math.E;   // theoretical max
        const midY = h / 2;

        // Reference line at ω = 1
        ctx.strokeStyle = '#444466';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        const refY = h - (1 / maxW) * h;
        ctx.beginPath();
        ctx.moveTo(0, refY);
        ctx.lineTo(w, refY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.font = '9px monospace';
        ctx.fillStyle = '#666';
        ctx.fillText('ω=1', 2, refY - 3);

        for (let b = 0; b < B; b++) {
            const val = omega[b];
            const barH = (val / maxW) * h;
            const x = b * barW;

            // Color: green if ≈1, red if high, blue if low
            let r, g, bl;
            if (val > 1) {
                const t = Math.min(1, (val - 1) / (maxW - 1));
                r = Math.floor(200 + 55 * t);
                g = Math.floor(180 * (1 - t));
                bl = 50;
            } else {
                const t = Math.min(1, (1 - val) / (1 - 1/maxW));
                r = 50;
                g = Math.floor(180 * (1 - t));
                bl = Math.floor(200 + 55 * t);
            }

            ctx.fillStyle = `rgb(${r},${g},${bl})`;
            ctx.fillRect(x + 1, h - barH, barW - 2, barH);
        }
    }
}

// ─── Playback ─────────────────────────────────────────────

class Playback {
    constructor() {
        this.ctx = null;
        this.sources = [];
        this.playing = false;
    }

    async play(tracks) {
        this.stop();
        this.ctx = new AudioContext();
        this.playing = true;

        for (const track of tracks) {
            const source = this.ctx.createBufferSource();
            source.buffer = track.audioBuffer;

            // Gain automation via gain node
            const gainNode = this.ctx.createGain();
            source.connect(gainNode);
            gainNode.connect(this.ctx.destination);

            // Schedule gain envelope
            const knots = track.gainKnots;
            const dur = track.duration;
            const K = knots.length;
            const now = this.ctx.currentTime;
            const startTime = now + Math.max(0, track.offset);
            const trackStart = Math.max(0, -track.offset);

            gainNode.gain.setValueAtTime(knots[0], startTime);
            for (let i = 1; i < K; i++) {
                const t = startTime + (i / (K - 1)) * dur;
                gainNode.gain.linearRampToValueAtTime(
                    Math.max(0, Math.min(1, knots[i])), t
                );
            }

            source.start(startTime, trackStart);
            this.sources.push({ source, gainNode });
        }
    }

    stop() {
        for (const s of this.sources) {
            try { s.source.stop(); } catch (e) {}
        }
        this.sources = [];
        if (this.ctx) {
            this.ctx.close();
            this.ctx = null;
        }
        this.playing = false;
    }
}

// ─── WAV Export ───────────────────────────────────────────

function exportWav(tracks, sr) {
    sr = sr || 44100;
    const hopSize = DSP.HOP_SIZE;

    // Determine output length
    let maxEndSample = 0;
    for (const t of tracks) {
        const end = Math.ceil((t.offset + t.duration) * sr);
        if (end > maxEndSample) maxEndSample = end;
    }

    const output = new Float32Array(maxEndSample);

    for (const track of tracks) {
        const offsetSample = Math.round(track.offset * sr);
        const gain = DSP.interpolateGain(track.gainKnots, track.samples.length);
        const samples = track.samples;

        for (let i = 0; i < samples.length; i++) {
            const outIdx = offsetSample + i;
            if (outIdx >= 0 && outIdx < maxEndSample) {
                output[outIdx] += samples[i] * gain[Math.min(i, gain.length - 1)];
            }
        }
    }

    // Normalize
    let peak = 0;
    for (let i = 0; i < output.length; i++) {
        const a = Math.abs(output[i]);
        if (a > peak) peak = a;
    }
    if (peak > 0) {
        const scale = 0.95 / peak;
        for (let i = 0; i < output.length; i++) output[i] *= scale;
    }

    // Encode WAV
    const nSamples = output.length;
    const buffer = new ArrayBuffer(44 + nSamples * 2);
    const view = new DataView(buffer);

    function writeStr(off, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    }

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + nSamples * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);       // PCM
    view.setUint16(22, 1, true);       // mono
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);  // byte rate
    view.setUint16(32, 2, true);       // block align
    view.setUint16(34, 16, true);      // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, nSamples * 2, true);

    for (let i = 0; i < nSamples; i++) {
        const s = Math.max(-1, Math.min(1, output[i]));
        view.setInt16(44 + i * 2, s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

// ─── Public API ───────────────────────────────────────────

return {
    TRACK_COLORS,
    Timeline, GainEditor, Visualiser, Playback,
    exportWav,
};

})();
