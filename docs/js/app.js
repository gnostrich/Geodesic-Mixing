/* ═══════════════════════════════════════════════════════════
   App: initialisation, file loading, test tone generation
   ═══════════════════════════════════════════════════════════ */

'use strict';

class App {
    constructor() {
        this.tracks = [];
        this.timeline = null;
        this.gainEditors = [];
        this.visualiser = null;
        this.playback = new UI.Playback();
        this.adaptive = false;
        this.lastResult = null;
    }

    init() {
        // Drop zone
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', e => {
            this.handleFiles(e.target.files);
            e.target.value = '';
        });

        // Generate test tones
        document.getElementById('gen-test-btn').addEventListener('click', () => {
            this.generateTestTones();
        });

        // Controls
        document.getElementById('play-btn').addEventListener('click', () => this.play());
        document.getElementById('stop-btn').addEventListener('click', () => this.stop());
        document.getElementById('optimize-btn').addEventListener('click', () => this.optimize());
        document.getElementById('adaptive-toggle').addEventListener('change', e => {
            this.adaptive = e.target.checked;
            document.getElementById('weights-panel').style.display = this.adaptive ? '' : 'none';
            this.onConfigChanged();
        });

        // Zoom
        document.getElementById('zoom-in-btn').addEventListener('click', () => {
            if (this.timeline) this.timeline.zoom(1.3);
        });
        document.getElementById('zoom-out-btn').addEventListener('click', () => {
            if (this.timeline) this.timeline.zoom(1 / 1.3);
        });

        // Export
        document.getElementById('export-config-btn').addEventListener('click', () => this.exportConfig());
        document.getElementById('export-wav-btn').addEventListener('click', () => this.exportWavFile());

        // Timeline
        this.timeline = new UI.Timeline(
            document.getElementById('timeline-canvas'), this
        );

        // Visualiser
        this.visualiser = new UI.Visualiser(
            document.getElementById('mix-spec-canvas'),
            document.getElementById('flux-canvas'),
            document.getElementById('weights-canvas'),
        );
    }

    setStatus(msg) {
        const el = document.getElementById('status-bar');
        if (el) el.textContent = msg;
    }

    // ─── File handling ────────────────────────────────────

    async handleFiles(fileList) {
        const audioCtx = new AudioContext();
        for (const file of fileList) {
            if (!file.type.startsWith('audio/') && !file.name.match(/\.(wav|mp3|ogg|flac|m4a|aac)$/i)) {
                continue;
            }
            this.setStatus(`Loading ${file.name}...`);
            try {
                const arrayBuf = await file.arrayBuffer();
                const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
                this.addTrack(file.name, audioBuf, audioCtx.sampleRate);
            } catch (err) {
                console.error('Failed to decode:', file.name, err);
                this.setStatus(`Failed to load ${file.name}`);
            }
        }
        audioCtx.close();
    }

    addTrack(name, audioBuffer, sr) {
        const track = new Engine.Track(name, audioBuffer, sr);
        track.color = UI.TRACK_COLORS[this.tracks.length % UI.TRACK_COLORS.length];

        // Default offset: after the last track with some overlap
        if (this.tracks.length > 0) {
            const last = this.tracks[this.tracks.length - 1];
            track.offset = Math.max(0, last.offset + last.duration - 15);
        }

        this.setStatus(`Computing spectrogram for ${name}...`);
        track.computeSpectrogram();

        this.tracks.push(track);
        this.updateTrackList();
        this.showSections();
        this.rebuildGainEditors();
        this.timeline.render();
        this.onConfigChanged();
        this.setStatus(`Added ${name} (${track.duration.toFixed(1)}s)`);
    }

    removeTrack(index) {
        this.tracks.splice(index, 1);
        this.updateTrackList();
        this.rebuildGainEditors();
        if (this.tracks.length === 0) this.hideSections();
        else {
            this.timeline.render();
            this.onConfigChanged();
        }
    }

    // ─── Track list UI ────────────────────────────────────

    updateTrackList() {
        const list = document.getElementById('track-list');
        list.innerHTML = '';
        this.tracks.forEach((track, i) => {
            const row = document.createElement('div');
            row.className = 'track-item';
            row.style.borderLeftColor = track.color;

            const dot = document.createElement('div');
            dot.className = 'track-color';
            dot.style.background = track.color;

            const name = document.createElement('div');
            name.className = 'track-name';
            name.textContent = track.name;

            const meta = document.createElement('div');
            meta.className = 'track-meta';
            meta.textContent = `${track.duration.toFixed(1)}s @ ${track.sr} Hz`;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'btn btn-danger';
            removeBtn.textContent = '×';
            removeBtn.addEventListener('click', () => this.removeTrack(i));

            row.appendChild(dot);
            row.appendChild(name);
            row.appendChild(meta);
            row.appendChild(removeBtn);
            list.appendChild(row);
        });
    }

    showSections() {
        for (const id of ['timeline-section', 'controls-section', 'gain-section', 'viz-section', 'export-section']) {
            document.getElementById(id).style.display = '';
        }
    }

    hideSections() {
        for (const id of ['timeline-section', 'controls-section', 'gain-section', 'viz-section', 'export-section']) {
            document.getElementById(id).style.display = 'none';
        }
    }

    rebuildGainEditors() {
        const container = document.getElementById('gain-editors');
        container.innerHTML = '';
        this.gainEditors = [];
        this.tracks.forEach((track, i) => {
            const editor = new UI.GainEditor(container, track, i, this);
            this.gainEditors.push(editor);
        });
    }

    // ─── Config change handler ────────────────────────────

    onConfigChanged() {
        if (this.tracks.length === 0) return;

        const ready = this.tracks.every(t => t.spec);
        if (!ready) return;

        // Compute energy
        const result = Engine.computeEnergy(this.tracks, this.adaptive);
        this.lastResult = result;

        // Update energy display
        const eStr = result.energy < 1e6
            ? result.energy.toFixed(2)
            : result.energy.toExponential(3);
        document.getElementById('energy-value').textContent = eStr;

        // Update visualisations
        if (result.mix) {
            this.visualiser.drawSpectrogram(
                document.getElementById('mix-spec-canvas'),
                result.mix.mag, result.mix.nBins, result.mix.nFrames
            );
        }
        if (result.flux) {
            this.visualiser.drawFlux(
                document.getElementById('flux-canvas'),
                result.flux
            );
        }
        if (result.omega) {
            this.visualiser.drawWeights(
                document.getElementById('weights-canvas'),
                result.omega
            );
        }

        // Update timeline info
        let totalDur = 0;
        for (const t of this.tracks) {
            const end = t.offset + t.duration;
            if (end > totalDur) totalDur = end;
        }
        document.getElementById('timeline-info').textContent =
            `${this.tracks.length} tracks · ${totalDur.toFixed(1)}s total`;
    }

    // ─── Playback ─────────────────────────────────────────

    play() {
        this.playback.play(this.tracks);
        this.setStatus('Playing...');
    }

    stop() {
        this.playback.stop();
        this.setStatus('Stopped');
    }

    // ─── Optimizer ────────────────────────────────────────

    async optimize() {
        if (this.tracks.length < 2) {
            this.setStatus('Need at least 2 tracks to optimise');
            return;
        }

        const btn = document.getElementById('optimize-btn');
        btn.disabled = true;
        btn.classList.add('computing');
        this.setStatus('Optimising offsets...');

        // Run in next frame to allow UI update
        await new Promise(r => setTimeout(r, 50));

        try {
            Engine.optimizeSweep(this.tracks, this.adaptive, (progress, energy) => {
                this.setStatus(`Optimising... ${(progress * 100).toFixed(0)}% (E=${energy.toFixed(1)})`);
            });

            this.setStatus('Optimising gains...');
            await new Promise(r => setTimeout(r, 50));

            Engine.optimizeGains(this.tracks, this.adaptive, (progress, energy) => {
                this.setStatus(`Gain opt... ${(progress * 100).toFixed(0)}% (E=${energy.toFixed(1)})`);
            });

            this.timeline.render();
            this.rebuildGainEditors();
            this.onConfigChanged();
            this.setStatus('Optimisation complete');
        } catch (err) {
            console.error('Optimisation error:', err);
            this.setStatus('Optimisation failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.classList.remove('computing');
        }
    }

    // ─── Test tone generation ─────────────────────────────

    async generateTestTones() {
        this.setStatus('Generating test tones...');
        const sr = 44100;
        const duration = 20; // seconds each

        const tones = [
            { name: 'Kick + Bass (120 BPM)', bpm: 120, type: 'kick_bass' },
            { name: 'Synth Pad (122 BPM)',   bpm: 122, type: 'pad' },
            { name: 'Hi-Hats + Lead (118 BPM)', bpm: 118, type: 'hats_lead' },
        ];

        for (const tone of tones) {
            const buffer = generateTone(tone.type, duration, tone.bpm, sr);
            this.addTrack(tone.name, buffer, sr);
        }
        this.setStatus('Test tones ready');
    }

    // ─── Export ───────────────────────────────────────────

    exportConfig() {
        const config = {
            tracks: this.tracks.map(t => ({
                name: t.name,
                offset: t.offset,
                duration: t.duration,
                gainKnots: Array.from(t.gainKnots),
            })),
            adaptive: this.adaptive,
            energy: this.lastResult ? this.lastResult.energy : null,
        };
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'mix-config.json');
    }

    exportWavFile() {
        this.setStatus('Exporting WAV...');
        const sr = this.tracks.length > 0 ? this.tracks[0].sr : 44100;
        const blob = UI.exportWav(this.tracks, sr);
        downloadBlob(blob, 'geodesic-mix.wav');
        this.setStatus('WAV exported');
    }
}

// ─── Synthetic tone generator ─────────────────────────────

function generateTone(type, duration, bpm, sr) {
    const nSamples = Math.floor(duration * sr);
    const data = new Float32Array(nSamples);
    const beatInterval = 60 / bpm; // seconds per beat

    switch (type) {
        case 'kick_bass': {
            for (let i = 0; i < nSamples; i++) {
                const t = i / sr;
                const beatPhase = (t % beatInterval) / beatInterval;
                // Kick: short low-freq burst on each beat
                const kickEnv = Math.exp(-beatPhase * 30);
                const kickFreq = 55 * (1 + 3 * Math.exp(-beatPhase * 20));
                const kick = Math.sin(2 * Math.PI * kickFreq * t) * kickEnv * 0.5;
                // Bass: sustained low note
                const bassFreq = 55 * (1 + 0.5 * Math.sin(2 * Math.PI * t / (beatInterval * 4)));
                const bass = Math.sin(2 * Math.PI * bassFreq * t) * 0.2;
                data[i] = kick + bass;
            }
            break;
        }
        case 'pad': {
            for (let i = 0; i < nSamples; i++) {
                const t = i / sr;
                // Rich harmonic pad
                let val = 0;
                const baseFreq = 220;
                for (let h = 1; h <= 6; h++) {
                    val += Math.sin(2 * Math.PI * baseFreq * h * t + h * 0.3) / h;
                }
                // Slow amplitude modulation
                val *= 0.15 * (0.7 + 0.3 * Math.sin(2 * Math.PI * t / (beatInterval * 8)));
                // Add subtle rhythm
                const beatPhase = (t % beatInterval) / beatInterval;
                val *= 0.8 + 0.2 * Math.cos(2 * Math.PI * beatPhase);
                data[i] = val;
            }
            break;
        }
        case 'hats_lead': {
            // Seed a simple RNG for the noise
            let rng = 42;
            const rand = () => { rng = (rng * 1103515245 + 12345) & 0x7fffffff; return rng / 0x7fffffff * 2 - 1; };

            for (let i = 0; i < nSamples; i++) {
                const t = i / sr;
                const eighthInterval = beatInterval / 2;
                const eighthPhase = (t % eighthInterval) / eighthInterval;
                // Hi-hat: noise burst on 8th notes
                const hatEnv = Math.exp(-eighthPhase * 40);
                const hat = rand() * hatEnv * 0.15;
                // Lead: simple melody pattern
                const bar = Math.floor(t / (beatInterval * 4));
                const beatInBar = Math.floor((t % (beatInterval * 4)) / beatInterval);
                const notes = [440, 494, 523, 587]; // A4, B4, C5, D5
                const noteFreq = notes[beatInBar % notes.length] * (1 + (bar % 2) * 0.5);
                const noteEnv = Math.exp(-((t % beatInterval) / beatInterval) * 5);
                const lead = Math.sin(2 * Math.PI * noteFreq * t) * noteEnv * 0.15;
                data[i] = hat + lead;
            }
            break;
        }
    }

    // Create AudioBuffer
    const ctx = new OfflineAudioContext(1, nSamples, sr);
    const buf = ctx.createBuffer(1, nSamples, sr);
    buf.getChannelData(0).set(data);
    return buf;
}

// ─── Utility ──────────────────────────────────────────────

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Boot ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window.app.init();
});
