/* ═══════════════════════════════════════════════════════════════════════════
   CCTV Anomaly Detection — Dashboard JavaScript
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── STATE ───
const state = {
    currentSection: 'dashboard',
    selectedAnomalyType: 'fire',
    selectedAudioType: 'ambient',
    modulesLoaded: { dreaming: false, audio_visual: false, reasoning: false },
};

// ─── NAVIGATION ───
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const section = item.dataset.section;
        switchSection(section);
    });
});

function switchSection(sectionId) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector(`.nav-item[data-section="${sectionId}"]`)?.classList.add('active');

    // Update sections
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(`section-${sectionId}`);
    if (target) target.classList.add('active');

    state.currentSection = sectionId;
}

// ─── LOADING OVERLAY ───
function showLoading(text = 'Processing…') {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-text').textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

// ─── API HELPER ───
async function api(url, options = {}) {
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const config = { ...defaults, ...options };
    try {
        const res = await fetch(url, config);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        return data;
    } catch (err) {
        console.error(`API error [${url}]:`, err);
        throw err;
    }
}


// ═══════ SYSTEM INITIALIZATION ═══════

async function checkStatus() {
    try {
        const data = await api('/api/system/status');
        updateModuleStatus(data);
    } catch (e) {
        console.error('Status check failed:', e);
    }
}

function updateModuleStatus(data) {
    const modules = ['dreaming', 'audio_visual', 'reasoning'];
    let allLoaded = true;

    modules.forEach(mod => {
        const dotId = `status-${mod.replace('_', '-')}`;
        const dot = document.getElementById(dotId);
        const card = document.getElementById(`module-${mod.replace('_', '-')}`);
        const isLoaded = data[mod]?.loaded;

        state.modulesLoaded[mod] = isLoaded;

        if (dot) {
            dot.className = 'module-status-dot';
            if (isLoaded) {
                dot.classList.add('online');
            } else if (data[mod]?.error) {
                dot.classList.add('error');
            }
        }

        if (card) {
            card.classList.toggle('loaded', isLoaded);
        }

        if (!isLoaded) allLoaded = false;
    });

    // Update sidebar footer
    const indicator = document.getElementById('system-status-indicator');
    const statusDot = indicator.querySelector('.status-dot');
    const statusText = indicator.querySelector('span');

    if (allLoaded) {
        statusDot.className = 'status-dot online';
        statusText.textContent = 'All Modules Online';
    } else if (Object.values(state.modulesLoaded).some(v => v)) {
        statusDot.className = 'status-dot online';
        const count = Object.values(state.modulesLoaded).filter(v => v).length;
        statusText.textContent = `${count}/3 Modules Online`;
    } else {
        statusDot.className = 'status-dot';
        statusText.textContent = 'Modules Not Initialized';
    }
}

async function initAllModules() {
    const btn = document.getElementById('btn-init-all');
    btn.disabled = true;
    btn.innerHTML = `<span class="loading-spinner" style="width:16px;height:16px;border-width:2px;display:inline-block"></span> Initializing…`;
    showLoading('Initializing all modules… This may take 30-60 seconds.');

    try {
        const data = await api('/api/system/init', { method: 'POST' });
        updateModuleStatus(data);
    } catch (e) {
        console.error('Init failed:', e);
    }

    hideLoading();
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Initialize All Modules`;
}

async function initModule(moduleName) {
    const dotId = `status-${moduleName.replace('_', '-')}`;
    const dot = document.getElementById(dotId);
    if (dot) {
        dot.className = 'module-status-dot loading';
    }
    showLoading(`Initializing ${moduleName.replace('_', '-')}…`);

    try {
        const data = await api(`/api/system/init/${moduleName}`, { method: 'POST' });
        updateModuleStatus(data);
    } catch (e) {
        console.error(`Init ${moduleName} failed:`, e);
        if (dot) dot.className = 'module-status-dot error';
    }

    hideLoading();
}


// ═══════ DREAMING ENGINE ═══════

function selectAnomalyType(btn) {
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.selectedAnomalyType = btn.dataset.type;
}

async function generateDream() {
    if (!state.modulesLoaded.dreaming) {
        alert('Please initialize the Dreaming Engine first (Dashboard → Initialize)');
        return;
    }

    const numFrames = parseInt(document.getElementById('dream-frames').value);
    const brightness = parseInt(document.getElementById('dream-brightness').value) / 100;

    showLoading('Generating synthetic anomalies…');

    try {
        const data = await api('/api/dreaming/generate', {
            method: 'POST',
            body: JSON.stringify({
                anomaly_type: state.selectedAnomalyType,
                num_frames: numFrames,
                brightness: brightness,
            }),
        });

        // Show background
        const bgFrame = document.getElementById('dream-bg-frame');
        bgFrame.innerHTML = `<img src="data:image/png;base64,${data.background}" alt="Clean Background">`;

        // Show first anomaly frame
        const anomalyFrame = document.getElementById('dream-anomaly-frame');
        if (data.frames.length > 0) {
            anomalyFrame.innerHTML = `<img src="data:image/png;base64,${data.frames[0]}" alt="Synthetic Anomaly">`;
        }

        // Gallery
        const gallery = document.getElementById('dream-gallery');
        const gallerySection = document.getElementById('dream-gallery-section');
        gallery.innerHTML = data.frames.map((f, i) =>
            `<img src="data:image/png;base64,${f}" alt="Frame ${i + 1}" title="Frame ${i + 1}">`
        ).join('');
        gallerySection.style.display = 'block';

    } catch (e) {
        alert('Error generating anomalies: ' + e.message);
    }

    hideLoading();
}


// ═══════ AUDIO-VISUAL ═══════

function selectAudioType(btn) {
    document.querySelectorAll('.audio-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.selectedAudioType = btn.dataset.type;
}

function drawWaveform(points) {
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 240;
    canvas.style.height = '120px';

    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.06)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (!points || points.length === 0) return;

    // Waveform
    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#6366f1');
    gradient.addColorStop(0.5, '#06b6d4');
    gradient.addColorStop(1, '#8b5cf6');

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = w / points.length;
    const mid = h / 2;
    const maxVal = Math.max(...points.map(Math.abs), 0.001);

    points.forEach((val, i) => {
        const x = i * step;
        const y = mid - (val / maxVal) * (mid * 0.85);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();

    // Fill under curve
    ctx.lineTo(w, mid);
    ctx.lineTo(0, mid);
    ctx.closePath();

    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, 'rgba(99, 102, 241, 0.08)');
    fillGrad.addColorStop(1, 'rgba(99, 102, 241, 0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();
}

async function runAudioVisual() {
    if (!state.modulesLoaded.audio_visual) {
        alert('Please initialize the Audio-Visual module first (Dashboard → Initialize)');
        return;
    }

    showLoading('Running cross-modal detection…');

    try {
        const data = await api('/api/audio-visual/detect', {
            method: 'POST',
            body: JSON.stringify({
                audio_type: state.selectedAudioType,
                brightness: 0.5,
            }),
        });

        // Frame
        const avFrame = document.getElementById('av-frame');
        avFrame.innerHTML = `<img src="data:image/png;base64,${data.frame}" alt="Visual Scene">`;

        // Waveform
        drawWaveform(data.waveform);

        // Metrics
        document.getElementById('av-similarity').textContent = data.similarity.toFixed(4);
        document.getElementById('av-zscore').textContent = data.z_score.toFixed(4);
        document.getElementById('av-severity').textContent = data.severity.toUpperCase();

        // Bars
        const simPct = Math.max(0, Math.min(100, (data.similarity + 1) * 50));
        document.getElementById('av-sim-bar').style.width = `${simPct}%`;

        const zPct = Math.max(0, Math.min(100, Math.abs(data.z_score) * 15));
        document.getElementById('av-z-bar').style.width = `${zPct}%`;

        // Severity color
        const sevEl = document.getElementById('av-severity');
        sevEl.style.color = data.severity === 'critical' ? 'var(--danger)' :
                            data.severity === 'high' ? '#f97316' :
                            data.severity === 'medium' ? 'var(--warning)' :
                            'var(--success)';

        // Verdict
        const verdict = document.getElementById('av-verdict');
        verdict.style.display = 'flex';
        verdict.className = `av-verdict glass-card ${data.is_anomaly ? 'anomaly' : 'normal'}`;

        document.getElementById('av-verdict-icon').textContent = data.is_anomaly ? '⚠️' : '✅';
        document.getElementById('av-verdict-label').textContent = data.is_anomaly
            ? 'CROSS-MODAL ANOMALY DETECTED'
            : 'AUDIO-VIDEO CONSISTENT — No Anomaly';
        document.getElementById('av-verdict-label').style.color = data.is_anomaly
            ? 'var(--danger)' : 'var(--success)';
        document.getElementById('av-verdict-desc').textContent = data.is_anomaly
            ? `Audio (${data.audio_type}) does not match the calm visual scene. Z-score: ${data.z_score.toFixed(2)}`
            : `Audio (${data.audio_type}) matches the visual scene normally.`;

    } catch (e) {
        alert('Error running detection: ' + e.message);
    }

    hideLoading();
}


// ═══════ REASONING ENGINE ═══════

async function loadPresets() {
    try {
        const data = await api('/api/reasoning/presets');
        const grid = document.getElementById('preset-grid');

        grid.innerHTML = data.presets.map(p => `
            <div class="preset-card" onclick='applyPreset(${JSON.stringify(p.params)})'>
                <div class="preset-name">${p.name}</div>
                <div class="preset-desc">${p.desc}</div>
                <span class="preset-expected ${p.expected}">${p.expected}</span>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load presets:', e);
    }
}

function applyPreset(params) {
    document.getElementById('reason-alert-type').value = params.alert_type;
    document.getElementById('reason-location').value = params.location;

    document.getElementById('reason-score').value = Math.round(params.anomaly_score * 100);
    document.getElementById('reason-score-val').textContent = params.anomaly_score.toFixed(2);

    document.getElementById('reason-brightness').value = Math.round(params.brightness * 100);
    document.getElementById('reason-bright-val').textContent = params.brightness.toFixed(2);

    document.getElementById('reason-av-score').value = Math.round(params.audio_visual_score * 100);
    document.getElementById('reason-av-val').textContent = params.audio_visual_score.toFixed(2);

    // Auto-run
    runReasoning();
}

async function runReasoning() {
    if (!state.modulesLoaded.reasoning) {
        alert('Please initialize the Reasoning Engine first (Dashboard → Initialize)');
        return;
    }

    const params = {
        alert_type: document.getElementById('reason-alert-type').value,
        anomaly_score: parseInt(document.getElementById('reason-score').value) / 100,
        location: document.getElementById('reason-location').value,
        brightness: parseInt(document.getElementById('reason-brightness').value) / 100,
        audio_visual_score: parseInt(document.getElementById('reason-av-score').value) / 100,
    };

    showLoading('Applying common sense reasoning…');

    try {
        const data = await api('/api/reasoning/process', {
            method: 'POST',
            body: JSON.stringify(params),
        });

        // Decision Card
        const card = document.getElementById('decision-card');
        const icons = { suppress: '🟢', confirm: '🟡', escalate: '🔴' };

        card.innerHTML = `
            <div class="decision-display">
                <div class="decision-icon">${icons[data.decision] || '❓'}</div>
                <div class="decision-label ${data.decision}">${data.decision}</div>
                <div class="decision-confidence">Confidence: ${(data.confidence * 100).toFixed(0)}%</div>
                <div class="decision-reasoning">${data.reasoning}</div>
                <div class="decision-llm-badge">${data.llm_used ? '🤖 LLM-Powered' : '📋 Rule-Based'}</div>
            </div>
        `;

        // Glow effect on card
        card.style.borderColor = data.decision === 'suppress' ? 'rgba(16,185,129,0.3)' :
                                  data.decision === 'confirm' ? 'rgba(245,158,11,0.3)' :
                                  'rgba(239,68,68,0.3)';

        // Scene Context
        if (data.scene_context) {
            const ctx = data.scene_context;
            const contextCard = document.getElementById('context-card');
            contextCard.style.display = 'block';
            document.getElementById('context-grid').innerHTML = `
                <div class="context-item"><span class="ctx-label">Location</span><span class="ctx-value">${ctx.location}</span></div>
                <div class="context-item"><span class="ctx-label">Time</span><span class="ctx-value">${ctx.time_of_day}</span></div>
                <div class="context-item"><span class="ctx-label">Crowd</span><span class="ctx-value">${ctx.crowd_density}</span></div>
                <div class="context-item"><span class="ctx-label">Lighting</span><span class="ctx-value">${ctx.lighting}</span></div>
                <div class="context-item"><span class="ctx-label">Actions</span><span class="ctx-value">${(ctx.actions || []).join(', ') || '—'}</span></div>
                <div class="context-item"><span class="ctx-label">Objects</span><span class="ctx-value">${(ctx.objects || []).join(', ') || '—'}</span></div>
            `;
        }

        // Stats
        if (data.stats) {
            const s = data.stats;
            const statsCard = document.getElementById('reasoning-stats-card');
            statsCard.style.display = 'block';
            document.getElementById('reasoning-mini-stats').innerHTML = `
                <div class="mini-stat"><span class="ms-value">${s.total_alerts}</span><span class="ms-label">Total</span></div>
                <div class="mini-stat"><span class="ms-value" style="color:var(--success)">${s.suppressed}</span><span class="ms-label">Suppressed</span></div>
                <div class="mini-stat"><span class="ms-value" style="color:var(--warning)">${s.confirmed}</span><span class="ms-label">Confirmed</span></div>
                <div class="mini-stat"><span class="ms-value" style="color:var(--danger)">${s.escalated}</span><span class="ms-label">Escalated</span></div>
            `;

            // Update dashboard stats too
            document.getElementById('stat-total').textContent = s.total_alerts;
            document.getElementById('stat-suppressed').textContent = s.suppressed;
            document.getElementById('stat-confirmed').textContent = s.confirmed;
            document.getElementById('stat-escalated').textContent = s.escalated;
        }

    } catch (e) {
        alert('Error processing alert: ' + e.message);
    }

    hideLoading();
}


// ═══════ FULL PIPELINE ═══════

async function runPipeline() {
    const anomaly = document.getElementById('pipe-anomaly').value;
    const audio = document.getElementById('pipe-audio').value;
    const location = document.getElementById('pipe-location').value;

    // Reset steps
    [5, 6, 7].forEach(phase => {
        const status = document.getElementById(`pipe-status-${phase}`);
        const step = document.getElementById(`pipe-step-${phase}`);
        status.textContent = 'Waiting';
        status.className = 'step-status';
        step.className = 'pipeline-step glass-card';
        const body = document.getElementById(`pipe-body-${phase}`);
        const existingResult = body.querySelector('.step-result');
        if (existingResult) existingResult.remove();
    });

    // Animate each step
    const setStepActive = (phase) => {
        document.getElementById(`pipe-step-${phase}`).classList.add('active');
        const status = document.getElementById(`pipe-status-${phase}`);
        status.textContent = 'Running…';
        status.className = 'step-status running';
    };

    setStepActive(5);

    try {
        const data = await api('/api/pipeline/run', {
            method: 'POST',
            body: JSON.stringify({
                anomaly_type: anomaly,
                audio_type: audio,
                location: location,
                brightness: 0.55,
            }),
        });

        data.steps.forEach(step => {
            const phase = step.phase;
            const stepEl = document.getElementById(`pipe-step-${phase}`);
            const status = document.getElementById(`pipe-status-${phase}`);
            const body = document.getElementById(`pipe-body-${phase}`);

            stepEl.classList.remove('active');

            if (step.status === 'success') {
                stepEl.classList.add('done');
                status.textContent = 'Complete';
                status.className = 'step-status done';

                let resultHtml = '<div class="step-result">';
                if (phase === 5) {
                    resultHtml += `
                        <div class="sr-item"><span class="sr-label">Type</span><span class="sr-value">${step.data.anomaly_type}</span></div>
                        <div class="sr-item"><span class="sr-label">Frames</span><span class="sr-value">${step.data.frames_generated}</span></div>
                    `;
                    if (step.data.sample_frame) {
                        resultHtml += `<img src="data:image/png;base64,${step.data.sample_frame}" alt="Sample" style="width:100%;margin-top:8px;border-radius:8px;">`;
                    }
                } else if (phase === 6) {
                    const anomalyIcon = step.data.is_anomaly ? '⚠️' : '✅';
                    resultHtml += `
                        <div class="sr-item"><span class="sr-label">Audio</span><span class="sr-value">${step.data.audio_type}</span></div>
                        <div class="sr-item"><span class="sr-label">Similarity</span><span class="sr-value">${step.data.similarity}</span></div>
                        <div class="sr-item"><span class="sr-label">Z-Score</span><span class="sr-value">${step.data.z_score}</span></div>
                        <div class="sr-item"><span class="sr-label">Anomaly</span><span class="sr-value">${anomalyIcon} ${step.data.severity}</span></div>
                    `;
                } else if (phase === 7) {
                    const icons = { suppress: '🟢', confirm: '🟡', escalate: '🔴' };
                    resultHtml += `
                        <div class="sr-item"><span class="sr-label">Decision</span><span class="sr-value">${icons[step.data.decision]} ${step.data.decision.toUpperCase()}</span></div>
                        <div class="sr-item"><span class="sr-label">Confidence</span><span class="sr-value">${(step.data.confidence * 100).toFixed(0)}%</span></div>
                        <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);border-left:2px solid var(--accent-primary);padding-left:10px;">${step.data.reasoning}</div>
                    `;
                }
                resultHtml += '</div>';
                body.insertAdjacentHTML('beforeend', resultHtml);
            } else {
                status.textContent = 'Unavailable';
                status.className = 'step-status error';
            }
        });

    } catch (e) {
        alert('Pipeline error: ' + e.message);
        [5, 6, 7].forEach(phase => {
            const status = document.getElementById(`pipe-status-${phase}`);
            status.textContent = 'Error';
            status.className = 'step-status error';
        });
    }
}


// ═══════ INIT ═══════

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadPresets();
    drawWaveform([]);
});
