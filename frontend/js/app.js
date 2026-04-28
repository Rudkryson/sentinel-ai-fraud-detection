/**
 * Sentinel AI Fraud Intelligence Platform - Main JS
 */

const state = {
    activePage: 'dashboard',
    stats: null,
    alerts: [],
    lastPrediction: null
};

// ─────────────────────────────────────────────
// UI COMPONENTS (Templates)
// ─────────────────────────────────────────────

const Templates = {
    dashboard: () => `
        <div class="kpi-grid">
            <div class="kpi-card">
                <p class="kpi-title">Total Transactions</p>
                <p class="kpi-value">${state.stats?.total_transactions.toLocaleString() || '145,280'}</p>
                <p class="kpi-trend up">↑ 12.5% vs last month</p>
            </div>
            <div class="kpi-card">
                <p class="kpi-title">Fraud Detected</p>
                <p class="kpi-value">${state.stats?.fraud_detected.toLocaleString() || '3,842'}</p>
                <p class="kpi-trend down">↑ 4.2% higher risk</p>
            </div>
            <div class="kpi-card">
                <p class="kpi-title">Risk Alerts</p>
                <p class="kpi-value">${state.stats?.risk_alerts || '156'}</p>
                <p class="kpi-trend up">↓ 8% from yesterday</p>
            </div>
            <div class="kpi-card">
                <p class="kpi-title">Fraud Rate</p>
                <p class="kpi-value">${state.stats?.fraud_rate || '2.64%'}</p>
                <p class="kpi-trend">Stable</p>
            </div>
        </div>

        <div class="main-grid">
            <div class="panel">
                <div class="panel-header">
                    <h3>Risk Trend (Real-time)</h3>
                </div>
                <div style="height: 300px;">
                    <canvas id="riskTrendChart"></canvas>
                </div>
            </div>
            <div class="panel">
                <div class="panel-header">
                    <h3>Live Alerts</h3>
                    <a href="#" style="font-size: 0.8rem; color: var(--primary);">View All</a>
                </div>
                <div class="alert-list" id="dashboard-alerts">
                    ${renderAlertList(state.alerts)}
                </div>
            </div>
        </div>
    `,

    analysis: () => `
        <div class="panel">
            <div class="panel-header">
                <h3>Manual Transaction Scoring</h3>
                <p style="font-size: 0.85rem; color: var(--text-dim);">Input details for instant AI risk assessment</p>
            </div>
            
            <form id="analysis-form" class="analysis-form">
                <div class="form-group">
                    <label>Transaction Amount ($)</label>
                    <input type="number" name="amount" placeholder="e.g. 5000.00" step="0.01" required>
                </div>
                <div class="form-group">
                    <label>Transaction Hour (0-23)</label>
                    <input type="number" name="transaction_hour" min="0" max="23" placeholder="14" required>
                </div>
                <div class="form-group">
                    <label>Location</label>
                    <select name="location">
                        <option value="urban">Urban</option>
                        <option value="suburban">Suburban</option>
                        <option value="online">Online</option>
                        <option value="international">International</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Transaction Type</label>
                    <select name="transaction_type">
                        <option value="pos">Point of Sale (POS)</option>
                        <option value="online">Online Payment</option>
                        <option value="atm">ATM Withdrawal</option>
                        <option value="wire_transfer">Wire Transfer</option>
                        <option value="contactless">Contactless</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Frequency (7-day count)</label>
                    <input type="number" name="transaction_freq_7d" min="1" placeholder="5" required>
                </div>
                <div class="form-group">
                    <label>Avg Amount (7-day)</label>
                    <input type="number" name="avg_amount_7d" placeholder="250.00" step="0.01" required>
                </div>
                
                <input type="hidden" name="transaction_day" value="1">
                <input type="hidden" name="amount_deviation" value="0.0">
                <input type="hidden" name="is_night" value="0">
                <input type="hidden" name="is_weekend" value="0">

                <button type="submit" class="submit-btn" id="analyze-btn">Analyze Transaction</button>
            </form>

            <div id="result-container" class="result-card">
                <!-- Result content -->
            </div>
        </div>
    `,

    alerts: () => `
        <div class="panel">
            <div class="panel-header">
                <h3>System Intelligence Alerts</h3>
                <div class="filters">
                    <span class="badge" style="background: var(--danger);">Critical</span>
                    <span class="badge" style="background: var(--warning);">High</span>
                </div>
            </div>
            <div class="alert-list" style="margin-top: 1rem;">
                ${renderAlertList(state.alerts, true)}
            </div>
        </div>
    `,

    analytics: () => `
        <div class="main-grid" style="grid-template-columns: 1fr 1fr;">
            <div class="panel">
                <h3>Feature Importance</h3>
                <div style="height: 350px;"><canvas id="featureChart"></canvas></div>
            </div>
            <div class="panel">
                <h3>Risk Score Distribution</h3>
                <div style="height: 350px;"><canvas id="distChart"></canvas></div>
            </div>
            <div class="panel full-width" style="grid-column: span 2; margin-top: 2rem;">
                <h3>Model Performance Matrix</h3>
                <div style="display: flex; gap: 2rem; align-items: center; justify-content: center; padding: 2rem;">
                    <div class="metrics-block">
                        <p class="kpi-title">ROC-AUC</p>
                        <p class="kpi-value" style="color: var(--success);">0.998</p>
                    </div>
                    <div class="metrics-block">
                        <p class="kpi-title">Precision</p>
                        <p class="kpi-value" style="color: var(--accent);">0.985</p>
                    </div>
                    <div class="metrics-block">
                        <p class="kpi-title">Recall</p>
                        <p class="kpi-value" style="color: var(--warning);">0.991</p>
                    </div>
                    <div class="metrics-block">
                        <p class="kpi-title">F1 Score</p>
                        <p class="kpi-value">0.988</p>
                    </div>
                </div>
            </div>
        </div>
    `,

    batch: () => `
        <div class="panel">
            <div class="panel-header">
                <h3>Batch Transaction Processing</h3>
            </div>
            <div class="upload-area" id="drop-zone" style="border: 2px dashed var(--glass-border); padding: 4rem; text-align: center; border-radius: 16px; cursor: pointer;">
                <p style="font-size: 2rem; margin-bottom: 1rem;">📁</p>
                <p>Drag and drop CSV file here or <span style="color: var(--primary);">click to browse</span></p>
                <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 1rem;">Supported formats: .CSV (Max 10MB)</p>
                <input type="file" id="batch-file" hidden accept=".csv">
            </div>
            
            <div id="batch-results" style="margin-top: 2rem; display: none;">
                <h3>Results Preview</h3>
                <table style="width: 100%; margin-top: 1rem; border-collapse: collapse;">
                    <thead style="background: rgba(255,255,255,0.05);">
                        <tr>
                            <th style="padding: 0.75rem; text-align: left;">Index</th>
                            <th style="padding: 0.75rem; text-align: left;">Amount</th>
                            <th style="padding: 0.75rem; text-align: left;">Type</th>
                            <th style="padding: 0.75rem; text-align: left;">Risk Score</th>
                            <th style="padding: 0.75rem; text-align: left;">Status</th>
                        </tr>
                    </thead>
                    <tbody id="batch-table-body"></tbody>
                </table>
            </div>
        </div>
    `,

    settings: () => `
        <div class="panel">
            <h3>Account Settings</h3>
            <div style="margin-top: 2rem; border-bottom: 1px solid var(--glass-border); padding-bottom: 2rem;">
                <p style="font-weight: 600;">System Configuration</p>
                <div class="form-group" style="margin-top: 1rem;">
                    <label>Fraud Probability Threshold</label>
                    <input type="range" min="0" max="1" step="0.05" value="0.4">
                    <span style="font-size: 0.8rem;">Current: 0.40 (Recommended)</span>
                </div>
            </div>
            <div style="margin-top: 2rem;">
                <p style="font-weight: 600;">Security Profile</p>
                <p style="font-size: 0.85rem; color: var(--text-dim); margin-top: 0.5rem;">Two-Factor Authentication: <span style="color: var(--success);">Enabled</span></p>
            </div>
        </div>
    `
};

// ─────────────────────────────────────────────
// RENDER HELPERS
// ─────────────────────────────────────────────

function renderAlertList(alerts, detailed = false) {
    if (!alerts || alerts.length === 0) return '<p style="color: var(--text-dim); text-align: center; padding: 2rem;">No active alerts</p>';
    
    return alerts.map(alert => `
        <div class="alert-item ${alert.risk_level.toLowerCase()}">
            <div class="alert-info">
                <p>$${alert.amount.toLocaleString()} - ${alert.type.replace('_', ' ')}</p>
                <span>${new Date(alert.timestamp).toLocaleTimeString()} • ${alert.location}</span>
            </div>
            <div class="risk-tag ${alert.risk_level.toLowerCase()}">${alert.risk_level}</div>
            ${detailed ? `<button class="badge" style="background: var(--bg-surface); cursor: pointer;">Details</button>` : ''}
        </div>
    `).join('');
}

function showResult(result) {
    const container = document.getElementById('result-container');
    const isFraud = result.fraud === 1;
    const color = isFraud ? 'var(--danger)' : 'var(--success)';
    const statusText = isFraud ? '⚠ FRAUD DETECTED' : '✅ LEGITIMATE';
    
    container.style.display = 'block';
    container.style.border = `1px solid ${color}`;
    container.style.background = `rgba(${isFraud ? '239, 68, 68' : '16, 185, 129'}, 0.05)`;
    
    container.innerHTML = `
        <h4 style="color: ${color}; margin-bottom: 1rem;">${statusText}</h4>
        <div class="score-circle" style="border-color: ${color}">
            <span class="score-val">${(result.risk_score * 100).toFixed(1)}%</span>
            <span class="score-label">Risk Score</span>
        </div>
        <p style="font-size: 0.9rem;">Level: <strong style="color: ${color}">${result.risk_level}</strong></p>
        <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 1rem;">Analyzed by ${result.model} at threshold ${result.threshold}</p>
    `;
}

// ─────────────────────────────────────────────
// CHARTS LOGIC
// ─────────────────────────────────────────────

function initCharts() {
    const ctx = document.getElementById('riskTrendChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                datasets: [{
                    label: 'Risk Volatility',
                    data: [12, 19, 3, 5, 2, 3],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }

    const featureCtx = document.getElementById('featureChart');
    if (featureCtx) {
        new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: ['Hour', 'Freq', 'Night', 'Amount', 'AvgAmt'],
                datasets: [{
                    label: 'Importance',
                    data: [0.31, 0.20, 0.19, 0.10, 0.08],
                    backgroundColor: ['#3b82f6', '#0ea5e9', '#6366f1', '#8b5cf6', '#a855f7']
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { grid: { display: false } }
                }
            }
        });
    }
}

// ─────────────────────────────────────────────
// CORE LOGIC
// ─────────────────────────────────────────────

function getAuthHeader() {
    const token = localStorage.getItem('sentinel_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function fetchStats() {
    try {
        const response = await fetch('/api/dashboard/stats', { headers: getAuthHeader() });
        if (response.status === 401) window.location.href = 'login.html';
        state.stats = await response.json();
    } catch (e) { console.error("Stats error", e); }
}

async function fetchAlerts() {
    try {
        const response = await fetch('/api/dashboard/alerts', { headers: getAuthHeader() });
        if (response.status === 401) window.location.href = 'login.html';
        state.alerts = await response.json();
        const badge = document.getElementById('alert-badge');
        if (badge) badge.textContent = state.alerts.length;
    } catch (e) { console.error("Alerts error", e); }
}

function navigate(pageId) {
    state.activePage = pageId;
    const container = document.getElementById('page-container');
    const title = document.getElementById('page-title');
    
    // Update sidebar
    document.querySelectorAll('.sidebar-nav li').forEach(li => {
        li.classList.toggle('active', li.dataset.page === pageId);
    });

    // Update Content
    title.textContent = pageId.charAt(0).toUpperCase() + pageId.slice(1) + (pageId === 'dashboard' ? ' Intelligence' : '');
    container.innerHTML = Templates[pageId]();
    
    initCharts();
    
    // Setup form if analysis page
    if (pageId === 'analysis') {
        const form = document.getElementById('analysis-form');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('analyze-btn');
            btn.textContent = 'Processing...';
            btn.disabled = true;
            
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            // Coerce types
            data.amount = parseFloat(data.amount);
            data.transaction_hour = parseInt(data.transaction_hour);
            data.transaction_freq_7d = parseInt(data.transaction_freq_7d);
            data.avg_amount_7d = parseFloat(data.avg_amount_7d);
            data.transaction_day = parseInt(data.transaction_day);
            data.amount_deviation = parseFloat(data.amount_deviation);
            data.is_night = parseInt(data.is_night);
            data.is_weekend = parseInt(data.is_weekend);

            try {
                const res = await fetch('/api/predict/', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        ...getAuthHeader()
                    },
                    body: JSON.stringify(data)
                });
                if (res.status === 401) window.location.href = 'login.html';
                if (res.status === 429) {
                    alert("Rate Limit Exceeded: Please wait a moment before trying again.");
                    return;
                }
                const result = await res.json();
                showResult(result);
            } catch (err) {
                alert("API Error: " + err.message);
            } finally {
                btn.textContent = 'Analyze Transaction';
                btn.disabled = false;
            }
        });
    }

    // Setup Batch Processing
    if (pageId === 'batch') {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('batch-file');
        
        dropZone.onclick = () => fileInput.click();
        
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            dropZone.innerHTML = `<div class="loader"></div><p>Processing batch...</p>`;
            
            try {
                const res = await fetch('/api/predict/batch', {
                    method: 'POST',
                    headers: getAuthHeader(),
                    body: formData
                });
                if (res.status === 401) window.location.href = 'login.html';
                if (res.status === 429) {
                    alert("Rate Limit Exceeded: Batch processing is limited to 5 per minute.");
                    navigate('batch');
                    return;
                }
                const data = await res.json();
                
                document.getElementById('batch-results').style.display = 'block';
                const tbody = document.getElementById('batch-table-body');
                tbody.innerHTML = data.results.slice(0, 10).map((r, i) => `
                    <tr>
                        <td style="padding: 0.5rem;">${i+1}</td>
                        <td>$${r.amount?.toFixed(2) || 'N/A'}</td>
                        <td>${r.type || 'N/A'}</td>
                        <td style="color: ${r.fraud ? 'var(--danger)' : 'var(--success)'}">${(r.risk_score * 100).toFixed(2)}%</td>
                        <td>${r.fraud ? '🚩 Flagged' : '✅ Clear'}</td>
                    </tr>
                `).join('');
                
                dropZone.innerHTML = `<p style="color: var(--success);">✅ Batch Processed: ${data.total} records</p>`;
            } catch (err) {
                alert("Upload failed: " + err.message);
                navigate('batch');
            }
        };
    }
}

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', async () => {
    // Auth Check
    const token = localStorage.getItem('sentinel_token');
    if (!token && !window.location.pathname.includes('login.html') && !window.location.pathname.includes('landing.html')) {
        window.location.href = 'login.html';
        return;
    }

    // Update user info in sidebar
    const userEmail = localStorage.getItem('user_email');
    if (userEmail) {
        const nameEl = document.querySelector('.user-name');
        const roleEl = document.querySelector('.user-role');
        if (nameEl) nameEl.textContent = userEmail.split('@')[0];
        if (roleEl) roleEl.textContent = 'Authorized User';
    }

    // Initial Load
    await fetchStats();
    await fetchAlerts();
    navigate('dashboard');

    // Sidebar listeners
    document.querySelectorAll('.sidebar-nav li').forEach(li => {
        li.addEventListener('click', () => navigate(li.dataset.page));
    });

    // Robust Logout Listener (Event Delegation)
    document.body.addEventListener('click', (e) => {
        if (e.target.closest('.logout-btn')) {
            e.preventDefault();
            localStorage.removeItem('sentinel_token');
            localStorage.removeItem('user_email');
            window.location.href = 'landing.html';
        }
    });

    // Simulated real-time polling
    setInterval(async () => {
        await fetchAlerts();
        if (state.activePage === 'dashboard' || state.activePage === 'alerts') {
            const alertList = document.getElementById(state.activePage === 'dashboard' ? 'dashboard-alerts' : 'alert-list');
            if (alertList) alertList.innerHTML = renderAlertList(state.alerts, state.activePage === 'alerts');
        }
    }, 10000);
});
