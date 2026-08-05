let historyData = [];
let polling = true;

function showToast(message, type) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast ' + type + ' show';
  setTimeout(() => toast.classList.remove('show'), 3000);
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-dashboard').style.display = name === 'dashboard' ? 'block' : 'none';
  document.getElementById('tab-settings').style.display = name === 'settings' ? 'block' : 'none';
  document.querySelector(`.tab[onclick*="${name}"]`).classList.add('active');
}

async function fetchStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    document.getElementById('good-count').textContent = data.good;
    document.getElementById('ng-count').textContent = data.ng;
    document.getElementById('total-count').textContent = data.total;

    const fpsEl = document.getElementById('fps-count');
    const fpsDisp = document.getElementById('fps-display');
    fpsEl.textContent = data.fps;
    fpsDisp.textContent = data.fps + ' FPS';

    const rate = data.rate || 100;
    document.getElementById('rate-text').textContent = rate + '%';
    const bar = document.getElementById('rate-bar');
    bar.style.width = rate + '%';
    bar.className = 'bar-fill ' + (rate >= 90 ? 'good' : rate >= 70 ? 'medium' : 'bad');

    const conveyor = document.getElementById('conveyor-status');
    conveyor.textContent = data.conveyor === 'running' ? '🟢 RUNNING' : '🔴 STOPPED';
    conveyor.className = 'value ' + data.conveyor;
  } catch (e) {
    console.error('Status fetch error:', e);
  }
}

async function fetchHistory() {
  try {
    const res = await fetch('/history');
    const data = await res.json();
    if (JSON.stringify(data) === JSON.stringify(historyData)) return;
    historyData = data;
    renderHistory(data);
  } catch (e) {
    console.error('History fetch error:', e);
  }
}

function renderHistory(data) {
  const container = document.getElementById('history-body');
  document.getElementById('history-count').textContent = data.length + ' records';

  if (!data.length) {
    container.innerHTML = '<div class="empty-state">No detections yet. Start the system to begin.</div>';
    return;
  }

  let html = '<table class="history-table"><thead><tr><th>Time</th><th>ID</th><th>Result</th><th>Defects</th><th>Score</th></tr></thead><tbody>';
  data.slice().reverse().forEach(item => {
    const cls = item.verdict === 'GOOD' ? 'badge-good' : 'badge-ng';
    const defectsHtml = item.defects && item.defects.length
      ? item.defects.map(d => `<span class="badge badge-defect">${d}</span>`).join('')
      : '<span style="color:#64748b;font-size:12px;">-</span>';
    html += `<tr>
      <td>${item.timestamp}</td>
      <td>#${item.id}</td>
      <td><span class="badge ${cls}">${item.verdict}</span></td>
      <td>${defectsHtml}</td>
      <td style="color:#94a3b8;">${item.score}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

async function startSystem() {
  await fetch('/control/start', { method: 'POST' });
  showToast('System started', 'success');
  document.getElementById('video-stream').src = '/video?' + Date.now();
}

async function stopSystem() {
  await fetch('/control/stop', { method: 'POST' });
  showToast('System stopped', 'info');
}

async function resetCounters() {
  if (!confirm('Reset all counters and history?')) return;
  await fetch('/reset', { method: 'POST' });
  historyData = [];
  renderHistory([]);
  showToast('Counters reset', 'info');
}

async function updateThreshold(algo, value) {
  document.getElementById(algo + '-value').textContent = parseFloat(value).toFixed(2);
  await fetch('/control/threshold', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo, value: parseFloat(value) }),
  });
}

// Polling
setInterval(() => { if (polling) fetchStatus(); }, 1500);
setInterval(() => { if (polling) fetchHistory(); }, 2000);

fetchStatus();
fetchHistory();
