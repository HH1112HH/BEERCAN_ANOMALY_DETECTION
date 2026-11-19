let historyData = [];
let polling = true;
let sliderInitialized = false;
let statusInFlight = false;
let historyInFlight = false;

function showToast(message, type) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = message;
  toast.className = 'toast ' + type + ' show';
  setTimeout(() => toast.classList.remove('show'), 3000);
}

function setConveyorStateUI(isConvRunning) {
  const convStatusEl = document.getElementById('conveyor-status');
  const btnConvStart = document.getElementById('btn-conveyor-start');
  const btnConvStop = document.getElementById('btn-conveyor-stop');

  if (convStatusEl) {
    convStatusEl.textContent = isConvRunning ? 'RUNNING' : 'STOPPED';
    convStatusEl.className = 'status-pill ' + (isConvRunning ? 'running' : 'stopped');
  }
  if (btnConvStart && btnConvStop) {
    if (isConvRunning) {
      btnConvStart.className = 'btn btn-solid-success';
      btnConvStop.className = 'btn btn-outline-danger';
    } else {
      btnConvStart.className = 'btn btn-outline-success';
      btnConvStop.className = 'btn btn-solid-danger';
    }
  }
}

function setAIStateUI(isAiActive) {
  const aiStatusEl = document.getElementById('ai-status');
  const btnAiStart = document.getElementById('btn-ai-start');
  const btnAiStop = document.getElementById('btn-ai-stop');
  const aiModeBadge = document.getElementById('ai-mode-badge');

  if (aiStatusEl) {
    aiStatusEl.textContent = isAiActive ? 'ACTIVE' : 'IDLE';
    aiStatusEl.className = 'status-pill ' + (isAiActive ? 'active' : 'idle');
  }
  if (btnAiStart && btnAiStop) {
    if (isAiActive) {
      btnAiStart.className = 'btn btn-solid-primary';
      btnAiStop.className = 'btn btn-outline-warning';
    } else {
      btnAiStart.className = 'btn btn-outline-primary';
      btnAiStop.className = 'btn btn-solid-warning';
    }
  }
  if (aiModeBadge) {
    aiModeBadge.textContent = isAiActive ? 'AI Inspection Active' : 'Live Preview Only (AI Idle)';
    aiModeBadge.className = 'mode-badge ' + (isAiActive ? 'active' : 'idle');
  }
}

async function fetchStatus() {
  if (statusInFlight) return;
  statusInFlight = true;
  try {
    const res = await fetch('/status');
    const data = await res.json();

    // Session Info
    const sessNameEl = document.getElementById('session-name');
    const sessTargetEl = document.getElementById('session-target');
    const sessCountEl = document.getElementById('session-count');
    if (sessNameEl && data.session_name) sessNameEl.textContent = data.session_name;
    if (sessTargetEl && data.target_count) sessTargetEl.textContent = data.target_count;
    if (sessCountEl) sessCountEl.textContent = data.total;

    // Stats
    const goodEl = document.getElementById('good-count');
    const ngEl = document.getElementById('ng-count');
    const totEl = document.getElementById('total-count');
    if (goodEl) goodEl.textContent = data.good;
    if (ngEl) ngEl.textContent = data.ng;
    if (totEl) totEl.textContent = data.total;

    // FPS
    const fpsEl = document.getElementById('fps-count');
    const fpsDisp = document.getElementById('fps-display');
    if (fpsEl) fpsEl.textContent = data.fps;
    if (fpsDisp) fpsDisp.textContent = data.fps + ' FPS';

    // Rate
    const rate = data.rate !== undefined ? data.rate : 100;
    const rateTxt = document.getElementById('rate-text');
    const bar = document.getElementById('rate-bar');
    if (rateTxt) rateTxt.textContent = rate + '%';
    if (bar) {
      bar.style.width = rate + '%';
      bar.className = 'bar-fill ' + (rate >= 90 ? 'good' : rate >= 70 ? 'medium' : 'bad');
    }

    // 1. Conveyor State & Buttons
    setConveyorStateUI(data.conveyor === 'running');

    // 2. AI Inspection State & Buttons & Video Badge
    setAIStateUI(!!data.processing);

    // 3. Threshold sliders init
    if (!sliderInitialized && data.draem_threshold !== undefined) {
      const draemSlider = document.getElementById('draem-slider');
      const draemVal = document.getElementById('draem-value');
      if (draemSlider && draemVal) {
        draemSlider.value = data.draem_threshold;
        draemVal.textContent = parseFloat(data.draem_threshold).toFixed(2);
      }
      const confSlider = document.getElementById('conf-slider');
      const confVal = document.getElementById('conf-value');
      if (confSlider && confVal && data.conf_threshold !== undefined) {
        confSlider.value = data.conf_threshold;
        confVal.textContent = parseFloat(data.conf_threshold).toFixed(2);
      }
      sliderInitialized = true;
    }
  } catch (e) {
    console.error('Status fetch error:', e);
  } finally {
    statusInFlight = false;
  }
}

async function fetchHistory() {
  if (historyInFlight) return;
  historyInFlight = true;
  try {
    const res = await fetch('/history');
    const data = await res.json();
    if (JSON.stringify(data) === JSON.stringify(historyData)) return;
    historyData = data;
    renderHistory(data);
  } catch (e) {
    console.error('History fetch error:', e);
  } finally {
    historyInFlight = false;
  }
}

function renderHistory(data) {
  const container = document.getElementById('history-body');
  const countEl = document.getElementById('history-count');
  if (countEl) countEl.textContent = data.length + ' records';
  if (!container) return;

  if (!data.length) {
    container.innerHTML = '<div class="empty-state">No detections yet. Start the system to begin.</div>';
    return;
  }

  let html = '<table class="history-table"><thead><tr><th>Time</th><th>ID</th><th>Result</th><th>Defects</th><th>Score</th><th>Ảnh Phân Tích</th></tr></thead><tbody>';
  data.slice().reverse().forEach(item => {
    const cls = item.verdict === 'GOOD' ? 'badge-good' : 'badge-ng';
    const defectsHtml = item.defects && item.defects.length
      ? item.defects.map(d => `<span class="badge badge-defect">${d}</span>`).join('')
      : '<span style="color:#64748b;font-size:12px;">-</span>';
    
    const imgBtn = item.image_file 
      ? `<button class="btn-view-card" type="button" onclick="openImageModal('/captures/${item.image_file}', '${item.id}', '${item.verdict}')">Xem Ảnh</button>`
      : `<span style="color:#94a3b8;font-size:11px;">Chưa có ảnh</span>`;

    html += `<tr>
      <td>${item.timestamp}</td>
      <td><strong>#${item.id}</strong></td>
      <td><span class="badge ${cls}">${item.verdict}</span></td>
      <td>${defectsHtml}</td>
      <td style="color:#64748b;font-family:monospace;">${item.score}</td>
      <td>${imgBtn}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function openImageModal(imgUrl, canId, verdict) {
  const modal = document.getElementById('image-modal');
  const modalImg = document.getElementById('modal-img');
  const modalTitle = document.getElementById('modal-title');
  const downloadBtn = document.getElementById('modal-download-btn');
  if (!modal || !modalImg) return;
  modalImg.src = imgUrl;
  if (modalTitle) modalTitle.textContent = `Chi Tiết Phân Tích Khoa Học Lon #${canId} (${verdict})`;
  if (downloadBtn) {
    downloadBtn.href = imgUrl;
    downloadBtn.setAttribute('download', `Lon_${canId}_${verdict}.jpg`);
  }
  modal.classList.add('show');
}

function closeImageModal() {
  const modal = document.getElementById('image-modal');
  if (modal) modal.classList.remove('show');
}

// Conveyor Control Actions
async function startConveyor() {
  setConveyorStateUI(true);
  try {
    const res = await fetch('/conveyor/start', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      showToast('Đã khởi động băng tải', 'success');
    } else {
      showToast(data.message || 'Không thể khởi động băng tải', 'error');
    }
    fetchStatus();
  } catch (e) {
    showToast('Lỗi kết nối băng tải', 'error');
  }
}

async function stopConveyor() {
  setConveyorStateUI(false);
  try {
    const res = await fetch('/conveyor/stop', { method: 'POST' });
    const data = await res.json();
    showToast('Đã dừng băng tải', 'info');
    fetchStatus();
  } catch (e) {
    showToast('Lỗi kết nối băng tải', 'error');
  }
}

// AI Inspection Control Actions
async function startAI() {
  setAIStateUI(true);
  try {
    await fetch('/control/start', { method: 'POST' });
    showToast('Đã bật phân tích AI & servo gạt', 'success');
    fetchStatus();
  } catch (e) {
    showToast('Lỗi bật phân tích AI', 'error');
  }
}

async function stopAI() {
  setAIStateUI(false);
  try {
    await fetch('/control/stop', { method: 'POST' });
    showToast('Đã dừng phân tích AI (Chế độ xem camera an toàn)', 'warning');
    fetchStatus();
  } catch (e) {
    showToast('Lỗi dừng phân tích AI', 'error');
  }
}

// Legacy Aliases
window.startSystem = startAI;
window.stopSystem = stopAI;
window.startConveyor = startConveyor;
window.stopConveyor = stopConveyor;
window.startAI = startAI;
window.stopAI = stopAI;

// Delete Last Record (Undo)
async function deleteLastRecord() {
  if (!confirm('Bạn có chắc muốn xóa bản ghi lon vừa rồi (do kích hoạt nhầm)?')) return;
  try {
    const res = await fetch('/record/delete_last', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      showToast(`Đã xóa lon #${data.deleted.can_number} (${data.deleted.verdict})`, 'info');
      fetchHistory();
      fetchStatus();
    } else {
      showToast(data.message || 'Không có lon nào để xóa', 'error');
    }
  } catch (e) {
    showToast('Lỗi xóa bản ghi', 'error');
  }
}
window.deleteLastRecord = deleteLastRecord;

// Create New Test Session
async function createNewSession() {
  const targetStr = prompt('Nhập số lượng lon mục tiêu cần test cho đợt mới (Ví dụ: 10, 50, 100):', '10');
  if (!targetStr) return;
  const target = parseInt(targetStr, 10) || 10;
  const name = prompt('Nhập tên đợt test (hoặc để trống để tự tạo tên):', `Đợt test ${target} lon`);
  
  try {
    const res = await fetch('/session/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name || undefined, target_count: target }),
    });
    const data = await res.json();
    historyData = [];
    renderHistory([]);
    showToast(`Đã tạo: ${data.session.name} (Bắt đầu từ lon #1)`, 'success');
    fetchStatus();
  } catch (e) {
    showToast('Lỗi tạo đợt test mới', 'error');
  }
}
window.createNewSession = createNewSession;

// Reset Counters
async function resetCounters() {
  if (!confirm('Đặt lại tất cả bộ đếm và lịch sử?')) return;
  await fetch('/reset', { method: 'POST' });
  historyData = [];
  renderHistory([]);
  showToast('Đã đặt lại bộ đếm', 'info');
  fetchStatus();
}
window.resetCounters = resetCounters;

// Sliders Threshold Update
async function updateThreshold(algo, value) {
  const valEl = document.getElementById(algo + '-value');
  if (valEl) {
    valEl.textContent = parseFloat(value).toFixed(2);
  }
  await fetch('/control/threshold', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo, value: parseFloat(value) }),
  });
}
window.updateThreshold = updateThreshold;

// Polling (paused while the tab is hidden and never overlapping)
setInterval(() => { if (polling && !document.hidden) fetchStatus(); }, 1500);
setInterval(() => { if (polling && !document.hidden) fetchHistory(); }, 5000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    fetchStatus();
    fetchHistory();
  }
});

fetchStatus();
fetchHistory();
