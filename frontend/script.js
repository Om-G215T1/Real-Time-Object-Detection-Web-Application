// frontend/script.js
// Handles all UI interactions for the YOLOv8 detection app

const API = 'http://localhost:5000/api';

// ─── Utility Helpers ─────────────────────────────────────

function setStatus(msg) {
  document.getElementById('status-message').textContent = msg;
}

function getConf() {
  return parseFloat(document.getElementById('conf-slider').value);
}

function getModel() {
  return document.getElementById('model-select').value;
}

function showImage(imgEl, src) {
  imgEl.src = src;
  imgEl.classList.add('active');
}

function hideImage(imgEl) {
  imgEl.src = '';
  imgEl.classList.remove('active');
}

function renderDetections(containerId, detections) {
  const container = document.getElementById(containerId);
  if (!detections || detections.length === 0) {
    container.innerHTML = '<p class="empty-msg">No objects detected</p>';
    return;
  }
  container.innerHTML = detections.map(d => `
    <div class="detection-item">
      <span class="label">${d.class}</span>
      <span class="conf">${(d.confidence * 100).toFixed(1)}%</span>
    </div>
  `).join('');
}

// ─── API Health Check ─────────────────────────────────────

async function checkAPIHealth() {
  const badge = document.getElementById('api-status');
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    badge.textContent = `API Online — ${data.device.toUpperCase()}`;
    badge.className = 'api-badge online';

    // Update webcam device stat
    document.getElementById('webcam-device').textContent =
      data.device.toUpperCase();
  } catch {
    badge.textContent = 'API Offline';
    badge.className = 'api-badge offline';
    setStatus('⚠️ Cannot connect to Flask server. Is it running?');
  }
}

// ─── Confidence Slider ────────────────────────────────────

const slider = document.getElementById('conf-slider');
slider.addEventListener('input', () => {
  document.getElementById('conf-value').textContent =
    parseFloat(slider.value).toFixed(2);
});

// ─── Model Switching ──────────────────────────────────────

document.getElementById('model-select').addEventListener('change', async () => {
  const model = getModel();
  setStatus(`Switching model to ${model}...`);
  try {
    const res = await fetch(`${API}/model/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model })
    });
    const data = await res.json();
    setStatus(`Model switched to ${data.model} ✅`);
  } catch {
    setStatus('⚠️ Failed to switch model');
  }
});

// ─── Tab Switching ────────────────────────────────────────

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    // Stop webcam if switching away
    if (!tab.dataset.tab !== 'webcam') {
      stopWebcam();
    }
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ─── Webcam Detection ─────────────────────────────────────

let webcamActive = false;

function startWebcam() {
  const conf = getConf();
  const streamImg = document.getElementById('webcam-stream');
  const placeholder = document.getElementById('webcam-placeholder');

  streamImg.src = `${API}/stream/webcam?conf=${conf}&t=${Date.now()}`;
  showImage(streamImg, streamImg.src);
  placeholder.style.display = 'none';

  document.getElementById('start-webcam').disabled = true;
  document.getElementById('stop-webcam').disabled = false;
  document.getElementById('webcam-status').textContent = 'Live';

  webcamActive = true;
  setStatus('📷 Webcam stream active...');
}

function stopWebcam() {
  if (!webcamActive) return;
  const streamImg = document.getElementById('webcam-stream');
  hideImage(streamImg);

  document.getElementById('webcam-placeholder').style.display = 'flex';
  document.getElementById('start-webcam').disabled = false;
  document.getElementById('stop-webcam').disabled = true;
  document.getElementById('webcam-status').textContent = 'Idle';

  webcamActive = false;
  setStatus('Webcam stopped.');
}

document.getElementById('start-webcam').addEventListener('click', startWebcam);
document.getElementById('stop-webcam').addEventListener('click', stopWebcam);

// ─── Image Detection ──────────────────────────────────────

const imageInput = document.getElementById('image-input');
const detectImageBtn = document.getElementById('detect-image-btn');

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;

  // Show preview
  const preview = document.getElementById('image-preview');
  const previewBox = document.getElementById('image-preview-box');
  preview.src = URL.createObjectURL(file);
  previewBox.classList.remove('hidden');
  detectImageBtn.disabled = false;
  setStatus(`Image selected: ${file.name}`);
});

detectImageBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) return;

  setStatus('🔍 Detecting objects...');
  detectImageBtn.disabled = true;
  detectImageBtn.textContent = '⏳ Detecting...';

  try {
    const form = new FormData();
    form.append('file', file);
    form.append('conf', getConf());

    const res = await fetch(`${API}/detect/image`, {
      method: 'POST',
      body: form
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();

    // Show result image
    const resultImg = document.getElementById('image-result');
    const placeholder = document.getElementById('image-placeholder');
    showImage(resultImg, `data:image/jpeg;base64,${data.image}`);
    placeholder.style.display = 'none';

    // Show detections
    renderDetections('image-detections', data.detections);
    document.getElementById('image-count').textContent = data.count;

    setStatus(`✅ Found ${data.count} object(s) — FPS: ${data.fps}`);

  } catch (err) {
    setStatus(`⚠️ Detection failed: ${err.message}`);
  } finally {
    detectImageBtn.disabled = false;
    detectImageBtn.textContent = '🔍 Detect Objects';
  }
});

// ─── Drag & Drop — Image ──────────────────────────────────

const imageDropZone = document.getElementById('image-drop-zone');

imageDropZone.addEventListener('dragover', e => {
  e.preventDefault();
  imageDropZone.classList.add('drag-over');
});

imageDropZone.addEventListener('dragleave', () => {
  imageDropZone.classList.remove('drag-over');
});

imageDropZone.addEventListener('drop', e => {
  e.preventDefault();
  imageDropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    // Simulate file input change
    const dt = new DataTransfer();
    dt.items.add(file);
    imageInput.files = dt.files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

// ─── Video Detection ──────────────────────────────────────

const videoInput = document.getElementById('video-input');
const detectVideoBtn = document.getElementById('detect-video-btn');

videoInput.addEventListener('change', () => {
  const file = videoInput.files[0];
  if (!file) return;
  document.getElementById('video-filename').textContent = file.name;
  document.getElementById('video-info').classList.remove('hidden');
  detectVideoBtn.disabled = false;
  setStatus(`Video selected: ${file.name}`);
});

detectVideoBtn.addEventListener('click', async () => {
  const file = videoInput.files[0];
  if (!file) return;

  setStatus('🎬 Uploading and processing video...');
  detectVideoBtn.disabled = true;
  detectVideoBtn.textContent = '⏳ Processing...';

  document.getElementById('video-status').textContent = 'Processing';

  try {
    const form = new FormData();
    form.append('file', file);
    form.append('conf', getConf());

    // Upload video — server streams back MJPEG
    const res = await fetch(`${API}/detect/video`, {
      method: 'POST',
      body: form
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    // The response is MJPEG — point <img> to the stream URL
    // We re-upload to get the stream going
    const videoStream = document.getElementById('video-stream');
    const placeholder = document.getElementById('video-placeholder');

    // Create object URL for the stream response
    const blob = await res.blob();
    showImage(videoStream, URL.createObjectURL(blob));
    placeholder.style.display = 'none';

    document.getElementById('video-status').textContent = 'Done';
    setStatus('✅ Video processing complete');

  } catch (err) {
    setStatus(`⚠️ Video processing failed: ${err.message}`);
    document.getElementById('video-status').textContent = 'Error';
  } finally {
    detectVideoBtn.disabled = false;
    detectVideoBtn.textContent = '▶ Process Video';
  }
});

// ─── Drag & Drop — Video ──────────────────────────────────

const videoDropZone = document.getElementById('video-drop-zone');

videoDropZone.addEventListener('dragover', e => {
  e.preventDefault();
  videoDropZone.classList.add('drag-over');
});

videoDropZone.addEventListener('dragleave', () => {
  videoDropZone.classList.remove('drag-over');
});

videoDropZone.addEventListener('drop', e => {
  e.preventDefault();
  videoDropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) {
    const dt = new DataTransfer();
    dt.items.add(file);
    videoInput.files = dt.files;
    videoInput.dispatchEvent(new Event('change'));
  }
});

// ─── Init ─────────────────────────────────────────────────

// Check API health on load and every 30 seconds
checkAPIHealth();
setInterval(checkAPIHealth, 30000);