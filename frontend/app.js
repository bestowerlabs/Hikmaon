const API = 'http://localhost:8000';
let state = { suspiciousMediaId: null };

function b64(text) {
  return btoa(unescape(encodeURIComponent(text)));
}

function out(data) {
  document.getElementById('output').textContent = JSON.stringify(data, null, 2);
}

async function registerMedia() {
  const payload = {
    owner_id: document.getElementById('ownerId').value,
    owner_public_key: document.getElementById('ownerKey').value,
    media_type: document.getElementById('mediaType').value,
    filename: document.getElementById('filename').value,
    content_b64: b64(document.getElementById('content').value),
    metadata: { source: 'dashboard' },
  };

  const res = await fetch(`${API}/api/registrations`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  out(await res.json());
}

async function analyzeMedia() {
  const payload = {
    media_type: document.getElementById('mediaType').value,
    filename: document.getElementById('susFilename').value,
    content_b64: b64(document.getElementById('susContent').value),
  };

  const res = await fetch(`${API}/api/analyze`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await res.json();
  state.suspiciousMediaId = data.suspicious_media_id;
  out(data);
}

async function verifyOwnership() {
  const res = await fetch(`${API}/api/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ suspicious_media_id: state.suspiciousMediaId }),
  });
  out(await res.json());
}

async function generateEvidence() {
  const res = await fetch(`${API}/api/evidence/${state.suspiciousMediaId}`, { method: 'POST' });
  out(await res.json());
}

async function sendNotification() {
  const res = await fetch(`${API}/api/notifications`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      channel: 'dashboard',
      recipient: 'security-team',
      message: `High-priority media alert for ${state.suspiciousMediaId}`,
    }),
  });
  out(await res.json());
}
