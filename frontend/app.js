const API = 'http://localhost:8000';

function b64(text) {
  return btoa(unescape(encodeURIComponent(text)));
}

function output(data) {
  document.getElementById('output').textContent = JSON.stringify(data, null, 2);
}

async function request(path, payload, method = 'POST') {
  const response = await fetch(`${API}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: payload ? JSON.stringify(payload) : undefined,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(data));
  }
  return data;
}

async function connectAccount() {
  try {
    const data = await request('/api/connectors', {
      owner_id: document.getElementById('ownerId').value,
      owner_public_key: document.getElementById('ownerPub').value,
      provider: document.getElementById('provider').value,
      account_handle: document.getElementById('handle').value,
    });
    document.getElementById('connectorId').value = data.connector_id;
    output({ stage: 'account_connected', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function loadConnectors() {
  try {
    const res = await fetch(`${API}/api/connectors`);
    output(await res.json());
  } catch (err) {
    output({ error: err.message });
  }
}

async function ingestEvent() {
  try {
    const data = await request('/api/connectors/ingest', {
      connector_id: document.getElementById('connectorId').value,
      media_type: document.getElementById('mediaType').value,
      filename: document.getElementById('filename').value,
      content_b64: b64(document.getElementById('mediaText').value),
      source_url: document.getElementById('sourceUrl').value,
    });
    output({ stage: 'ingest_and_registration', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function runDetectionCycle() {
  try {
    const data = await request('/api/realtime/detect', {
      media_type: document.getElementById('mediaType').value,
      filename: document.getElementById('susFile').value,
      content_b64: b64(document.getElementById('susText').value),
    });
    output({ stage: 'incident_detection_cycle', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function listIncidents() {
  try {
    const res = await fetch(`${API}/api/incidents`);
    output(await res.json());
  } catch (err) {
    output({ error: err.message });
  }
}
