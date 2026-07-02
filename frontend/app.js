const API = 'http://localhost:8000';

let lastRegisteredMediaId = null;

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

function fileToB64(inputId) {
  return new Promise((resolve, reject) => {
    const input = document.getElementById(inputId);
    const file = input.files && input.files[0];
    if (!file) {
      reject(new Error(`Choose a file in "${inputId}" first`));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve({ b64: reader.result.split(',')[1], name: file.name });
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
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

async function registerMedia() {
  try {
    const { b64, name } = await fileToB64('registerFile');
    const connectorId = document.getElementById('connectorId').value.trim();
    let data;
    if (connectorId) {
      data = await request('/api/connectors/ingest', {
        connector_id: connectorId,
        media_type: document.getElementById('mediaType').value,
        filename: name,
        content_b64: b64,
        source_url: document.getElementById('sourceUrl').value,
      });
      lastRegisteredMediaId = data.media_id;
    } else {
      data = await request('/api/registrations', {
        owner_id: document.getElementById('ownerId').value,
        owner_public_key: document.getElementById('ownerPub').value,
        media_type: document.getElementById('mediaType').value,
        filename: name,
        content_b64: b64,
      });
      lastRegisteredMediaId = data.media_id;
    }
    output({ stage: 'registered_and_anchored', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function showCertificate() {
  try {
    if (!lastRegisteredMediaId) throw new Error('Register media first');
    const res = await fetch(`${API}/api/certificates/${lastRegisteredMediaId}`);
    const cert = await res.json();
    const verification = await request('/api/certificates/verify', { certificate: cert });
    output({ certificate: cert, verification });
  } catch (err) {
    output({ error: err.message });
  }
}

async function runDetectionCycle() {
  try {
    const { b64, name } = await fileToB64('suspiciousFile');
    const data = await request('/api/realtime/detect', {
      media_type: document.getElementById('mediaType').value,
      filename: name,
      content_b64: b64,
    });
    output({ stage: 'detection_cycle', ...data });
    listIncidents();
  } catch (err) {
    output({ error: err.message });
  }
}

async function analyzeOnly() {
  try {
    const { b64, name } = await fileToB64('suspiciousFile');
    const data = await request('/api/analyze', {
      media_type: document.getElementById('mediaType').value,
      filename: name,
      content_b64: b64,
    });
    output({ stage: 'analysis', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function indexPublicCopy() {
  try {
    const { b64 } = await fileToB64('suspiciousFile');
    const data = await request('/api/monitor/index', {
      media_url: document.getElementById('publicUrl').value,
      content_b64: b64,
    });
    output({ stage: 'public_sighting_indexed', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function listIncidents() {
  try {
    const res = await fetch(`${API}/api/incidents`);
    const incidents = await res.json();
    const container = document.getElementById('incidents');
    container.innerHTML = '';
    if (!incidents.length) {
      container.innerHTML = '<p class="muted">No incidents.</p>';
      return;
    }
    for (const incident of incidents.reverse()) {
      const div = document.createElement('div');
      div.className = 'incident';
      const canDecide = incident.status === 'pending_owner_review';
      div.innerHTML = `
        <span class="status ${incident.status}">${incident.status}</span>
        <div class="pct">${incident.match_percentage}% match</div>
        <div class="muted">Incident ${incident.incident_id} · forensics: ${incident.manipulation_verdict}
          · chain verified: ${incident.blockchain_verified}</div>
        <div class="muted">URLs: ${incident.matched_urls.join(', ') || '—'}</div>
        ${canDecide ? `<div class="row">
          <button class="allow" onclick="decide('${incident.incident_id}','allow')">Allow</button>
          <button class="remove" onclick="decide('${incident.incident_id}','remove')">Remove</button>
        </div>` : ''}`;
      container.appendChild(div);
    }
  } catch (err) {
    output({ error: err.message });
  }
}

async function decide(incidentId, decision) {
  try {
    const data = await request(`/api/incidents/${incidentId}/decision`, { decision });
    output({ stage: 'owner_decision', ...data });
    listIncidents();
    listTakedowns();
  } catch (err) {
    output({ error: err.message });
  }
}

async function listTakedowns() {
  try {
    const res = await fetch(`${API}/api/takedowns`);
    const cases = await res.json();
    const container = document.getElementById('takedowns');
    container.innerHTML = '';
    if (!cases.length) {
      container.innerHTML = '<p class="muted">No takedown cases.</p>';
      return;
    }
    for (const item of cases.reverse()) {
      const div = document.createElement('div');
      div.className = 'incident';
      div.innerHTML = `
        <span class="status ${item.status}">${item.status}</span>
        <div><strong>${item.case_id}</strong></div>
        <div class="muted">Targets: ${item.target_urls.join(', ')}</div>`;
      const noticeButton = document.createElement('button');
      noticeButton.className = 'secondary';
      noticeButton.textContent = 'View DMCA Notice';
      noticeButton.onclick = () => output(item);
      div.appendChild(noticeButton);
      container.appendChild(div);
    }
  } catch (err) {
    output({ error: err.message });
  }
}
