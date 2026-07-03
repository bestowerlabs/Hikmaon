const API = 'http://localhost:8000';

let lastRegisteredMediaId = null;

// ------------------------------------------------------------------ auth state
function tokens() {
  return {
    access: localStorage.getItem('hikmaon_access'),
    refresh: localStorage.getItem('hikmaon_refresh'),
    user: JSON.parse(localStorage.getItem('hikmaon_user') || 'null'),
  };
}

function saveTokens(data) {
  localStorage.setItem('hikmaon_access', data.access_token);
  localStorage.setItem('hikmaon_refresh', data.refresh_token);
  localStorage.setItem('hikmaon_user', JSON.stringify(data.user));
  updateAuthUI();
}

function clearTokens() {
  localStorage.removeItem('hikmaon_access');
  localStorage.removeItem('hikmaon_refresh');
  localStorage.removeItem('hikmaon_user');
  updateAuthUI();
}

function updateAuthUI() {
  const { user } = tokens();
  document.getElementById('authForms').style.display = user ? 'none' : 'block';
  document.getElementById('authSession').style.display = user ? 'block' : 'none';
  if (user) {
    document.getElementById('whoami').textContent = `${user.display_name} <${user.email}>`;
    document.getElementById('ownerKey').textContent = user.owner_public_key.slice(0, 24) + '…';
  }
}

function output(data) {
  document.getElementById('output').textContent = JSON.stringify(data, null, 2);
}

async function request(path, payload, method = 'POST', retried = false) {
  const headers = { 'Content-Type': 'application/json' };
  const { access } = tokens();
  if (access) headers['Authorization'] = `Bearer ${access}`;
  const response = await fetch(`${API}${path}`, {
    method,
    headers,
    body: payload ? JSON.stringify(payload) : undefined,
  });
  if (response.status === 401 && !retried && tokens().refresh) {
    const refreshed = await fetch(`${API}/api/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: tokens().refresh }),
    });
    if (refreshed.ok) {
      saveTokens(await refreshed.json());
      return request(path, payload, method, true);
    }
    clearTokens();
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

const get = (path) => request(path, null, 'GET');

// ---------------------------------------------------------------- auth actions
async function authRegister() {
  try {
    const data = await request('/api/auth/register', {
      email: document.getElementById('authEmail').value,
      password: document.getElementById('authPassword').value,
      display_name: document.getElementById('authName').value,
    });
    saveTokens(data);
    output({ stage: 'account_created', user: data.user });
    refreshAll();
  } catch (err) {
    output({ error: err.message });
  }
}

async function authLogin() {
  try {
    const data = await request('/api/auth/login', {
      email: document.getElementById('authEmail').value,
      password: document.getElementById('authPassword').value,
    });
    saveTokens(data);
    output({ stage: 'logged_in', user: data.user });
    refreshAll();
  } catch (err) {
    output({ error: err.message });
  }
}

async function authLogout() {
  const { refresh } = tokens();
  try {
    if (refresh) await request('/api/auth/logout', { refresh_token: refresh });
  } finally {
    clearTokens();
    output({ stage: 'logged_out' });
  }
}

// ------------------------------------------------------------------- statusbar
async function loadStatusBar() {
  try {
    const health = await (await fetch(`${API}/health`)).json();
    document.getElementById('chipChain').textContent = `chain: ${health.chain_mode}`;
    const model = await (await fetch(`${API}/api/model/status`)).json();
    document.getElementById('chipModel').textContent = `neural model: ${model.neural_detector}`;
    const providers = await (await fetch(`${API}/api/integrations/status`)).json();
    const configured = providers.filter((p) => p.configured).length;
    document.getElementById('chipProviders').textContent =
      `integrations: ${configured}/${providers.length} configured`;
  } catch (err) {
    document.getElementById('chipChain').textContent = 'backend offline';
  }
}

function refreshAll() {
  loadStatusBar();
  if (tokens().user) {
    listIncidents();
    listTakedowns();
    listCrawlJobs();
  }
}
window.addEventListener('load', () => { updateAuthUI(); refreshAll(); });

// ------------------------------------------------------------------ connectors
function fileToB64(inputId) {
  return new Promise((resolve, reject) => {
    const input = document.getElementById(inputId);
    const file = input.files && input.files[0];
    if (!file) { reject(new Error(`Choose a file in "${inputId}" first`)); return; }
    const reader = new FileReader();
    reader.onload = () => resolve({ b64: reader.result.split(',')[1], name: file.name });
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function oauthStart() {
  try {
    const provider = document.getElementById('provider').value;
    const data = await get(`/api/connectors/oauth/${provider}/start`);
    output({ stage: 'oauth_authorization_url', ...data });
    window.open(data.authorization_url, '_blank');
  } catch (err) {
    output({ error: err.message });
  }
}

async function connectAccount() {
  try {
    const data = await request('/api/connectors', {
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
  try { output(await get('/api/connectors')); } catch (err) { output({ error: err.message }); }
}

async function syncConnector() {
  try {
    const connectorId = document.getElementById('connectorId').value.trim();
    if (!connectorId) throw new Error('Enter a connector ID first');
    output({ stage: 'media_sync', ...(await request(`/api/connectors/${connectorId}/sync`, null)) });
  } catch (err) {
    output({ error: err.message });
  }
}

// ---------------------------------------------------------------- registration
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
    } else {
      data = await request('/api/registrations', {
        media_type: document.getElementById('mediaType').value,
        filename: name,
        content_b64: b64,
      });
    }
    lastRegisteredMediaId = data.media_id;
    output({ stage: 'registered_and_anchored', ...data });
  } catch (err) {
    output({ error: err.message });
  }
}

async function showCertificate() {
  try {
    if (!lastRegisteredMediaId) throw new Error('Register media first');
    const cert = await get(`/api/certificates/${lastRegisteredMediaId}`);
    const verification = await request('/api/certificates/verify', { certificate: cert });
    output({ certificate: cert, verification });
  } catch (err) {
    output({ error: err.message });
  }
}

// ------------------------------------------------------------------- detection
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

// --------------------------------------------------------------------- crawler
async function startCrawl() {
  try {
    const seeds = document.getElementById('crawlSeeds').value
      .split(',').map((s) => s.trim()).filter(Boolean);
    const data = await request('/api/crawler/jobs', {
      seed_urls: seeds,
      max_pages: parseInt(document.getElementById('crawlMaxPages').value || '50', 10),
      max_depth: 2,
    });
    output({ stage: 'crawl_started', ...data });
    listCrawlJobs();
  } catch (err) {
    output({ error: err.message });
  }
}

async function listCrawlJobs() {
  try {
    const jobs = await get('/api/crawler/jobs');
    const container = document.getElementById('crawlJobs');
    container.innerHTML = jobs.length ? '' : '<p class="muted">No crawl jobs.</p>';
    for (const job of jobs.reverse()) {
      const div = document.createElement('div');
      div.className = 'incident';
      div.innerHTML = `
        <span class="status ${job.status === 'completed' ? 'closed' : ''}">${job.status}</span>
        <div><strong>${job.job_id}</strong></div>
        <div class="muted">${job.seed_urls.join(', ')}</div>
        <div class="muted">pages ${job.pages_crawled} · media ${job.media_indexed} · matches ${job.matches_found}</div>`;
      container.appendChild(div);
    }
  } catch (err) {
    output({ error: err.message });
  }
}

// ----------------------------------------------------------- incidents & cases
async function listIncidents() {
  try {
    const incidents = await get('/api/incidents');
    const container = document.getElementById('incidents');
    container.innerHTML = incidents.length ? '' : '<p class="muted">No incidents.</p>';
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
    const cases = await get('/api/takedowns');
    const container = document.getElementById('takedowns');
    container.innerHTML = cases.length ? '' : '<p class="muted">No takedown cases.</p>';
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
