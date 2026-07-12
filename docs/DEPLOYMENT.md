# Hikmaon Deployment Guide

Hikmaon by Bestower Labs — AI-based deepfake detection and prevention using a
hybrid blockchain network (Hikmalayer).

This guide covers: running the API, deploying the trained neural detector,
connecting Hikmalayer, activating platform integrations, and the production
checklist.

---

## 1. API server

### Requirements
- Python 3.11+
- `pip install -r backend/requirements.txt`
- (optional, for the neural detector) `pip install onnxruntime` — already in requirements

### Run (development)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload            # http://localhost:8000
```
Open `frontend/index.html` in a browser.

### Run (production)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```
Put nginx/Caddy in front for TLS. Note: each worker keeps its own in-process
state snapshot; run a single worker until the Postgres storage backend lands,
or pin state-mutating routes to one instance.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HIKMAON_JWT_SECRET` | auto-generated | HMAC secret for access tokens. **Set from a secrets manager in production** |
| `HIKMAON_BOOTSTRAP_ADMIN` | *(unset)* | Set to `1` only while creating the first admin account, then unset. Without it, all accounts are scoped `owner` (no global visibility) — safe default for public deployments |
| `HIKMAON_FFMPEG` | auto-detected | ffmpeg binary path (falls back to PATH, then the `imageio-ffmpeg` bundled binary) for video/audio fingerprinting |
| `HIKMAON_CRAWLER_SEEDS` | *(unset)* | Comma-separated seed URLs for the autonomous crawl schedule |
| `HIKMAON_CRAWLER_INTERVAL_MINUTES` | `0` (off) | Re-crawl the seeds on this interval |
| `HIKMAON_DATA_DIR` | `data` | Persistence directory (state snapshot + signing key + auth secret) |
| `HIKMAON_SIGNING_KEY` | auto-generated | Hex seed of the Ed25519 certificate-issuing key. **Set explicitly in production and back it up** — certificates verify against this key |
| `HIKMALAYER_RPC_URL` | *(unset)* | Real Hikmalayer node URL. Unset = local dev ledger, labelled `dev-simulated` |
| `HIKMAON_MODEL_PATH` | *(unset)* | Path to the exported `hikmaonnet.onnx`. Unset = heuristics only. When set, the detector scores still images (with horizontal-flip test-time augmentation) **and video** (frames sampled, clip score = 80th percentile of frame scores) |
| `HIKMAON_MATCH_THRESHOLD` | `55` | Confirmed-match % (opens incident + alerts owner) |
| `HIKMAON_REVIEW_THRESHOLD` | `35` | Possible-match % (queued for review, no alert) |
| `HIKMAON_REQUIRE_OWNERSHIP_PROOF` | `0` | `1` = registrations must include an Ed25519 signature over the content hash |
| `HIKMAON_TOKEN_KEY` | derived | Fernet key for the OAuth token vault. **Set explicitly in production** |
| `HIKMAON_OAUTH_REDIRECT_BASE` | `http://localhost:8000` | Public base URL for OAuth callbacks |
| `HIKMAON_WEBHOOK_VERIFY_TOKEN` | *(unset)* | Token echoed in Meta-style webhook subscription handshakes |
| `HIKMAON_WEBHOOK_SHARED_SECRET` | *(unset)* | Shared secret for `X-Hikmaon-Webhook-Secret` deliveries |
| `HIKMAON_CORS_ORIGINS` | `*` | Comma-separated allowed origins — restrict in production |
| `HIKMAON_<PROVIDER>_CLIENT_ID/_CLIENT_SECRET` | *(unset)* | Per-platform OAuth app credentials (see §4) |

---

## 2. Neural detector (HikmaonNet) — training team workflow

The model lives in `backend/ml/`. Training happens on your GPU machines;
the API server only ever loads the exported ONNX file.

### 2.1 Prepare data
Build face-crop frame datasets from FaceForensics++ (c23 **and** c40), DFDC,
Celeb-DF v2, DeeperForensics, plus current diffusion-generated sets, then
generate the manifest with the bundled tool:

```bash
python -m ml.make_manifest \
    --real /data/ffpp/real --real /data/celebdf/real \
    --fake deepfakes=/data/ffpp/deepfakes \
    --fake face2face=/data/ffpp/face2face \
    --fake neuraltextures=/data/ffpp/neuraltextures \
    --fake celebdf=/data/celebdf/fake \
    --holdout celebdf \
    --val 0.1 --test 0.1 \
    --out /data/manifest.csv
```

The tool guarantees frames from one source video never span two splits (no
leakage), keeps splits deterministic as the dataset grows, and prints a
per-generator split summary. Resulting CSV format (`ml/data.py`):

```csv
path,label,generator,split
/data/ffpp/real/000/f001.png,0,real,train
/data/ffpp/df/000/f001.png,1,deepfakes,train
/data/celebdf/fake/x/f01.png,1,celebdf,val
```

**Hold at least one generator out of train entirely** (`--holdout`) — its
val/test AUC is your generalization number, and the one to gate releases on.

### 2.2 Train
```bash
cd backend
pip install -r ml/requirements.txt
python -m ml.train --manifest /data/manifest.csv --out runs/v1 \
    --epochs 30 --batch-size 64 --lr 3e-4
```
Checkpoints select on validation AUC; metrics stream to `runs/v1/log.jsonl`.

### 2.3 Evaluate + calibrate
```bash
python -m ml.evaluate --manifest /data/manifest.csv \
    --checkpoint runs/v1/best.pt --split test --fit-temperature
```
Reports overall AUC/EER, **per-generator AUC**, and calibration error;
`--fit-temperature` writes the fitted temperature into the checkpoint so the
served probability is calibrated.

### 2.4 Export + deploy
```bash
python -m ml.export --checkpoint runs/v1/best.pt --out hikmaonnet.onnx
# copy hikmaonnet.onnx to the API host, then:
HIKMAON_MODEL_PATH=/models/hikmaonnet.onnx uvicorn app.main:app ...
```
Check `GET /api/model/status` → `"neural_detector": "loaded"`. The model's
calibrated probability becomes the dominant manipulation signal, fused with
the forensic heuristics (`forensic-heuristics-v1+hikmaonnet-v1`).

### 2.5 Keep it current
Generators evolve monthly. Re-benchmark on new generators quarterly at
minimum, retrain when a held-out generator drops below your bar, and version
every deployment (the model version is embedded in every evidence report).

---

## 3. Hikmalayer connection

Hikmalayer is a separate Bestower Labs project. Hikmaon is a client only:

```bash
HIKMALAYER_RPC_URL=https://node.hikmalayer.example uvicorn app.main:app ...
```

Expected node API (adjust `TX_SUBMIT_PATH` / `TX_QUERY_PATH` in
`app/hikmalayer.py` to the node's real routes):

- `POST /transactions` with the `MEDIA_REGISTRATION` payload → `{"txid": ...}`
- `GET /transactions/{txid}` → the stored transaction

Submissions retry with backoff (1s/2s/4s). Every verification response and
certificate carries `chain_mode` (`rpc` vs `dev-simulated`) so simulated
proofs can never be mistaken for anchored ones.

---

## 4. Platform integrations (Step 1 of the product)

Activate each platform by registering an OAuth app in its developer console
and exporting credentials. Check `GET /api/integrations/status` for the exact
env var names and console links per provider.

```bash
export HIKMAON_OAUTH_REDIRECT_BASE=https://app.hikmaon.com
export HIKMAON_INSTAGRAM_CLIENT_ID=...     HIKMAON_INSTAGRAM_CLIENT_SECRET=...
export HIKMAON_GOOGLE_DRIVE_CLIENT_ID=...  HIKMAON_GOOGLE_DRIVE_CLIENT_SECRET=...
# ... per provider
```

Flow once configured:
1. `GET /api/connectors/oauth/{provider}/start?owner_id=..&owner_public_key=..`
   → send the user to `authorization_url` (PKCE included automatically).
2. Platform redirects to `/api/connectors/oauth/{provider}/callback` —
   tokens are exchanged and stored **Fernet-encrypted**.
3. `POST /api/connectors/{id}/sync` pulls recent uploads through the
   provider's media API and auto-registers them (implemented: Instagram,
   Facebook, Google Drive, Dropbox, OneDrive, X).
4. Webhooks: point the platform at `/api/webhooks/{provider}`; set
   `HIKMAON_WEBHOOK_VERIFY_TOKEN` for the Meta subscription handshake.
   Deliveries are authenticated via `X-Hub-Signature-256` HMAC or the
   `X-Hikmaon-Webhook-Secret` shared secret.

Snapchat has no consumer content-read API — that provider requires the
on-device attestation plug-in path described in the patent.

---

## 4.5 Accounts, video/audio matching, and the crawler

**Accounts** — users register/login at `/api/auth/register` and `/api/auth/login`
(Argon2id hashing, JWT access tokens, rotating refresh tokens with reuse
detection, lockout after 5 failed attempts). Every account is issued an
Ed25519 ownership keypair; registrations are signed with it automatically
(`ownership_proven: true`). All API data is scoped to the authenticated
account; the first registered account gets the `admin` role.

**Video/audio matching** — requires ffmpeg, which installs automatically via
the `imageio-ffmpeg` dependency (or set `HIKMAON_FFMPEG`). Video identity is
a sequence of per-frame perceptual hashes matched by temporal alignment
(re-encodes, downscales, and trims match; the offset reports where the clip
was cut). Audio uses Haitsma–Kalker spectral fingerprints matched by
bit-error-rate (canonical decision line BER < 0.35); video soundtracks are
fingerprinted too.

**Crawler** — `POST /api/crawler/jobs {"seed_urls": [...], "max_pages": 50}`
starts a background crawl scoped to the seed domains: robots.txt respected
per host, 1 req/s politeness, page/media size caps, and an SSRF guard that
DNS-resolves every host and refuses non-global addresses. Discovered media
is fingerprinted, indexed as a public sighting, and matched against all
registrations — matches open incidents through the standard consent
workflow. For continuous autonomous monitoring set `HIKMAON_CRAWLER_SEEDS`
and `HIKMAON_CRAWLER_INTERVAL_MINUTES`.

## 5. Production checklist

- [ ] `HIKMAON_SIGNING_KEY` and `HIKMAON_TOKEN_KEY` set from a secrets manager and backed up
- [ ] `HIKMALAYER_RPC_URL` pointing at a production node; verify `chain_mode: rpc` in `/health`
- [ ] Trained + calibrated `hikmaonnet.onnx` deployed; `/api/model/status` shows `loaded`
- [ ] `HIKMAON_CORS_ORIGINS` restricted to the real dashboard origin
- [ ] TLS termination in front of uvicorn; API authentication (gateway/JWT) in front of all `/api/*` routes
- [ ] `HIKMAON_REQUIRE_OWNERSHIP_PROOF=1` so registrations must prove key control
- [ ] Webhook verify token + shared secret set; provider OAuth apps approved for production scopes
- [ ] Data dir on durable storage with backups (until Postgres backend lands)
- [ ] Takedown platform-API submissions wired in `TakedownService._submit_platform_reports`
- [ ] Legal review of DMCA notice template per operating jurisdiction
