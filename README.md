# Hikmaon — by Bestower Labs

**AI-based deepfake detection and prevention using a hybrid blockchain network (Hikmalayer).**

Hikmaon is a blockchain-anchored digital authenticity platform: it proves who owns a piece of media, recognizes stolen or edited copies of it anywhere with a percentage-match score, analyzes media for manipulation, and enforces the owner's decision with automated takedowns.

## Live system evidence

The screenshot below is the **actual dashboard of a running Hikmaon instance** (not a mockup), captured during end-to-end verification — logged-in session for Ayan Rao with his account's Ed25519 ownership key, a completed autonomous crawler job, and two real incidents produced by the detection engine:

- a **stolen video** (trimmed 2 s, downscaled, bitrate-crushed) detected at **89.8% match** — owner refused consent, takedown case filed with DMCA notice
- a **stolen image** (resized + JPEG-recompressed) detected at **97.1% match** — awaiting the owner's Allow/Remove decision

![Hikmaon Command Center — live logged-in session (Ayan Rao) with real incidents and a takedown case](docs/dashboard.png)

### Verified detection results

All numbers below were measured on this codebase during development (test suite reproduces them):

| Scenario | Result |
|---|---|
| Image: identical / re-encoded (JPEG q40) / resized / blurred | **100% match** |
| Image: brightness +25% / grayscale / 90% crop | **93.6% / 87.7% / 62.3%** |
| Image: unrelated content | 0–28% (no match) |
| Video: trimmed 2 s + downscaled + heavily recompressed | **89.8% match**, trim offset located |
| Video: unrelated content | 17.3% (no match) |
| Audio: MP3-32k re-encode + 3 s trim | **81.1% match** (BER 0.09) |
| Audio: volume +60% + re-encode | **85.9% match** (BER 0.07) |
| Audio: unrelated content | 13.5% (BER 0.43, no match) |
| Certificate tampering (any field changed) | verification fails |
| Refresh-token replay after rotation | session family revoked (401) |

Decision thresholds: **match ≥ 55%** (incident opened, owner alerted), **35–55%** review band (queued, no alert), **< 35%** no match. Audio uses the canonical Haitsma–Kalker BER < 0.35 decision line.

## What it does

1. **Accounts** — modern login/register: Argon2id password hashing, JWT sessions with rotating refresh tokens (reuse detection revokes the session family), login throttling with lockout, and an **Ed25519 ownership keypair issued to every account** so all registrations are cryptographically signed automatically. All owner data is scoped per account.
2. **Connect** — owners link social/cloud accounts (Instagram, Facebook, X, YouTube, TikTok, LinkedIn, Reddit, Google Drive, Dropbox, OneDrive) via real OAuth2 + PKCE flows bound to the logged-in user; uploads flow in through media sync and realtime webhooks.
3. **Anchor** — every media item gets a SHA-256 hash, perceptual fingerprints, and an AI embedding; the proof is anchored on **Hikmalayer** and the owner receives an **Ed25519-signed Certificate of Ownership** anyone can verify.
4. **Detect** — suspicious media is compared perceptually with a **0–100% match score** across all modalities:
   - **Images**: DCT pHash + dHash + visual embeddings (survives re-encode, resize, blur, brightness, moderate crops)
   - **Video**: per-frame perceptual hashing with temporal alignment — re-encoded, downscaled, bitrate-crushed, and *trimmed* copies still match, and the alignment reports where the clip was cut
   - **Audio**: Haitsma–Kalker spectral fingerprinting (the industrial robust-audio-hash design) — MP3/AAC re-encodes, volume changes, and trims match by bit-error-rate; a video's soundtrack is fingerprinted too, so clips match on either channel
   - Manipulation analysis fuses **HikmaonNet** (trainable neural detector) with explainable forensic signals (Error Level Analysis, noise residuals, frequency spectrum, AI-generator metadata), with honest abstention.
5. **Monitor** — an **autonomous web crawler** (robots.txt-compliant, per-host politeness, SSRF-hardened, domain-scoped) scans seed sites, extracts and fingerprints media, and automatically opens incidents on matches. Run on demand via API/dashboard or on a schedule (`HIKMAON_CRAWLER_SEEDS` + `HIKMAON_CRAWLER_INTERVAL_MINUTES`).
6. **Enforce** — on a confirmed match the owner is alerted and chooses **Allow** or **Remove**; refusal auto-files a DMCA-style takedown case with the blockchain evidence attached, tracked to `removed`/`rejected`.

## Architecture

```text
                 ┌────────────────────── Hikmaon ──────────────────────┐
Social/Cloud ─▶ OAuth + Webhooks + Sync ─▶ Registration ─▶ Hikmalayer client ─▶ Hikmalayer
platforms        (encrypted token vault)   hash + fingerprints          (separate project,
                                           + Ed25519 certificate         hybrid PoW+PoS)
Public web  ─▶ Autonomous crawler ─▶ Sighting index ─┐
                                                      ▼
Suspicious media ─▶ AI engine: % match (image/video/audio) + forensics + HikmaonNet
                                                      ▼
                          Incident ─▶ Owner consent (Allow / Remove)
                                                      ▼
                          Takedown case ─▶ DMCA notice + tracking ─▶ removed/rejected
```

| Layer | Where | Status |
|---|---|---|
| Perceptual matching (pHash/dHash + embeddings, % score) | `backend/app/perceptual.py` | working, calibrated |
| **Video/audio matching** (frame-hash temporal alignment + Haitsma–Kalker audio fingerprints, ffmpeg-based) | `backend/app/av_fingerprint.py` | working, calibrated |
| **Auth** (Argon2id, JWT + rotating refresh, throttling, per-user Ed25519 keys) | `backend/app/auth.py` | working |
| **Autonomous crawler** (robots.txt, politeness, SSRF guard, auto-incidents) | `backend/app/services/crawler.py` | working |
| Manipulation forensics (ELA, noise, spectrum, metadata) | `backend/app/forensics.py` | working, calibrated |
| **HikmaonNet neural detector** (spatial + frequency + SRM-noise branches, attention fusion) | `backend/ml/` | architecture + full training/eval/export pipeline ready — **train on your GPU cluster** |
| Model serving (ONNX, torch-free) | `backend/app/services/model_serving.py` | working (`HIKMAON_MODEL_PATH`) |
| Hikmalayer client (RPC + retries; chain is a separate project) | `backend/app/hikmalayer.py` | working (`HIKMALAYER_RPC_URL`) |
| Certificates of Ownership (Ed25519) | `backend/app/services/certificate.py` | working |
| Platform OAuth2 + PKCE, encrypted token vault | `backend/app/integrations/oauth.py` | working — activates per provider via credentials |
| Media sync (Graph API, Drive, Dropbox, OneDrive, X) | `backend/app/integrations/sync.py` | implemented against real provider APIs |
| Webhooks (Meta handshake, HMAC verification) | `backend/app/integrations/webhooks.py` | working |
| Consent + takedown workflow | `backend/app/services/takedown.py` | working |
| Dashboard | `frontend/` | working |

## API overview

| Area | Endpoints |
|---|---|
| Auth | `POST /api/auth/register` · `login` · `refresh` · `logout` · `GET /api/auth/me` |
| Registration | `POST/GET /api/registrations` (auto-signed with the account key) |
| Certificates | `GET /api/certificates/{media_id}` · `POST /api/certificates/verify` (public) |
| Connectors | `POST/GET/DELETE /api/connectors` · `POST /api/connectors/ingest` · `POST /api/connectors/{id}/sync` |
| OAuth | `GET /api/connectors/oauth/{provider}/start` · `/callback` · `GET /api/integrations/status` |
| Webhooks | `GET/POST /api/webhooks/{provider}` (HMAC / shared-secret authenticated) |
| Detection | `POST /api/analyze` · `POST /api/realtime/detect` · `POST /api/monitor/index` |
| Crawler | `POST/GET /api/crawler/jobs` · `GET /api/crawler/jobs/{id}` |
| Incidents | `GET /api/incidents` · `POST /api/incidents/{id}/decision` (Allow/Remove) |
| Takedowns | `GET /api/takedowns` · `POST /api/takedowns/{id}/status` |
| Evidence | `POST /api/verify` · `POST /api/evidence/{id}` · `GET/POST /api/notifications` |
| Model | `GET /api/model/status` |

## Quick start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload           # API on :8000
# open frontend/index.html, create an account, and go
```

Run tests (45 tests: auth, image/video/audio matching, forensics, certificates, crawler, integrations, full lifecycle):

```bash
cd backend && PYTHONPATH=. pytest
```

## Training HikmaonNet (your GPU team)

```bash
pip install -r ml/requirements.txt

# 0) Build the manifest from your dataset folders (generates /data/manifest.csv).
#    Frames of one video never leak across splits; --holdout keeps a generator
#    out of training so its test AUC measures true generalization.
python -m ml.make_manifest \
    --real /data/ffpp/real --real /data/celebdf/real \
    --fake deepfakes=/data/ffpp/deepfakes \
    --fake face2face=/data/ffpp/face2face \
    --fake celebdf=/data/celebdf/fake \
    --holdout celebdf \
    --out /data/manifest.csv

# 1-3) Train, evaluate + calibrate, export
python -m ml.train    --manifest /data/manifest.csv --out runs/v1 --epochs 30
python -m ml.evaluate --manifest /data/manifest.csv --checkpoint runs/v1/best.pt --split test --fit-temperature
python -m ml.export   --checkpoint runs/v1/best.pt --out hikmaonnet.onnx

# 4) Deploy
HIKMAON_MODEL_PATH=hikmaonnet.onnx uvicorn app.main:app
```

Manifest format, dataset guidance (FaceForensics++, DFDC, Celeb-DF, diffusion sets), and the cross-generator evaluation discipline are documented in `backend/ml/make_manifest.py`, `backend/ml/data.py`, and `docs/DEPLOYMENT.md`.

## Documentation

- **`docs/DEPLOYMENT.md`** — full deployment guide: server, accounts, AV matching, crawler, model training→serving, Hikmalayer connection, per-platform OAuth activation, production checklist.
- `docs/architecture.md` — technical target architecture.
- `docs/deepfake_detection_assessment.md` — capability assessment, roadmap, and delivery status.
- `docs/development_stages.md` — stage-by-stage build log (Stages 1–8).
- `docs/Hikmaon Whitepaper v1.1.pdf` — whitepaper.

## Honest scope notes

Deepfake detection is an adversarial race; no system is perfect and anyone claiming 100% is wrong. Hikmaon's design goal is **calibrated accuracy with honest abstention**: percentage scores with explicit thresholds, per-signal explanations in every evidence report, model versions logged for auditability, and `dev-simulated` vs `rpc` chain modes that can never be confused. Audio fingerprinting is strongest on real-world audio (speech/music); spectrally sparse synthetic tones are a known weak case. Remaining production increments: train HikmaonNet on the full dataset mix (pipeline ready), per-platform takedown API submission, per-platform OAuth app registration, and Postgres/vector-DB storage.

---

Hikmaon © Bestower Labs Limited — Inventor: Muhammad Ayan Rao.
