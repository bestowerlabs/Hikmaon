# Hikmaon

Hikmaon is a **blockchain-anchored digital authenticity and deepfake misuse detection platform**.

It combines:

- Cryptographic hashing (SHA-256 content hashes)
- **Real perceptual fingerprinting** (64-bit DCT pHash + dHash) — edited copies still match
- **Visual similarity embeddings** (DCT energy + color + edge-orientation features)
- **Manipulation forensics** (Error Level Analysis, noise-residual uniformity, frequency-spectrum analysis, AI-generator metadata detection) with per-signal explanations
- **Percentage-match scoring** with calibrated match / review / no-match thresholds
- Blockchain ownership anchoring via **Hikmalayer** (separate project; integrated via RPC client)
- **Ed25519-signed Certificates of Ownership**, independently verifiable
- **Owner-consent enforcement**: Allow / Remove decision per incident, automated DMCA-style takedown case filing and tracking
- Public internet sighting index matched by perceptual distance
- Optional social media + cloud connectors for auto-ingestion

## Operating flow

1. User connects social/cloud accounts (optional module).
2. New media upload event is ingested automatically.
3. Hikmaon computes hash + perceptual fingerprint + embedding off-chain.
4. Hikmaon anchors digital proof (`content_hash`, fingerprint commitment, ownership, metadata pointer) on Hikmalayer and issues a signed **Certificate of Ownership**.
5. Monitoring indexes public internet sightings; detection compares suspicious media perceptually and reports a **0–100% match**.
6. Manipulation forensics report artifact indicators (or honestly abstain).
7. Verification checks the ownership proof against the chain.
8. On a confirmed match, the owner is alerted and chooses **Allow** or **Remove**; refusal auto-files a takedown case with the blockchain evidence attached, tracked to `removed`/`rejected`.

## Repository structure

- `backend/` FastAPI backend: perceptual engine, forensics, Hikmalayer client, certificates, consent/takedown workflow.
- `frontend/` Dashboard: file upload, % match display, incident consent buttons, takedown case tracking.
- `docs/architecture.md` Full technical target architecture.
- `docs/deepfake_detection_assessment.md` Capability assessment and production roadmap.
- `docs/development_stages.md` Stage-by-stage progress and next milestones.
- `docs/Hikmaon Whitepaper v1.1.pdf` Hikmaon whitepaper.

## Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HIKMALAYER_RPC_URL` | *(unset)* | Point at a real Hikmalayer node; without it a clearly-labelled local dev ledger is used (`chain_mode: dev-simulated`) |
| `HIKMAON_DATA_DIR` | `data` | JSON snapshot persistence directory (registrations, incidents, ledger, signing key) |
| `HIKMAON_SIGNING_KEY` | *(auto-generated)* | Hex seed for the Ed25519 certificate-issuing key |
| `HIKMAON_MATCH_THRESHOLD` | `55` | % above which a probe is a confirmed match (incident opened) |
| `HIKMAON_REVIEW_THRESHOLD` | `35` | % above which a probe is queued as possible match (no alert) |
| `HIKMAON_REQUIRE_OWNERSHIP_PROOF` | `0` | Set `1` to require an Ed25519 signature over the content hash at registration |
| `HIKMAON_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Run frontend dashboard

Open `frontend/index.html` in a browser while the backend runs at `http://localhost:8000`.

## Testing

```bash
cd backend
PYTHONPATH=. pytest
```

The suite covers: edited-copy matching (re-encode, resize, blur, brightness, crop), unrelated-media rejection, forensic verdicts (natural / AI-metadata / spliced), certificate issue + tamper detection, ownership signature proof, the full connector → incident → consent → takedown lifecycle, and abstention on undecodable media.

## Honest scope notes

- **Detection**: perceptual matching and forensic heuristics are real and calibrated, but the manipulation analysis is heuristic — production accuracy requires trained detector ensembles (face forgery, temporal video, audio anti-spoofing) served behind the same `DetectorResult` interface, with continuous retraining and benchmark evaluation.
- **Video/audio**: currently fingerprinted by content chunks (exact/trimmed copies match); frame-level and waveform-level perceptual matching require a media-decoding pipeline (ffmpeg + vPDQ/TMK, audio fingerprinting).
- **Connectors**: account linking and ingest are functional, but provider OAuth + official APIs/webhooks are the next production increment.
- **Takedown**: notices are generated and cases tracked; platform-specific report submission APIs plug into `TakedownService._submit_platform_reports`.
- **Chain**: Hikmalayer is developed separately. The RPC client is ready (`HIKMALAYER_RPC_URL`); adjust its route constants to the node's actual API when connecting.
