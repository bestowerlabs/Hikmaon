# Hikmaon Development Stages (Current Build)

## Stage 7 — HikmaonNet neural detector + platform API access

### What was done
- **HikmaonNet** (`backend/ml/`): trainable multi-branch deepfake detector — ConvNeXt-style spatial branch, FFT frequency branch, fixed-SRM noise branch, attention fusion, calibration temperature (9.4M params). Full pipeline validated end-to-end: manifest-driven dataset with anti-recompression augmentations, AMP training loop with class balancing and cosine schedule, evaluation with per-generator AUC/EER and temperature fitting, ONNX export verified to 1e-8 against torch.
- **Model serving** (`app/services/model_serving.py`): torch-free ONNX serving via `HIKMAON_MODEL_PATH`; the calibrated neural probability becomes the dominant manipulation signal fused with forensic heuristics; `/api/model/status` reports deployment state honestly.
- **Platform API access** (`app/integrations/`): OAuth2 authorization-code flow with PKCE for 11 providers, Fernet-encrypted token vault with refresh, media-sync adapters against the real Graph API / Google Drive / Dropbox / OneDrive / X endpoints, and webhook receivers with Meta handshake + HMAC signature verification. Providers activate via per-platform credentials; unconfigured providers return precise setup instructions.
- Branded dashboard (Bestower Labs) with live status chips, OAuth connect, media sync, and consent workflow; deployment guide at `docs/DEPLOYMENT.md`.

### Why
- The neural detector is the accuracy ceiling-raiser — the team trains it on GPU with the provided pipeline; everything downstream (fusion, evidence, versioning) is already wired.
- Platform credentials are now the only thing between the scaffold and live ingestion.

### What next
- Train HikmaonNet on FaceForensics++/DFDC/Celeb-DF + diffusion sets; gate release on held-out-generator AUC.
- Register production OAuth apps per platform; wire provider-native webhook payload translation as each goes live.
- ffmpeg frame/audio extraction feeding the same model interface for video/audio.

## Stage 6 — Real detection engine, certificates, and consent-driven takedown

### What was done
- Replaced all simulated AI with a real perceptual engine:
  - 64-bit DCT pHash + dHash, visual feature embeddings, chunk fingerprints for undecodable media.
  - Calibrated 0–100% match scoring (edited copies 62–100%, unrelated 0–28%) with match ≥55% / review 35–55% / no-match bands — unrelated media no longer creates incidents.
- Added manipulation forensics (ELA, noise-residual uniformity, frequency spectrum, AI-generator metadata) with per-signal explanations and honest abstention.
- Separated the three verdicts: perceptual match, manipulation indicators, and chain ownership — chain status no longer inflates detection confidence.
- Hikmalayer integration became a real RPC client (`HIKMALAYER_RPC_URL`) with retry/backoff; the local dev ledger is explicitly labelled `dev-simulated`.
- Ed25519-signed Certificate of Ownership issued per registration, verifiable via API; tampering is detected.
- Owner consent flow: incidents open as `pending_owner_review`; Allow closes, Remove auto-files a DMCA-style takedown case tracked open → reported → removed/rejected.
- Optional registration ownership proof (Ed25519 signature over the content hash).
- JSON snapshot persistence (state and signing key survive restarts), CORS, real file upload in the dashboard, 15-test suite.

### Why
- Delivers the product's core promises: percentage match, detection of edited copies, verifiable ownership certificates, and consent-driven removal.

### What next
- Trained deepfake detector ensembles (face forgery, temporal video, audio anti-spoofing) behind the existing `DetectorResult` interface.
- ffmpeg-based frame/audio decoding for video and audio perceptual matching.
- Provider OAuth + webhooks for connectors; platform abuse-API submission in `TakedownService._submit_platform_reports`.
- Postgres/object/vector storage; connect to the production Hikmalayer node.

## Stage 1 — Registration + Hikmalayer anchoring

### What was done
- Implemented automated registration pipeline for direct API and connector-ingested uploads.
- Every ingested media item now receives:
  - SHA-256 content hash
  - fingerprint commitment
  - deterministic embedding vector
  - Hikmalayer `MEDIA_REGISTRATION` transaction payload

### Why
- Preserves patent requirement: hash/fingerprint before on-chain proof.
- Keeps raw media off-chain while storing verifiable digital representation.

### What next
- Replace simulated chain write with live Hikmalayer RPC client and retry queues.

## Stage 2 — AI and detection fusion

### What was done
- Similarity search (cosine over stored embeddings).
- Deepfake probability estimation placeholder.
- Decision fusion score combining similarity, deepfake probability, and blockchain status.

### Why
- Enables AI-assisted misuse detection while preserving blockchain-backed legal proof.

### What next
- Deploy production model-serving stack for deepfake and multi-modal embeddings.

## Stage 3 — Verification, evidence, and alerts

### What was done
- Verification checks ownership/hash/timestamp against chain payload.
- Incident evidence report generation.
- Owner notification event emission.

### Why
- This creates legally traceable incident handling from detection to notification.

### What next
- Signed PDF evidence packaging and enterprise callback reliability policies.

## Stage 4 — Internet indexing and incident automation

### What was done
- Public media indexing endpoint (URL + fingerprint + first-seen timestamp).
- Realtime detection cycle endpoint that performs AI analysis, verification, evidence generation, and owner alerting in one pipeline.

### Why
- Delivers near-realtime misuse response flow as required in system goals.

### What next
- robots.txt-compliant crawler scheduler and selective deep-AI execution policy.

## Stage 5 — Optional social/cloud integration

### What was done
- Implemented connector APIs for social/cloud account linking and simulated realtime upload ingestion.
- Supported providers in scaffold:
  - X, Instagram, Facebook, YouTube, TikTok, LinkedIn
  - Google Drive, Dropbox, OneDrive

### Why
- Matches requirement that user uploads across accounts/storage can be auto-registered and anchored.

### What next
- Provider-by-provider OAuth + webhook integration with encrypted token vault and key rotation.
