# Hikmaon — by Bestower Labs

Hikmaon is a **blockchain-anchored digital authenticity and deepfake misuse detection & prevention platform**: AI-based deepfake detection and prevention using a hybrid blockchain network (Hikmalayer).

![Hikmaon Command Center dashboard](docs/dashboard.png)

## What it does

1. **Connect** — owners link social/cloud accounts (Instagram, Facebook, X, YouTube, TikTok, LinkedIn, Reddit, Google Drive, Dropbox, OneDrive) via real OAuth2 flows; uploads flow in through media sync and realtime webhooks.
2. **Anchor** — every media item gets a SHA-256 hash, perceptual fingerprints, and an AI embedding; the proof is anchored on **Hikmalayer** and the owner receives an **Ed25519-signed Certificate of Ownership** anyone can verify.
3. **Detect** — monitoring compares suspicious media perceptually and reports a **0–100% match** (edited copies — re-encoded, resized, cropped — still match). Manipulation analysis fuses **HikmaonNet** (trainable neural detector) with forensic signals (Error Level Analysis, noise residuals, frequency spectrum, AI-generator metadata), each explained, with honest abstention.
4. **Enforce** — on a confirmed match the owner is alerted and chooses **Allow** or **Remove**; refusal auto-files a DMCA-style takedown case with the blockchain evidence attached, tracked to `removed`/`rejected`.

## Architecture

| Layer | Where | Status |
|---|---|---|
| Perceptual matching (pHash/dHash + embeddings, % score) | `backend/app/perceptual.py` | working, calibrated |
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

## Quick start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload           # API on :8000
# open frontend/index.html in a browser
```

Run tests (26 tests, full lifecycle covered):

```bash
cd backend && PYTHONPATH=. pytest
```

## Training HikmaonNet (your GPU team)

> **New to this? Read [`docs/HOW_TO_TRAIN_THE_MODEL.md`](docs/HOW_TO_TRAIN_THE_MODEL.md)** —
> a complete, plain-language, step-by-step guide written so anyone can train the
> model (covers getting FaceForensics++, preparing data, training, and deploying).

```bash
pip install -r ml/requirements.txt

# 0a) Turn downloaded videos into training frames (once per folder).
python -m ml.prepare_dataset --videos /data/ffpp/original_sequences --out /data/frames/real
python -m ml.prepare_dataset --videos /data/ffpp/manipulated_sequences/Deepfakes --out /data/frames/deepfakes

# 0b) Build the manifest from your frame folders (generates /data/manifest.csv).
#    Frames of one video never leak across splits; --holdout keeps a generator
#    out of training so its test AUC measures true generalization.
python -m ml.make_manifest \
    --real /data/ffpp/real --real /data/celebdf/real \
    --fake deepfakes=/data/ffpp/deepfakes \
    --fake face2face=/data/ffpp/face2face \
    --fake celebdf=/data/celebdf/fake \
    --holdout celebdf \
    --out /data/manifest.csv

# 1-3) Train (pretrained EfficientNet-B0 backbone by default — needs internet
#       to download weights once), evaluate + calibrate, export
python -m ml.train    --manifest /data/manifest.csv --out runs/v1 --epochs 30
python -m ml.evaluate --manifest /data/manifest.csv --checkpoint runs/v1/best.pt --split test --fit-temperature
python -m ml.export   --checkpoint runs/v1/best.pt --out hikmaonnet.onnx
HIKMAON_MODEL_PATH=hikmaonnet.onnx uvicorn app.main:app
```

Manifest format, dataset guidance (FaceForensics++, DFDC, Celeb-DF, diffusion sets), and the cross-generator evaluation discipline are documented in `backend/ml/data.py` and `docs/DEPLOYMENT.md`.

## Documentation

- **`docs/HOW_TO_TRAIN_THE_MODEL.md`** — beginner-friendly, step-by-step training guide (data → prepare → train → deploy), written for non-programmers.
- **`docs/DEPLOYMENT.md`** — full deployment guide: server, accounts, billing, AV matching, crawler, model training→serving, Hikmalayer connection, per-platform OAuth activation, production checklist.
- `docs/architecture.md` — technical target architecture.
- `docs/deepfake_detection_assessment.md` — capability assessment and roadmap.
- `docs/development_stages.md` — stage-by-stage build log.
- `docs/Hikmaon Whitepaper v1.1.pdf` — whitepaper.

## Honest scope notes

Deepfake detection is an adversarial race; no system is perfect and anyone claiming 100% is wrong. Hikmaon's design goal is **calibrated accuracy with honest abstention**: percentage scores with explicit thresholds, per-signal explanations in every evidence report, model versions logged for auditability, and `dev-simulated` vs `rpc` chain modes that can never be confused. Remaining production increments: train HikmaonNet on the full dataset mix (pipeline ready), ffmpeg-based video/audio frame matching, per-platform takedown API submission, and Postgres/vector-DB storage.

---

Hikmaon © Bestower Labs — Inventor: Muhammad Ayan Rao.
