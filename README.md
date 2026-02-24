# Hikmaon

Hikmaon is a **blockchain-anchored digital authenticity and deepfake misuse detection platform**.

It combines:

- Cryptographic hashing
- Perceptual fingerprinting
- AI similarity embeddings
- Deepfake detection classifiers
- Public internet discovery and indexing
- Blockchain ownership anchoring via **Hikmalayer**
- Automated verification and notification
- Optional social media + cloud connectors for auto-ingestion

## Patent-aligned operating flow

1. User connects social/cloud accounts (optional module).
2. New media upload event is ingested automatically.
3. Hikmaon computes hash + fingerprint + embedding off-chain.
4. Hikmaon anchors digital proof (`content_hash`, fingerprint commitment, ownership, metadata pointer) on Hikmalayer.
5. Monitoring service indexes public internet media and executes similarity-first detection.
6. Deepfake + similarity decision fusion raises incident confidence.
7. Verification service checks blockchain ownership proof.
8. Hikmaon generates evidence and notifies owner automatically.

## Repository structure

- `backend/` FastAPI backend with modular services and orchestration pipeline.
- `frontend/` Patent-aligned command center UI for connectors and realtime detection flow.
- `docs/architecture.md` Full technical target architecture.
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

## Run frontend dashboard

Open `frontend/index.html` in browser while backend runs at `http://localhost:8000`.

## Testing

```bash
cd backend
PYTHONPATH=. pytest
```

## Important implementation note

Current connector and Hikmalayer interactions are scaffolded for development velocity. Next production increment should replace placeholders with:
- official provider OAuth/webhooks/APIs,
- real Hikmalayer RPC submission/query,
- persistent Postgres/object/vector stores,
- production deepfake model serving.
