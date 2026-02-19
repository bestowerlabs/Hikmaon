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

## Core principle

- **Off-chain AI handles detection and analysis.**
- **On-chain blockchain stores proof and ownership anchoring.**
- Hikmaon connects both to produce verifiable evidence.

## Repository structure

- `backend/` FastAPI backend with modular services.
- `frontend/` Operational dashboard for manual workflow execution.
- `docs/architecture.md` Full target architecture specification.
- `docs/development_stages.md` Iteration-by-iteration implementation log (what/why/next).

## Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run frontend dashboard

Open `frontend/index.html` in a browser while backend runs at `http://localhost:8000`.

## Testing

```bash
cd backend
PYTHONPATH=. pytest
```
