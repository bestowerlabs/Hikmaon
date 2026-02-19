# Hikmaon Development Stages (Implemented in this iteration)

## Stage 1 — Foundation + Registration + Blockchain anchoring

### What was built
- FastAPI backend scaffold with modular service boundaries.
- `RegistrationService` to accept media payloads, compute SHA-256, build fingerprint commitment, generate deterministic embeddings, and submit a simulated Hikmalayer `MEDIA_REGISTRATION` transaction.
- In-memory storage contract for registrations and chain records.

### Why it was built this way
- This stage follows the mandated order: registration and anchoring first.
- Hash/fingerprint are generated before blockchain submission, preserving patent consistency.
- Raw media is not persisted on-chain.

### What is next
- Replace simulated chain write with real Hikmalayer RPC client.
- Add Postgres + object storage persistence.

## Stage 2 — AI Similarity + Dashboard integration

### What was built
- `AIService` containing:
  - Embedding engine
  - Similarity engine (cosine similarity against stored vectors)
  - Deepfake probability estimator (placeholder classifier)
  - Decision fusion confidence scoring
- Frontend dashboard (`frontend/index.html` + `frontend/app.js`) supporting register/analyze/verify/evidence/notify operational flow.

### Why it was built this way
- Keeps AI computation off-chain and model outputs auditable.
- Provides immediate human-in-the-loop control via UI for operations teams.

### What is next
- Replace placeholder deepfake estimator with production model serving.
- Add vector database (FAISS/pgvector/etc.) and ANN indexing.

## Stage 3 — Verification + Evidence + Notification

### What was built
- `VerificationService` to prove ownership from blockchain records.
- `EvidenceService` for structured legal-grade report JSON generation.
- `NotificationService` for logging and dispatch abstraction across channels.

### Why it was built this way
- Verification and evidence are the legal backbone of Hikmaon.
- Notification closes the loop for automated misuse response.

### What is next
- Add signed PDF report export.
- Add enterprise webhook retry policies and delivery telemetry.

## Stage 4 — Internet Discovery baseline

### What was built
- `MonitoringService` with a public media indexing primitive storing URL + first-seen timestamp and lightweight fingerprint map.

### Why it was built this way
- Enables monitoring records without forcing heavy model inference over all web media.

### What is next
- Introduce lawful crawler orchestration with robots.txt support and domain policy controls.
- Trigger deep AI only on thresholded candidates.

## Stage 5 — Optional social/cloud integration

### What was built
- Not implemented yet in code (kept intentionally optional per architecture requirements).

### What is next
- Add OAuth account connectors and encrypted token vault.
