# Hikmaon Deepfake Detection Assessment

**Date:** 2026-07-02
**Scope:** Full repository review — backend services, frontend dashboard, tests, and architecture docs — assessing what the system does today and what it must do to deliver accurate deepfake **detection** and meaningful **prevention**.

---

## Delivery status update (2026-07-04)

This assessment described the repository as it stood on 2026-07-02. Since then the roadmap has been executed (Stages 6–8 in `docs/development_stages.md`). Status against the priority table in §5:

| Priority | Work item | Status |
|---|---|---|
| P0 | Similarity threshold + `no_match` path; fix tautological `blockchain_verified`; fix requirements | ✅ Done |
| P0 | Authentication + key-ownership proof at registration | ✅ Done — Argon2id/JWT account system; per-user Ed25519 keys auto-sign every registration |
| P1 | Real perceptual hashing + real embeddings | ✅ Done — images (pHash/dHash/embeddings), **video** (frame-hash temporal alignment), **audio** (Haitsma–Kalker fingerprints); vector DB still pending at scale |
| P1 | Persistent storage | 🟡 JSON snapshot persistence done; Postgres/object/vector stores pending |
| P2 | Production deepfake classifier + calibration + evaluation harness | 🟡 HikmaonNet architecture + train/eval/export/serving pipeline done; **GPU training on real datasets pending (team)** |
| P2 | Hikmalayer RPC client + Certificate of Ownership | ✅ Done — RPC client with retries; Ed25519-signed certificates with verify endpoint |
| P3 | C2PA signing/verification + invisible watermarking | ❌ Open |
| P3 | Crawler + platform ingestion + takedown automation | ✅ Mostly done — autonomous robots.txt-compliant crawler with auto-incidents; OAuth/webhook/sync layer built (platform app credentials pending); takedown cases + DMCA notices generated (per-platform submission APIs pending) |
| P4 | Learned fusion, explainability, signed evidence, human review queue | 🟡 Signal-level explanations + review band done; learned fusion and signed PDF evidence open |

The remainder of this document is preserved as the original assessment.

---

## 1. What this repository does today

Hikmaon is a **blockchain-anchored digital authenticity and deepfake misuse detection platform**. The intended flow (per `README.md` and `docs/architecture.md`):

1. Owners register original media (directly or via social/cloud connectors).
2. The backend computes a SHA-256 hash, a perceptual fingerprint, and an AI embedding.
3. A proof (`content_hash`, fingerprint commitment, owner key, timestamp) is anchored on the **Hikmalayer** blockchain.
4. A monitoring service indexes public media and runs similarity-first detection.
5. Similarity + deepfake probability + blockchain verification are fused into a confidence score.
6. Verified incidents produce evidence reports and owner notifications.

The implementation is a **FastAPI scaffold** (`backend/`, ~600 lines) with an in-memory store, a vanilla-JS demo dashboard (`frontend/`), and a single happy-path integration test. The architecture is coherent and patent-aligned, and the service decomposition (registration / AI / monitoring / verification / evidence / notification / pipeline) is a sound skeleton.

**However, every intelligence and trust component is currently simulated.** The repo is an API-shape prototype, not a detection system:

| Component | Claimed | Actual implementation |
|---|---|---|
| Embedding (`ai.py:_embedding`, `registration.py:_embedding`) | ResNet/ViT similarity encoder | Random 512-dim vector seeded from the file's SHA-256 |
| Deepfake probability (`ai.py:_deepfake_probability`) | CNN classifier | Last 6 hex digits of SHA-256 mod 1000 ÷ 1000 — a deterministic **random number**, uncorrelated with manipulation |
| Perceptual fingerprint (`registration.py:_fingerprint`) | pHash / MFCC | SHA-256 of the first 4 KB — a cryptographic hash, not perceptual |
| Blockchain anchoring (`registration.py:_submit_hikmalayer_tx`) | Hikmalayer RPC transaction | Write to an in-process Python dict with a random `txid` |
| Verification (`verification.py`) | Query chain node | Reads back the same in-process dict, so it always verifies |
| Public monitoring (`monitoring.py`) | Crawler + ANN index | An endpoint the caller must push bytes to, keyed by **exact** content hash |
| Storage (`storage.py`) | Postgres + object store + vector DB | `InMemoryStore` dataclass — all state lost on restart |

## 2. Correctness defects in the current scaffold

These are bugs even at prototype level:

1. **No similarity threshold → guaranteed false incidents.** `AIService.analyze` returns the best-scoring registration regardless of score, and `AutomationPipelineService.run_detection_cycle` treats any returned match as an incident. Once one media item is registered, *every* probe — including completely unrelated content — creates an incident, an evidence report, and an owner notification.
2. **Exact-hash matching defeats the purpose.** Because embeddings are seeded from SHA-256, a single changed byte (re-encoding, resizing, cropping, a real deepfake of the registered face) produces an unrelated embedding with ~0 cosine similarity. The system can only "detect" bit-identical copies — the one case deepfakes never are.
3. **`blockchain_verified` is tautological.** `ai.py` checks `best_match.blockchain_txid in self.store.blockchain_records`; the same process wrote that record, so it is always `True`, silently adding +0.2 confidence to every analysis.
4. **Fusion formula conflates two unrelated questions.** Blockchain verification proves *ownership of the original*; it says nothing about whether the *probe* is manipulated, yet it inflates the "misuse" confidence.
5. **No authentication or authorization.** Anyone can register media under any `owner_id`/public key (claiming ownership of others' content), list/disconnect connectors, read all incidents, and fire notifications.
6. **Unbounded in-process state.** `analysis_cache` in `main.py` grows forever; all stores vanish on restart, orphaning anchored "proofs".
7. **Connector "encryption" is base64 of a fabricated string** — no OAuth, no tokens, no KMS.
8. **Test suite doesn't run from a clean install**: `httpx` (required by `fastapi.testclient`) is missing from `requirements.txt`. The single test covers only the happy path where the "fake" is a byte-identical copy — precisely the case that can't be a deepfake.
9. **CORS/frontend gap**: the dashboard "uploads" are base64 of textarea *text*; there is no file upload, and no CORS middleware is configured for cross-origin use beyond same-origin file:// quirks.

## 3. What it needs to become an accurate deepfake detection system

### 3.1 Real perceptual fingerprinting and similarity (Detection tier 1)

- **Images:** pHash + PDQ (Meta's open-source perceptual hash); store in an indexable Hamming-distance structure.
- **Video:** frame sampling (1–3 fps) + PDQ per frame, plus TMK+PDQF or vPDQ for temporal matching.
- **Audio:** chromaprint-style fingerprints or MFCC landmarks.
- **Embeddings:** replace the seeded RNG with real encoders — CLIP/ViT for general imagery, a face-recognition embedding (e.g., ArcFace-class) for identity matching, wav2vec-style encoders for voice — served behind a model-serving layer (TorchServe/Triton/ONNX Runtime).
- **Vector search:** FAISS/pgvector/Milvus ANN index instead of a linear scan over a dict, with a **calibrated similarity threshold** and a "no match" outcome.
- **Percentage match:** expose the calibrated combination of perceptual-hash distance and embedding similarity as a 0–100% match score in incidents, alerts, and evidence reports — this is the owner-facing number the product promises, and it is what detects *edited/partially modified* copies, not just exact duplicates.

### 3.2 Real deepfake classifiers (Detection tier 2)

- **Face/image forensics:** an ensemble of complementary detectors — spatial-artifact CNNs/ViTs, frequency-domain detectors, blending-boundary detectors — trained/fine-tuned on FaceForensics++, DFDC, Celeb-DF, and current diffusion-generated data.
- **Video:** frame-level classification + temporal-consistency models (optical-flow/rPPG-based physiological cues) with score smoothing.
- **Audio:** anti-spoofing models of the AASIST/ASVspoof lineage for TTS/voice-conversion detection.
- **Generalization discipline:** deepfake generators evolve monthly. Accuracy claims require a continuous retraining + evaluation pipeline: held-out *cross-generator* test sets, drift monitoring, scheduled re-benchmarks, and red-teaming with the newest open generators.
- **Calibration and abstention:** report calibrated probabilities (temperature scaling, ECE tracking), define per-modality operating thresholds tuned to explicit FP/FN targets, and add an "uncertain — needs human review" band. A system that only outputs `real/fake` with no abstention will be confidently wrong.
- **Explainability:** the architecture promises an `explanation_map`; deliver Grad-CAM/attention heatmaps and artifact localization in evidence reports — essential if evidence is meant to be legally useful.

### 3.3 Honest decision fusion

- Replace the fixed `0.5/0.3/0.2` linear formula with a **learned, calibrated fusion model** (even logistic regression over the component scores is defensible) validated on labeled incident data.
- Separate the two verdicts the product actually needs:
  1. **"Is this derived from a registered original?"** (similarity + fingerprint + chain ownership proof)
  2. **"Is this media synthetic/manipulated?"** (deepfake classifiers + provenance signals)
  Blockchain verification belongs to verdict 1 only.

### 3.4 Real infrastructure

- **Persistence:** Postgres (users, registrations, incidents, notifications), object storage (media, evidence), vector DB (embeddings). Migrations, backups.
- **Blockchain:** Hikmalayer (the hybrid PoW+PoS chain) is developed as a **separate project — do not rebuild it here**. Hikmaon needs only the client side: an RPC client with retry queues and confirmation tracking that submits `MEDIA_REGISTRATION` transactions, stores the real `txid`, and queries the chain for verification. On top of that, issue a **Certificate of Ownership** per registration — a signed JSON + PDF containing content hash, owner public key, timestamp, and Hikmalayer `txid`, independently verifiable against the chain.
- **Discovery:** a real robots.txt-compliant crawler with a scheduler, media extractor, and the documented "cheap fingerprint first, deep AI only on candidate hits" tiering — plus platform-specific ingestion (official APIs/webhooks) since most misuse happens on platforms, not the open web.
- **Security:** OAuth2/JWT auth with proof-of-key-ownership at registration (sign a challenge with `owner_public_key` — otherwise ownership claims are meaningless), rate limiting, KMS-encrypted connector tokens with rotation, audit logs, signed evidence reports (the schema promises "signed digital report"; implement an actual signature).
- **Engineering hygiene:** fix `requirements.txt` (add `httpx`), add CORS middleware, add negative-path tests (non-matching probe → `no_match`; manipulated copy → match), CI, and load-shedding for the analysis cache.

## 4. What it needs for prevention (currently absent entirely)

Detection tells you misuse happened; prevention reduces the harm or stops it recurring. None of this exists in the repo yet:

1. **Provenance at creation — C2PA / Content Credentials.** Sign media at registration time with C2PA manifests and verify manifests on probes. This is the industry-standard "prevention" layer: consumers and platforms can verify authenticity *before* a fake spreads, and it composes naturally with Hikmalayer anchoring (anchor the manifest hash on-chain).
2. **Robust invisible watermarking.** Embed an imperceptible, transformation-robust watermark in registered media so derivatives remain attributable even after re-encoding/cropping; check for watermark presence/absence during analysis as another fusion signal.
3. **Owner-consent takedown flow.** Detection must end in a decision, not just an alert: notify the owner with the match evidence and a **percentage-match score** → owner chooses **Allow** (case closed, logged) or **Remove** → the system automatically files templated DMCA notices / platform abuse-report API submissions with the blockchain ownership certificate attached, tracks the case (open → reported → removed/rejected), and re-scans to confirm removal. Note: platforms make the final removal decision; Hikmaon automates the filing and maximizes success with chain-anchored evidence.
4. **Real-time protection surface.** A public verification API/badge ("verify this media against Hikmaon") and webhook subscriptions so platforms can check content at upload time — moving from after-the-fact discovery to pre-publication screening.
5. **Owner-side hygiene features.** Alerting SLAs, monitored identity profiles (faces/voices, with explicit consent), and periodic sweeps of high-risk platforms for registered identities, not just registered files.

## 5. Priority roadmap

| Priority | Work item | Rationale |
|---|---|---|
| P0 | Similarity threshold + `no_match` path; fix tautological `blockchain_verified`; add `httpx` to requirements | Current behavior fabricates incidents; tests must run |
| P0 | Authentication + key-ownership proof at registration | Without it, ownership anchoring is spoofable and worthless |
| P1 | Real perceptual hashing (PDQ/pHash) + real embeddings + vector DB | Minimum bar for detecting *transformed* copies |
| P1 | Persistent storage (Postgres/object/vector) | Proofs and incidents must survive restarts |
| P2 | Production deepfake classifier ensemble + calibration + evaluation harness | The actual "accurate detection" requirement |
| P2 | Hikmalayer RPC client integration (chain itself is a separate, already-developed project) + Certificate of Ownership issuance | Externally verifiable proof owners can hold and present |
| P3 | C2PA signing/verification + invisible watermarking | The core prevention layer |
| P3 | Crawler + platform ingestion + takedown automation | Closes the detect → act loop |
| P4 | Learned fusion, explainability maps, signed PDF evidence, human review queue | Legal-grade output quality |

## 6. Bottom line

The repository is a well-organized **architectural prototype**: the module boundaries, data models, and patent-aligned flow are all in place, and the docs are candid that connectors and Hikmalayer are scaffolded. But today it contains **no actual deepfake detection** — the deepfake score is a hash-derived random number, similarity only matches bit-identical files, the blockchain is a Python dict, and the pipeline raises an incident for any input once a single file is registered. "Perfect" is not attainable in this domain (detection is an adversarial arms race and any honest system reports calibrated probabilities with an abstention band), but *accurate and useful* is: it requires the real model stack (§3.1–3.3), real infrastructure and security (§3.4), and a prevention layer built on provenance, watermarking, and takedown automation (§4), in roughly the priority order above.
