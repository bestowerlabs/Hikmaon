# Hikmaon Technical Architecture Specification

> **Implementation status:** this document is the original *target* architecture.
> The current implementation (Stages 1–8, see `development_stages.md`) covers
> Modules 1–7 with working code: perceptual image/video/audio matching,
> forensic + neural manipulation analysis, Hikmalayer client + ownership
> certificates, autonomous crawler, OAuth/webhook connectors, accounts, and
> the consent-driven takedown workflow. Deltas from this spec are tracked in
> `deepfake_detection_assessment.md` (delivery status table).

## 1) Technical Definition

Hikmaon is a blockchain-anchored digital authenticity and deepfake misuse detection platform.

### Core components

- Cryptographic hashing
- Perceptual fingerprinting
- AI similarity embeddings
- Deepfake detection classifiers
- Public internet discovery + indexing
- Blockchain ownership anchoring (Hikmalayer)
- Automated verification + notification

### Core idea

- Off-chain AI does detection.
- On-chain blockchain does proof.
- Hikmaon connects both and produces verifiable evidence.

## 2) System Architecture

Hikmaon is built as modular services:

```text
Frontend (Web App / Dashboard)
        |
Application API Layer
        |
-----------------------------------
| Registration Service
| AI Service
| Monitoring Service
| Verification Service
| Notification Service
-----------------------------------
        |
Off-chain Database + Vector Index
        |
Hikmalayer Blockchain Network
```

## 3) Core Modules

## Module 1: Registration Service

### Purpose
Register original media and anchor proof on blockchain.

### Workflow
1. User uploads media (image/video/audio).
2. Generate cryptographic hash: `SHA-256(file_bytes)`.
3. Generate perceptual fingerprint:
   - Image: pHash
   - Video: frame sampling + pHash per frame
   - Audio: MFCC-based fingerprint
4. Generate AI embedding:
   - Use pretrained image/video/audio encoder
   - Output vector (512–1024 dimensions)
5. Store off-chain:
   - hash
   - fingerprint
   - embedding
   - owner ID
   - metadata
6. Create blockchain transaction payload:

```json
{
  "content_hash": "...",
  "fingerprint_commitment": "...",
  "owner_public_key": "...",
  "timestamp": "...",
  "metadata_pointer": "..."
}
```

7. Submit transaction to Hikmalayer node.
8. Store returned `txid` in database.

### Rule
Raw media is **not stored on-chain**.

## Module 2: AI Service (Core Intelligence)

### A) Embedding Engine (Similarity AI)

**Purpose:** Detect derived or similar content.

**Models:**
- Images: ResNet50 or ViT encoder
- Video: frame extraction (1–3 fps), frame encoding, pooled video embedding
- Audio: spectrogram conversion and pretrained audio encoder

**Storage + search:**
- Store embeddings in vector database
- Use ANN index (FAISS or equivalent)

**Matching logic:**
- `cosine_similarity(new_embedding, stored_embedding)`
- If similarity > threshold, classify as potential match

### B) Deepfake Detection Engine

**Purpose:** Detect manipulation artifacts.

- Image detection: CNN classifier (real vs fake)
- Video detection: frame-level classifier + temporal smoothing
- Audio detection: spectrogram classifier

**Output:**

```json
{
  "deepfake_probability": 0.0,
  "explanation_map": "...",
  "model_version": "..."
}
```

### C) Decision Fusion Engine

Combine:
- similarity score
- deepfake probability
- blockchain verification status

Example formula:

```text
confidence =
(similarity_score * 0.5) +
(deepfake_probability * 0.3) +
(blockchain_verified ? 0.2 : 0)
```

If confidence > threshold, trigger alert and evidence generation.

## Module 3: Internet Discovery & Indexing Service

A lawful public-media discovery system.

### Components
- `PublicCrawler()`
  - Crawl public web pages
  - Respect robots.txt
- `MediaExtractor()`
  - Extract media URLs
- `FingerprintGenerator()`
  - Generate lightweight fingerprints
- `FingerprintIndexStore()`
  - Store fingerprint, URL, and first-seen timestamp
- `ANNMatcher()`
  - Compare against registered embeddings

### Optimization rule
Do not run heavy AI on all discovered media.
Run deep AI only when similarity threshold is hit.

## Module 4: Social Media / Cloud Integration (Optional)

OAuth-based integration.

### Flow
1. User connects account via OAuth.
2. Store encrypted access token.
3. Periodically fetch new media via official APIs.
4. Run registration-style fingerprinting.
5. Compare against registered originals.
6. Trigger alert on match.
7. Support account disconnect at any time.

This module is explicitly optional.

## Module 5: Verification Service

### Purpose
Verify ownership using blockchain.

### Input
`suspicious_media_id`

### Process
1. Get matched original.
2. Fetch blockchain `txid`.
3. Query Hikmalayer node.
4. Confirm:
   - hash matches
   - owner matches
   - timestamp exists

### Output
`verified` / `not_verified`

## Module 6: Evidence Generator

Generate structured report:

```json
{
  "registered_txid": "...",
  "owner_public_key": "...",
  "timestamp": "...",
  "similarity_score": 0.0,
  "deepfake_probability": 0.0,
  "matched_URLs": [],
  "analysis_metadata": {},
  "model_versions": {}
}
```

### Export formats
- JSON
- PDF
- Signed digital report

## Module 7: Notification Service

Trigger channels:
- Email
- Dashboard alerts
- Webhooks
- Enterprise API callbacks

All notifications must be logged.

## 4) Hikmaon ↔ Hikmalayer Integration

### Smart transaction schema

Transaction type: `MEDIA_REGISTRATION`

```json
{
  "type": "MEDIA_REGISTRATION",
  "payload": {
    "content_hash": "...",
    "fingerprint_commitment": "...",
    "owner_pubkey": "...",
    "timestamp": "...",
    "metadata_pointer": "..."
  }
}
```

### Hikmalayer node responsibilities
- Validate transaction structure
- Add valid transaction to block
- Store in chain state
- Return `txid`

### Hikmaon backend responsibilities
- Connect to Hikmalayer RPC as client
- Submit registration transactions
- Query transactions for verification

## 5) Data Storage Design

Use three storage systems:

1. **Relational database (Postgres)**
   - Users
   - Registrations
   - Monitoring results
   - Notifications
2. **Object storage**
   - Raw media (optional)
   - Evidence files
3. **Vector database**
   - Embeddings
   - Fast similarity search

## 6) Development Order (Phased)

### Phase 1
- Registration
- Hashing
- Blockchain anchoring
- Manual verification

### Phase 2
- Embedding system
- Similarity search
- Dashboard UI

### Phase 3
- Deepfake classifier integration
- Evidence generation

### Phase 4
- Public web crawler
- Fingerprint index

### Phase 5
- Social media integrations

## 7) Security Requirements

- Encrypt uploads
- Encrypt stored embeddings
- Never store private tokens in plaintext
- Apply API rate limiting
- Log model versions for auditability

## 8) Patent Consistency Checklist

- [x] Hash + fingerprint created before blockchain anchoring
- [x] AI runs off-chain
- [x] Blockchain stores only proof
- [x] Verification queries chain
- [x] Notification is automated
- [x] Social media module is optional
- [x] Monitoring works via public indexing

## 9) Final Architecture Summary

Hikmaon =

AI + Vector Similarity + Internet Index + Blockchain Proof + Evidence Automation

Hikmaon is not only deepfake detection; it is a blockchain-verified digital authenticity engine with AI-assisted misuse discovery.
