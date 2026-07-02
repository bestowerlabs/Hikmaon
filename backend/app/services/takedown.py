"""Owner consent and takedown enforcement.

Implements the consent-driven response flow:

    incident (pending_owner_review)
        └─ owner ALLOWS  -> allowed_by_owner (logged, case closed)
        └─ owner REMOVES -> removal_requested
                            └─ TakedownCase: DMCA-style notice generated with
                               the blockchain Certificate of Ownership attached,
                               tracked open -> reported -> removed/rejected

Hikmaon automates the *filing* and evidence packaging; the hosting platform
makes the final removal decision. Platform-specific report submission (via
each provider's abuse/DMCA API) plugs into `_submit_platform_reports`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import IncidentDecision, IncidentRecord, TakedownCase, TakedownStatus
from app.services.notification import NotificationService
from app.storage import InMemoryStore

DMCA_NOTICE_TEMPLATE = """NOTICE OF CLAIMED INFRINGEMENT / UNAUTHORIZED SYNTHETIC MEDIA

To whom it may concern,

This notice concerns media published at the following location(s):
{urls}

The undersigned rights holder ("{owner_id}") states that the published media
is derived from — or is a manipulated/synthetic version of — an original work
registered on the Hikmalayer blockchain prior to the publication above.

Match evidence:
  - Perceptual match to registered original: {match_percentage}%
  - Manipulation analysis verdict: {manipulation_verdict}
  - Blockchain registration txid: {txid}
  - Certificate of Ownership: {certificate_id}
  - Evidence report: {evidence_report_id}

The rights holder has NOT consented to this publication and requests its
expeditious removal. The blockchain-anchored registration (timestamped,
immutable) is available for independent verification.

Issued via the Hikmaon authenticity platform on {issued_at}.
"""


class TakedownService:
    def __init__(self, store: InMemoryStore, notification_service: NotificationService) -> None:
        self.store = store
        self.notification_service = notification_service

    def apply_owner_decision(self, incident: IncidentRecord, decision: IncidentDecision) -> dict:
        if incident.status != "pending_owner_review":
            raise ValueError(f"Incident already decided (status={incident.status})")

        incident.owner_decision_at = datetime.now(tz=timezone.utc)

        if decision.decision == "allow":
            incident.status = "allowed_by_owner"
            self.notification_service.notify(
                channel="dashboard",
                recipient=incident.notified_owner,
                message=f"Incident {incident.incident_id}: publication allowed by owner. Case closed.",
            )
            self.store.persist()
            return {"incident_id": incident.incident_id, "status": incident.status}

        incident.status = "removal_requested"
        case = self._open_takedown_case(incident, decision)
        incident.takedown_case_id = case.case_id
        self.store.persist()
        return {
            "incident_id": incident.incident_id,
            "status": incident.status,
            "takedown_case": case.model_dump(),
        }

    def _open_takedown_case(self, incident: IncidentRecord, decision: IncidentDecision) -> TakedownCase:
        registration = self.store.registrations.get(incident.matched_media_id)
        certificate_id = registration.certificate_id if registration else None
        owner_id = registration.owner_id if registration else incident.notified_owner
        txid = registration.blockchain_txid if registration else "unknown"

        urls = incident.matched_urls or ["<no indexed URL - manual target entry required>"]
        notice = DMCA_NOTICE_TEMPLATE.format(
            urls="\n".join(f"  - {url}" for url in urls),
            owner_id=owner_id,
            match_percentage=incident.match_percentage,
            manipulation_verdict=incident.manipulation_verdict,
            txid=txid,
            certificate_id=certificate_id or "n/a",
            evidence_report_id=incident.evidence_report_id,
            issued_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        case = TakedownCase(
            case_id=f"case_{uuid.uuid4().hex[:12]}",
            incident_id=incident.incident_id,
            matched_media_id=incident.matched_media_id,
            owner_id=owner_id,
            target_urls=urls,
            notice_text=notice,
            certificate_id=certificate_id,
            status="open",
            status_history=[{"status": "open", "at": datetime.now(tz=timezone.utc).isoformat(), "note": decision.reason or "owner refused consent"}],
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.takedown_cases[case.case_id] = case

        self._submit_platform_reports(case)
        self.notification_service.notify(
            channel="dashboard",
            recipient=owner_id,
            message=f"Takedown case {case.case_id} filed for incident {incident.incident_id} ({len(urls)} target URL(s)).",
        )
        return case

    def _submit_platform_reports(self, case: TakedownCase) -> None:
        """Submit the notice to hosting platforms.

        Production integration point: map each target URL to its platform's
        abuse/DMCA reporting API (or generate the email/web-form filing) and
        record the platform's ticket id. Until those integrations are
        configured, the case is marked reported with the generated notice
        ready to send.
        """
        self._transition(case, "reported", "notice generated and queued for platform submission")

    def update_case_status(self, case_id: str, status: TakedownStatus, note: str | None = None) -> TakedownCase:
        case = self.store.takedown_cases.get(case_id)
        if not case:
            raise KeyError("case_not_found")
        self._transition(case, status, note or "")
        incident = self.store.incidents.get(case.incident_id)
        if incident and status in ("removed", "rejected"):
            incident.status = "closed"
        self.store.persist()
        return case

    def _transition(self, case: TakedownCase, status: TakedownStatus, note: str) -> None:
        case.status = status
        case.status_history.append(
            {"status": status, "at": datetime.now(tz=timezone.utc).isoformat(), "note": note}
        )
