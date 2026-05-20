from __future__ import annotations

from typing import Any

from omnia import build_boundary_certificate_from_measurement
from omnia_limit import validate_certificate
from omnia_validation.enveloper import process_boundary_step


def run_backbone_orchestration(
    measurement: dict[str, Any],
    *,
    target_repository: str = "OMNIAMIND",
    certificate_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run the canonical OMNIA backbone without redefining its contracts.

    Flow:

    measurement
      -> OMNIA build_boundary_certificate_from_measurement()
      -> omnia-limit validate_certificate()
      -> OMNIA-VALIDATION process_boundary_step()
      -> ValidationEnvelope

    OMNIAMIND only orchestrates.

    It does not define BoundaryCertificate.
    It does not define ValidationEnvelope.
    It does not validate schema directly.
    It does not decide semantic truth.
    """

    raw_certificate = build_boundary_certificate_from_measurement(
        measurement,
        target_repository=target_repository,
        certificate_id=certificate_id,
        timestamp=timestamp,
    )

    validate_certificate(raw_certificate)

    return process_boundary_step(raw_certificate)
