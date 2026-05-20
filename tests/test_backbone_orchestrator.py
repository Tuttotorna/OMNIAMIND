from omniamind import run_backbone_orchestration


def test_backbone_orchestrator_stop_flow():
    measurement = {
        "drift_score": 0.44,
        "perturbation_step": 4,
        "gate_status": "STOP",
        "omega": 0.82,
        "sei": 0.01,
        "iri": 0.99,
    }

    envelope = run_backbone_orchestration(
        measurement,
        target_repository="OMNIAMIND",
        certificate_id="omniamind-stop-cert",
        timestamp="2026-05-20T20:00:00Z",
    )

    assert envelope["validation_status"] == "GATE_CLOSED_SATURATION_REACHED"
    assert envelope["details"]["certificate_id"] == "omniamind-stop-cert"
    assert envelope["details"]["target_repository"] == "OMNIAMIND"
    assert envelope["details"]["saturation_detected"] is True
    assert envelope["details"]["ast_deformation_index"] == 0.44
    assert envelope["details"]["perturbation_step"] == 4


def test_backbone_orchestrator_continue_flow():
    measurement = {
        "delta_omega": 0.12,
        "perturbation_step": 1,
        "gate_status": "CONTINUE",
        "omega": 0.91,
        "sei": 0.20,
        "iri": 0.40,
        "reason": "Additional measurement still yields structural information",
    }

    envelope = run_backbone_orchestration(
        measurement,
        target_repository="OMNIAMIND",
        certificate_id="omniamind-continue-cert",
        timestamp="2026-05-20T20:00:00Z",
    )

    assert envelope["validation_status"] == "GATE_OPEN_MEASUREMENT_REQUIRED"
    assert envelope["details"]["certificate_id"] == "omniamind-continue-cert"
    assert envelope["details"]["target_repository"] == "OMNIAMIND"
    assert envelope["details"]["saturation_detected"] is False
    assert envelope["details"]["ast_deformation_index"] == 0.12
    assert envelope["details"]["perturbation_step"] == 1
