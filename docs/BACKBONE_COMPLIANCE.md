# OMNIAMIND Backbone Compliance

## Role

OMNIAMIND is an orchestrator.

It is not the measurement engine.

It is not the boundary validator.

It is not the validation control plane.

It is not a decision engine.

## Canonical flow

OMNIAMIND must orchestrate the existing backbone:

OMNIAMIND measurement input
  -> OMNIA build_boundary_certificate_from_measurement()
  -> BoundaryCertificate
  -> omnia-limit validate_certificate()
  -> OMNIA-VALIDATION process_boundary_step()
  -> ValidationEnvelope

## Public API

OMNIAMIND exposes:

run_backbone_orchestration(...)

## Contract rule

OMNIAMIND does not redefine BoundaryCertificate.

OMNIAMIND does not redefine ValidationEnvelope.

OMNIAMIND does not bypass omnia-limit.

OMNIAMIND does not bypass OMNIA-VALIDATION.

## Boundary

measurement != validation
validation != orchestration
orchestration != decision
decision != measurement

OMNIAMIND stays in the orchestration layer.
