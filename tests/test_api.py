"""API tests for POST /api/pulses/deconvolve and the health endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

KERNEL = [3, 2, 1]
EXACT_SAMPLES = [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0]
NOISY_SAMPLES = [0, 0, 0, 0, 15, 11, 17, 7, 4, 0, 0, 0, 0, 0, 0, 0]


def make_payload(**overrides):
    payload = {
        "samples": EXACT_SAMPLES,
        "kernel": KERNEL,
        "maxResidual": 0,
        "maxAmplitude": 10,
        "maxEvents": 4,
    }
    payload.update(overrides)
    return payload


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_exact_overlapping_pulses():
    resp = client.post("/api/pulses/deconvolve", json=make_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["events"] == [
        {"position": 4, "amplitude": 5},
        {"position": 6, "amplitude": 4},
    ]
    assert body["amplitudes"] == [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert body["prediction"] == EXACT_SAMPLES
    assert body["residuals"] == [0] * 16
    assert body["objectives"] == {
        "maxAbsResidual": 0,
        "sumAbsResidual": 0,
        "eventCount": 2,
    }


def test_noisy_pulses_within_residual_limit():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(samples=NOISY_SAMPLES, maxResidual=1),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["objectives"]["maxAbsResidual"] <= 1
    assert len(body["prediction"]) == len(NOISY_SAMPLES)
    assert body["residuals"] == [
        NOISY_SAMPLES[i] - body["prediction"][i] for i in range(len(NOISY_SAMPLES))
    ]


def test_infeasible_residual_limit_is_identifiable():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(samples=NOISY_SAMPLES, maxResidual=0),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["firstUnexplainablePosition"] == 5


def test_infeasible_event_limit_is_identifiable():
    resp = client.post("/api/pulses/deconvolve", json=make_payload(maxEvents=1))
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["firstUnexplainablePosition"] == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples": EXACT_SAMPLES[:11]},            # too few samples
        {"samples": EXACT_SAMPLES + [0] * 21},      # too many samples (37)
        {"kernel": [2, 1]},                         # kernel too short
        {"kernel": [1, 1, 1, 1, 1, 1, 1]},          # kernel too long
        {"kernel": [3, 0, 1]},                      # non-positive kernel entry
        {"kernel": [3, -2, 1]},                     # negative kernel entry
        {"maxResidual": -1},                        # negative residual limit
        {"maxAmplitude": 0},                        # amplitude limit must be >= 1
        {"maxEvents": -1},                          # negative event limit
        {"samples": ["a"] + EXACT_SAMPLES[1:]},     # non-integer sample
    ],
)
def test_invalid_requests_return_400(overrides):
    resp = client.post("/api/pulses/deconvolve", json=make_payload(**overrides))
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid_request"


def test_missing_field_returns_400():
    payload = make_payload()
    del payload["kernel"]
    resp = client.post("/api/pulses/deconvolve", json=payload)
    assert resp.status_code == 400


def test_deterministic_responses():
    first = client.post("/api/pulses/deconvolve", json=make_payload(maxResidual=1, samples=NOISY_SAMPLES))
    second = client.post("/api/pulses/deconvolve", json=make_payload(maxResidual=1, samples=NOISY_SAMPLES))
    assert first.status_code == 200
    assert first.json() == second.json()


# --- Strong pulses: saturationLevel / totalAmplitudeRange -------------------

CLIP_LEVEL = 12
CLIP_SAMPLES = [0, 0, 0, 0, 12, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CLIP2_SAMPLES = [0, 0, 0, 0, 12, 10, 12, 8, 4, 0, 0, 0, 0, 0, 0, 0]


def test_legacy_response_has_no_saturation_block():
    resp = client.post("/api/pulses/deconvolve", json=make_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "saturation" not in body


def test_saturated_pulse_recovers_unclipped_with_evidence():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(samples=CLIP_SAMPLES, saturationLevel=CLIP_LEVEL),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["events"] == [{"position": 4, "amplitude": 5}]
    # Full unclipped prediction is retained, even above the platform.
    assert body["prediction"] == [0, 0, 0, 0, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert body["amplitudes"] == [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert body["objectives"] == {"maxAbsResidual": 0, "sumAbsResidual": 0, "eventCount": 1}
    saturation = body["saturation"]
    assert saturation == {
        "totalAmplitude": 5,
        "saturatedPoints": [
            {"position": 4, "sample": 12, "prediction": 15, "excess": 3},
        ],
    }
    # Non-clipped residuals remain samples - prediction; the clipped point's
    # raw residual is still reported for transparency.
    assert body["residuals"] == [0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_saturated_overlapping_pulses_with_total_range():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=CLIP2_SAMPLES,
            saturationLevel=CLIP_LEVEL,
            totalAmplitudeRange=[9, 9],
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["events"] == [
        {"position": 4, "amplitude": 5},
        {"position": 6, "amplitude": 4},
    ]
    assert body["prediction"] == [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0]
    saturation = body["saturation"]
    assert saturation["totalAmplitude"] == 9
    assert saturation["saturatedPoints"] == [
        {"position": 4, "sample": 12, "prediction": 15, "excess": 3},
        {"position": 6, "sample": 12, "prediction": 17, "excess": 5},
    ]


def test_total_range_conflict_is_identifiable_and_echoes_range():
    # Clip forces x[4] >= 4; total <= 4 lets x[4] = 4 survive the clipped
    # point, but sample 5 (= 10) then has no extendible candidate.
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=CLIP_SAMPLES,
            saturationLevel=CLIP_LEVEL,
            totalAmplitudeRange=[0, 4],
        ),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["firstUnexplainablePosition"] == 5
    assert body["totalAmplitudeRange"] == [0, 4]


def test_total_range_lower_bound_conflict_is_identifiable():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=CLIP_SAMPLES,
            saturationLevel=CLIP_LEVEL,
            totalAmplitudeRange=[9, 20],
        ),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    # Candidates keep dying as the required total cannot be accumulated; the
    # first point without an extendible candidate is the last sample.
    assert body["firstUnexplainablePosition"] == 15
    assert body["totalAmplitudeRange"] == [9, 20]


@pytest.mark.parametrize(
    "overrides",
    [
        # A sample strictly above the platform is physically impossible.
        {"samples": [0, 0, 0, 0, 13, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         "saturationLevel": 12},
        # Total range requires saturation semantics.
        {"totalAmplitudeRange": [1, 9]},
        # Malformed intervals.
        {"saturationLevel": 12, "totalAmplitudeRange": [5]},
        {"saturationLevel": 12, "totalAmplitudeRange": [9, 5]},
        {"saturationLevel": 12, "totalAmplitudeRange": [-1, 9]},
        {"saturationLevel": -1},
    ],
)
def test_saturation_validation_returns_400(overrides):
    resp = client.post("/api/pulses/deconvolve", json=make_payload(**overrides))
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid_request"


def test_saturated_response_is_deterministic():
    payload = make_payload(
        samples=CLIP2_SAMPLES,
        saturationLevel=CLIP_LEVEL,
        totalAmplitudeRange=[9, 12],
    )
    first = client.post("/api/pulses/deconvolve", json=payload)
    second = client.post("/api/pulses/deconvolve", json=payload)
    assert first.status_code == 200
    assert first.json() == second.json()
