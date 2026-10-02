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


CLIPPED_SAMPLES = [0, 0, 0, 0, 15, 10, 20, 12, 6, 0, 0, 0]


def test_legacy_response_shape_unchanged_without_clipping_fields():
    resp = client.post("/api/pulses/deconvolve", json=make_payload())
    body = resp.json()
    assert set(body) == {"status", "events", "amplitudes", "prediction", "residuals", "objectives"}


def test_clipped_overlap_recovers_unclipped_prediction():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=CLIPPED_SAMPLES,
            saturationLevel=20,
            totalAmplitudeRange=[11, 11],
        ),
    )
    assert resp.status_code == 200, f"{resp.json()}"
    body = resp.json()
    assert body["events"] == [
        {"position": 4, "amplitude": 5},
        {"position": 6, "amplitude": 6},
    ]
    assert body["prediction"][6] == 23
    assert body["residuals"][6] == -3
    assert body["totalAmplitude"] == 11
    assert body["totalAmplitudeRange"] == [11, 11]
    assert body["saturationEvidence"] == [
        {"position": 6, "observed": 20, "predicted": 23, "excess": 3}
    ]
    assert body["objectives"]["maxAbsResidual"] == 0


def test_clipped_total_conflict_returns_identifiable_422():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=CLIPPED_SAMPLES,
            saturationLevel=20,
            totalAmplitudeRange=[1, 2],
        ),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["constraint"] == "totalAmplitudeRange"
    assert body["totalAmplitudeRange"] == [1, 2]
    assert "firstUnexplainablePosition" in body


def test_sample_above_saturation_is_invalid():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=CLIPPED_SAMPLES,
            saturationLevel=15,  # samples contain 20 > 15
            totalAmplitudeRange=[11, 11],
        ),
    )
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid_request"


@pytest.mark.parametrize(
    "overrides",
    [
        {"saturationLevel": 15},                                  # range missing
        {"totalAmplitudeRange": [5, 9]},                          # level missing
        {"saturationLevel": 15, "totalAmplitudeRange": [9, 5]},   # lo > hi
        {"saturationLevel": 15, "totalAmplitudeRange": [5]},      # not a pair
        {"saturationLevel": 15, "totalAmplitudeRange": [-1, 5]},  # negative lo
    ],
)
def test_clipping_field_combinations_invalid(overrides):
    resp = client.post(
        "/api/pulses/deconvolve", json=make_payload(**overrides)
    )
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid_request"


def test_clipping_legacy_infeasible_body_keeps_shape():
    # A residual-limit infeasibility outside clipping mode stays unchanged.
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(samples=NOISY_SAMPLES, maxResidual=0),
    )
    body = resp.json()
    assert set(body) == {"status", "detail", "firstUnexplainablePosition"}


def test_underconstrained_clipping_reports_budget():
    # Every sample at the plateau with loose event/total limits leaves too
    # many admissible trains for exact joint inversion within the budget.
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(
            samples=[15] * 16,
            saturationLevel=15,
            totalAmplitudeRange=[0, 300],
            maxAmplitude=1000,
            maxEvents=16,
            maxResidual=0,
        ),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["constraint"] == "stateBudgetExceeded"
    assert isinstance(body["firstUnexplainablePosition"], int)
    assert body["totalAmplitudeRange"] == [0, 300]
