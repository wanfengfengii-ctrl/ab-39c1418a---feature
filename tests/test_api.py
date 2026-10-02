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
