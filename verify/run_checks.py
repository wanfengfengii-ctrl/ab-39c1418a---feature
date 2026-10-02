"""One-shot verification service.

Waits for the ``api`` service to be healthy (Docker Compose already gates on
the health check; the wait here only adds robustness), then runs:

1. code tests             -- the pytest unit/API suite
2. application build check -- byte-compiles all sources, imports the ASGI app,
                              verifies routes and the OpenAPI schema
3. API smoke tests        -- a set of overlapping-pulse scenarios against the
                             live service (exact recovery, noisy recovery,
                             infeasible residual/event limits, validation)

The process exits by itself; the exit code summarises the results:
0 = everything passed, otherwise a bitmask
(1 = code tests failed, 2 = build check failed, 4 = smoke tests failed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

APP_DIR = os.environ.get("APP_DIR", "/app")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000").rstrip("/")

EXIT_CODE_TESTS = 1
EXIT_BUILD_CHECK = 2
EXIT_SMOKE = 4


def _banner(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def run_code_tests() -> bool:
    _banner("1/3 code tests (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=APP_DIR)
    ok = proc.returncode == 0
    print(f"code tests: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_build_check() -> bool:
    _banner("2/3 application build check")
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # let compileall write bytecode
    steps = [
        [sys.executable, "-m", "compileall", "-q", "app", "tests", "verify"],
        [
            sys.executable,
            "-c",
            "from app.main import app\n"
            "paths = {getattr(r, 'path', None) for r in app.routes}\n"
            "assert '/api/pulses/deconvolve' in paths, paths\n"
            "assert '/healthz' in paths, paths\n"
            "schema = app.openapi()\n"
            "assert '/api/pulses/deconvolve' in schema['paths']\n"
            "print('ASGI app imports; routes and OpenAPI schema OK')",
        ],
    ]
    for cmd in steps:
        proc = subprocess.run(cmd, cwd=APP_DIR, env=env)
        if proc.returncode != 0:
            print("build check: FAIL", flush=True)
            return False
    print("build check: PASS", flush=True)
    return True


def _request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        API_BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body) if body else {}
        except json.JSONDecodeError:
            return exc.code, {"raw": body}


def _convolve(events: dict[int, int], kernel: list[int], n: int) -> list[int]:
    """Independent waveform construction for smoke-test fixtures."""
    x = [0] * n
    for pos, amp in events.items():
        x[pos] = amp
    return [
        sum(kernel[j] * x[i - j] for j in range(len(kernel)) if i - j >= 0)
        for i in range(n)
    ]


class Smoke:
    """Overlapping-pulse API smoke tests against the live service."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail and not cond else ""), flush=True)
        if not cond:
            self.failures.append(name)

    @staticmethod
    def wait_healthy() -> bool:
        for _ in range(60):
            try:
                code, _ = _request("GET", "/healthz")
                if code == 200:
                    return True
            except OSError:
                pass
            time.sleep(1)
        return False

    def run(self) -> bool:
        _banner("3/3 overlapping-pulse API smoke tests")
        if not self.wait_healthy():
            self.check("api /healthz becomes healthy", False)
            return False
        self.check("api /healthz becomes healthy", True)

        kernel = [3, 2, 1]
        n = 16
        # Two events whose response tails overlap (6 - 4 < len(kernel)).
        exact_samples = _convolve({4: 5, 6: 4}, kernel, n)
        noisy_samples = list(exact_samples)
        noisy_samples[5] += 1
        noisy_samples[7] -= 1

        # Case 1: exact recovery of overlapping pulses.
        code, body = _request("POST", "/api/pulses/deconvolve", {
            "samples": exact_samples, "kernel": kernel,
            "maxResidual": 0, "maxAmplitude": 10, "maxEvents": 4,
        })
        self.check("exact overlap returns 200", code == 200, f"got {code}: {body}")
        if code == 200:
            self.check(
                "exact overlap events recovered",
                body.get("events") == [
                    {"position": 4, "amplitude": 5},
                    {"position": 6, "amplitude": 4},
                ],
                f"got {body.get('events')}",
            )
            self.check("exact overlap prediction matches samples",
                       body.get("prediction") == exact_samples)
            self.check("exact overlap residuals all zero",
                       body.get("residuals") == [0] * n)
            self.check(
                "exact overlap objectives",
                body.get("objectives") == {
                    "maxAbsResidual": 0, "sumAbsResidual": 0, "eventCount": 2,
                },
                f"got {body.get('objectives')}",
            )

        # Case 2: noisy overlapping pulses within the residual limit.
        code, body = _request("POST", "/api/pulses/deconvolve", {
            "samples": noisy_samples, "kernel": kernel,
            "maxResidual": 1, "maxAmplitude": 10, "maxEvents": 4,
        })
        self.check("noisy overlap returns 200", code == 200, f"got {code}: {body}")
        if code == 200:
            objectives = body.get("objectives", {})
            prediction = body.get("prediction", [])
            residuals = body.get("residuals", [])
            self.check("noisy overlap respects residual limit",
                       objectives.get("maxAbsResidual", 99) <= 1,
                       f"got {objectives}")
            self.check("noisy overlap event count within limit",
                       objectives.get("eventCount", 99) <= 4,
                       f"got {objectives}")
            self.check("noisy overlap prediction length",
                       len(prediction) == n)
            self.check(
                "noisy overlap residuals consistent with prediction",
                len(residuals) == n
                and all(residuals[i] == noisy_samples[i] - prediction[i] for i in range(n)),
            )

        # Case 3: residual limit cannot be met -> identifiable infeasible result.
        code, body = _request("POST", "/api/pulses/deconvolve", {
            "samples": noisy_samples, "kernel": kernel,
            "maxResidual": 0, "maxAmplitude": 10, "maxEvents": 4,
        })
        self.check("tight residual limit returns 422", code == 422, f"got {code}: {body}")
        self.check("tight residual limit flagged infeasible",
                   body.get("status") == "infeasible", f"got {body}")
        self.check("tight residual limit first unexplainable position",
                   body.get("firstUnexplainablePosition") == 5,
                   f"got {body.get('firstUnexplainablePosition')}")

        # Case 4: event-count limit cannot be met -> identifiable infeasible result.
        code, body = _request("POST", "/api/pulses/deconvolve", {
            "samples": exact_samples, "kernel": kernel,
            "maxResidual": 0, "maxAmplitude": 10, "maxEvents": 1,
        })
        self.check("event limit returns 422", code == 422, f"got {code}: {body}")
        self.check("event limit flagged infeasible",
                   body.get("status") == "infeasible", f"got {body}")
        self.check("event limit first unexplainable position",
                   body.get("firstUnexplainablePosition") == 6,
                   f"got {body.get('firstUnexplainablePosition')}")

        # Case 5: request validation.
        code, _ = _request("POST", "/api/pulses/deconvolve", {
            "samples": exact_samples[:11], "kernel": kernel,
            "maxResidual": 0, "maxAmplitude": 10, "maxEvents": 4,
        })
        self.check("too few samples rejected with 400", code == 400, f"got {code}")
        code, _ = _request("POST", "/api/pulses/deconvolve", {
            "samples": exact_samples, "kernel": [1, 1, 1, 1, 1, 1, 1],
            "maxResidual": 0, "maxAmplitude": 10, "maxEvents": 4,
        })
        self.check("oversized kernel rejected with 400", code == 400, f"got {code}")

        ok = not self.failures
        print(f"smoke tests: {'PASS' if ok else 'FAIL'}", flush=True)
        return ok


def main() -> int:
    code = 0
    if not run_code_tests():
        code |= EXIT_CODE_TESTS
    if not run_build_check():
        code |= EXIT_BUILD_CHECK
    if not Smoke().run():
        code |= EXIT_SMOKE
    _banner("summary")
    print(f"code tests  : {'PASS' if not code & EXIT_CODE_TESTS else 'FAIL'}", flush=True)
    print(f"build check : {'PASS' if not code & EXIT_BUILD_CHECK else 'FAIL'}", flush=True)
    print(f"api smoke   : {'PASS' if not code & EXIT_SMOKE else 'FAIL'}", flush=True)
    print(f"verify exit code: {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
