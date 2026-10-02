"""Joint non-negative integer deconvolution of overlapping scintillator pulses.

Model
-----
Observed samples ``s[0..n-1]`` are modelled as the first ``n`` outputs of the
zero-padded discrete convolution of an event amplitude train ``x[0..n-1]``
(non-negative integers, each at most ``max_amplitude``) with a positive
integer response kernel ``k[0..m-1]``::

    pred[i] = sum_{j=0}^{m-1} k[j] * x[i-j]        (x[t] = 0 for t < 0)

An *event* is a position whose amplitude is non-zero.  Residuals are
``residuals[i] = samples[i] - prediction[i]``.

Objectives (minimised lexicographically, in this order)
-------------------------------------------------------
1. ``max  |residual[i]|``   -- hard limit ``max_residual``; otherwise infeasible
2. ``sum  |residual[i]|``
3. number of events          -- hard limit ``max_events``; otherwise infeasible
4. the amplitude sequence ``x`` itself, in lexicographic order

Algorithm
---------
The convolution matrix is banded lower-triangular: once ``x[0..p-1]`` are
fixed, sample ``p`` depends only on the last ``m-1`` amplitudes.  For a given
per-point residual radius the admissible values of ``x[p]`` form a contiguous
integer interval, so the joint optimum is found by an exact dynamic program
over "tail" states (the last ``m-1`` amplitudes plus the events used so far)
instead of greedy peak-by-peak tail subtraction:

* a feasibility scan finds, for a given radius, whether any complete candidate
  satisfies the limits, and the first sample position no complete candidate
  can explain;
* a binary search over the (integer) radius yields the minimal achievable
  max |residual|;
* a final dynamic program minimises (sum |residual|, events, amplitude
  sequence) under that radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class SolveResult:
    feasible: bool
    amplitudes: Optional[list[int]] = None
    prediction: Optional[list[int]] = None
    residuals: Optional[list[int]] = None
    max_abs_residual: Optional[int] = None
    sum_abs_residual: Optional[int] = None
    event_count: Optional[int] = None
    first_unexplainable_position: Optional[int] = None


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for integers with b > 0 (handles negative a)."""
    return -((-a) // b)


def _allowed_range(
    sample: int, radius: int, tail_sum: int, k0: int, max_amplitude: int
) -> tuple[int, int]:
    """Integer amplitudes v in [0, max_amplitude] with |k0*v + tail_sum - sample| <= radius."""
    lo = _ceil_div(sample - radius - tail_sum, k0)
    hi = (sample + radius - tail_sum) // k0
    if lo < 0:
        lo = 0
    if hi > max_amplitude:
        hi = max_amplitude
    return lo, hi


def _tail_weights(kernel: Sequence[int]) -> list[int]:
    """Weights of the tail state: tail[i] holds x[p-m+1+i], contributing kernel[m-1-i]."""
    m = len(kernel)
    return [kernel[m - 1 - i] for i in range(m - 1)]


def _scan(
    samples: Sequence[int],
    kernel: Sequence[int],
    radius: int,
    max_amplitude: int,
    max_events: int,
) -> tuple[bool, Optional[int]]:
    """Feasibility and first unexplainable position for a per-point residual ``radius``.

    Returns ``(feasible, first_fail)``.  ``first_fail`` is the smallest sample
    index ``p`` for which no complete candidate (amplitudes within
    ``[0, max_amplitude]``, at most ``max_events`` non-zero amplitudes) can
    keep every residual of samples ``0..p`` within ``+/-radius``; ``None``
    when all samples can be covered.
    """
    n = len(samples)
    m = len(kernel)
    k0 = kernel[0]
    w = _tail_weights(kernel)
    # states: (tail tuple of the last m-1 amplitudes, events used so far)
    states = {((0,) * (m - 1), 0)}
    for p in range(n):
        sp = samples[p]
        nxt: set[tuple[tuple[int, ...], int]] = set()
        for tail, used in states:
            tail_sum = 0
            for i in range(m - 1):
                ti = tail[i]
                if ti:
                    tail_sum += w[i] * ti
            lo, hi = _allowed_range(sp, radius, tail_sum, k0, max_amplitude)
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                nxt.add((tail[1:] + (v,), used2))
        if not nxt:
            return False, p
        states = nxt
    return True, None


def _optimize(
    samples: Sequence[int],
    kernel: Sequence[int],
    radius: int,
    max_amplitude: int,
    max_events: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Minimise (sum |residual|, events, amplitude sequence) with |residual| <= radius."""
    n = len(samples)
    m = len(kernel)
    k0 = kernel[0]
    w = _tail_weights(kernel)
    # (tail, events used) -> (sum |residual| so far, amplitude path so far)
    states: dict[tuple[tuple[int, ...], int], tuple[int, tuple[int, ...]]] = {
        ((0,) * (m - 1), 0): (0, ())
    }
    for p in range(n):
        sp = samples[p]
        nxt: dict[tuple[tuple[int, ...], int], tuple[int, tuple[int, ...]]] = {}
        for (tail, used), (cost, path) in states.items():
            tail_sum = 0
            for i in range(m - 1):
                ti = tail[i]
                if ti:
                    tail_sum += w[i] * ti
            lo, hi = _allowed_range(sp, radius, tail_sum, k0, max_amplitude)
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                key = (tail[1:] + (v,), used2)
                cand = (cost + abs(k0 * v + tail_sum - sp), path + (v,))
                cur = nxt.get(key)
                if cur is None or cand < cur:
                    nxt[key] = cand
        if not nxt:  # radius below the feasible minimum; solve() prevents this
            raise RuntimeError("no candidate within the given residual radius")
        states = nxt
    return min(
        (cost, used, path) for (tail, used), (cost, path) in states.items()
    )


def solve(
    samples: Sequence[int],
    kernel: Sequence[int],
    max_residual: int,
    max_amplitude: int,
    max_events: int,
) -> SolveResult:
    """Jointly deconvolve ``samples`` under the given limits.

    Returns a :class:`SolveResult`; when infeasible, only ``feasible`` and
    ``first_unexplainable_position`` are populated.
    """
    n = len(samples)
    m = len(kernel)

    feasible, first_fail = _scan(samples, kernel, max_residual, max_amplitude, max_events)
    if not feasible:
        return SolveResult(feasible=False, first_unexplainable_position=first_fail)

    # Samples, kernel and amplitudes are integers, so residuals are integers
    # and the optimal max |residual| is an integer in [0, max_residual].
    lo, hi = -1, max_residual  # hi feasible, lo infeasible
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, _ = _scan(samples, kernel, mid, max_amplitude, max_events)
        if ok:
            hi = mid
        else:
            lo = mid
    radius = hi

    sum_abs, event_count, path = _optimize(samples, kernel, radius, max_amplitude, max_events)
    amplitudes = list(path)
    prediction = []
    for i in range(n):
        acc = 0
        for j in range(m):
            t = i - j
            if t >= 0:
                acc += kernel[j] * amplitudes[t]
        prediction.append(acc)
    residuals = [samples[i] - prediction[i] for i in range(n)]
    max_abs = max(abs(r) for r in residuals)
    return SolveResult(
        feasible=True,
        amplitudes=amplitudes,
        prediction=prediction,
        residuals=residuals,
        max_abs_residual=max_abs,
        sum_abs_residual=sum_abs,
        event_count=event_count,
    )
