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

Saturation (strong pulses that clip the ADC)
--------------------------------------------
When ``saturation_level`` is given, a sample equal to it is a *clipped*
reading: it is only known that the true prediction reaches the platform, i.e.
``pred[i] >= saturation_level``.  Samples strictly below the platform keep
their usual two-sided residual ``|s[i] - pred[i]| <= radius`` and are the
only points contributing to the residual objectives; a sample above the
platform can never occur and is rejected before solving.  When
``total_amplitude_range`` ``[total_lo, total_hi]`` is additionally given, the
sum of all amplitudes must lie in that closed interval -- an independent
integrating readout constraining the clipped waveform.

Objectives (minimised lexicographically, in this order)
-------------------------------------------------------
1. ``max  |residual[i]|``   -- hard limit ``max_residual``; otherwise infeasible
2. ``sum  |residual[i]|``   (clipped points contribute no residual)
3. number of events          -- hard limit ``max_events``; otherwise infeasible
4. the amplitude sequence ``x`` itself, in lexicographic order

The clipping semantics, the event-count limit and the total-amplitude
interval are part of *one* dynamic program: feasibility is never decided on
the clipped waveform first and filtered by the total afterwards.

Algorithm
---------
The convolution matrix is banded lower-triangular: once ``x[0..p-1]`` are
fixed, sample ``p`` depends only on the last ``m-1`` amplitudes.  For a given
per-point residual radius the admissible values of ``x[p]`` form a contiguous
integer interval (for clipped points it is a one-sided interval, also
contiguous), so the joint optimum is found by an exact dynamic program over
"tail" states (the last ``m-1`` amplitudes and, when a total range is given,
the amplitude sum used so far -- with the events used carried as a frontier
value rather than enumerated as another state dimension) instead of greedy
peak-by-peak tail subtraction:

* a feasibility scan finds, for a given radius, whether any complete
  candidate satisfies every limit, and the first sample position no
  extendible complete candidate can explain;
* an exponential radius probe (radius 0 first -- the common exact case --
  then doubling towards the limit) followed by a binary search yields the
  minimal achievable max |residual|; infeasible probes are cheap because the
  scan stops at the first unexplainable position;
* a final dynamic program minimises (sum |residual|, events, amplitude
  sequence) under that radius, merging dominated prefixes online (per
  tail/sum state only the (events, cost) Pareto frontier survives) and
  tracing the chosen sequence through parent links rather than copied paths.

States that can no longer be extended into the total-amplitude interval with
the remaining positions and the remaining event budget are dropped during the
scan, so the reported first-unexplainable position accounts for the
total-amplitude constraint as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class SaturatedPoint:
    """Evidence for one clipped sample: the unclipped prediction tops the platform."""

    position: int
    sample: int
    prediction: int

    @property
    def excess(self) -> int:
        """Amount by which the (unclipped) prediction exceeds the reading."""
        return self.prediction - self.sample


@dataclass
class SaturationEvidence:
    """Evidence for a successful solve with saturation semantics enabled."""

    total_amplitude: int
    saturated_points: list[SaturatedPoint]


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
    saturation_evidence: Optional[SaturationEvidence] = None


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for integers with b > 0 (handles negative a)."""
    return -((-a) // b)


def _tail_weights(kernel: Sequence[int]) -> list[int]:
    """Weights of the tail state: tail[i] holds x[p-m+1+i], contributing kernel[m-1-i]."""
    m = len(kernel)
    return [kernel[m - 1 - i] for i in range(m - 1)]


def _tail_sum(tail: Sequence[int], w: Sequence[int]) -> int:
    total = 0
    for i in range(len(tail)):
        ti = tail[i]
        if ti:
            total += w[i] * ti
    return total


def _scan(
    samples: Sequence[int],
    kernel: Sequence[int],
    radius: int,
    max_amplitude: int,
    max_events: int,
    clipped_mask: Sequence[bool],
    total_lo: int,
    total_hi: int,
    constrain_total: bool,
) -> tuple[bool, Optional[int]]:
    """Feasibility and first unexplainable position for a per-point residual ``radius``.

    Returns ``(feasible, first_fail)``.  ``first_fail`` is the smallest sample
    index ``p`` for which no extendible complete candidate (amplitudes within
    ``[0, max_amplitude]``, at most ``max_events`` non-zero amplitudes and, when
    constrained, total amplitude in ``[total_lo, total_hi]``) can keep every
    residual of samples ``0..p`` within the per-point semantics and
    ``+/-radius`` at non-clipped points; ``None`` when all samples can be
    covered.

    States are ``(tail, amplitude sum) -> minimum events used``.  For an
    identical tail and running sum the prefix with fewer events leaves a
    superset of every feasible completion (the remaining event budget is the
    only thing the event count affects), so it dominates the others and the
    event-count dimension never needs to be enumerated.
    """
    n = len(samples)
    m = len(kernel)
    k0 = kernel[0]
    w = _tail_weights(kernel)
    # states: (tail tuple of the last m-1 amplitudes, amplitude sum) -> min events
    states: dict[tuple[tuple[int, ...], int], int] = {((0,) * (m - 1), 0): 0}
    # tail contribution at the current layer, cached per distinct tail.
    ts_cache: dict[tuple[int, ...], int] = {(0,) * (m - 1): 0}
    for p in range(n):
        sp = samples[p]
        clipped = clipped_mask[p]
        remaining = n - p - 1
        nxt: dict[tuple[tuple[int, ...], int], int] = {}
        nxt_cache: dict[tuple[int, ...], int] = {}
        for (tail, amp_sum), used in states.items():
            ts = ts_cache[tail]
            if clipped:
                lo = _ceil_div(sp - ts, k0)
                if lo < 0:
                    lo = 0
                hi = max_amplitude
            else:
                lo = _ceil_div(sp - radius - ts, k0)
                if lo < 0:
                    lo = 0
                hi = (sp + radius - ts) // k0
                if hi > max_amplitude:
                    hi = max_amplitude
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                if constrain_total:
                    sum2 = amp_sum + v
                    if sum2 > total_hi:
                        continue
                    deficit = total_lo - sum2
                    if deficit > 0:
                        if deficit > max_amplitude * remaining:
                            continue
                        # ceil(deficit / max_amplitude) further events needed
                        if used2 + (-((-deficit) // max_amplitude)) > max_events:
                            continue
                else:
                    sum2 = 0
                new_tail = tail[1:] + (v,)
                key = (new_tail, sum2)
                cur = nxt.get(key)
                if cur is None or used2 < cur:
                    nxt[key] = used2
                    if new_tail not in nxt_cache:
                        nxt_cache[new_tail] = _tail_sum(new_tail, w)
        if not nxt:
            return False, p
        states = nxt
        ts_cache = nxt_cache
    return True, None


def _optimize(
    samples: Sequence[int],
    kernel: Sequence[int],
    radius: int,
    max_amplitude: int,
    max_events: int,
    clipped_mask: Sequence[bool],
    total_lo: int,
    total_hi: int,
    constrain_total: bool,
) -> tuple[int, int, tuple[int, ...]]:
    """Minimise (sum |residual|, events, amplitude sequence) under the limits.

    Clipped points contribute no residual (only the one-sided platform
    constraint).  Final states must have their amplitude sum in
    ``[total_lo, total_hi]`` when a total range is given.

    For each ``(tail, amplitude sum)`` only a Pareto frontier over
    ``(events used, residual cost)`` is kept: a point is dropped when another
    with no more events and no larger cost exists, since neither objective can
    improve with future choices.  Amplitude paths are stored as parent links
    (node ``v`` at layer ``p`` points at a surviving node of layer ``p-1``)
    instead of copied prefixes; each layer's surviving nodes are ranked by the
    lexicographic order of their full prefix path.  For two children the path
    order is exactly the order of ``(parent rank, v)``, which also settles the
    objective-4 tie break cheaply.
    """
    n = len(samples)
    m = len(kernel)
    k0 = kernel[0]
    w = _tail_weights(kernel)

    # A surviving node: (events used, residual cost, v at this layer,
    # index into the previous layer's node table, lexicographic path rank).
    # ``layers[p]`` holds the node table for layer p (p = number of fixed
    # amplitudes); layer 0 is the single root node.
    root = (0, 0, 0, -1, 0)
    layers: list[list[tuple[int, int, int, int, int]]] = [[root]]
    # states: (tail, amplitude sum) -> indices into this layer's node table
    states: dict[tuple[tuple[int, ...], int], list[int]] = {
        ((0,) * (m - 1), 0): [0]
    }
    # Tail contribution at the current layer, cached per distinct tail.
    ts_cache: dict[tuple[int, ...], int] = {(0,) * (m - 1): 0}
    for p in range(n):
        sp = samples[p]
        clipped = clipped_mask[p]
        remaining = n - p - 1
        prev_table = layers[p]
        # For each successor state key, merge candidates online keyed by events
        # used: with an identical event count only the lexicographically
        # smallest (cost, parent path rank, v) can survive, so one entry per
        # (state, events) suffices.  Entry: (cost, parent rank, v, parent id).
        nxt_best: dict[
            tuple[tuple[int, ...], int], dict[int, tuple[int, int, int, int]]
        ] = {}
        nxt_cache: dict[tuple[int, ...], int] = {}
        for (tail, amp_sum), node_ids in states.items():
            ts = ts_cache[tail]
            if clipped:
                lo = _ceil_div(sp - ts, k0)
                if lo < 0:
                    lo = 0
                hi = max_amplitude
            else:
                lo = _ceil_div(sp - radius - ts, k0)
                if lo < 0:
                    lo = 0
                hi = (sp + radius - ts) // k0
                if hi > max_amplitude:
                    hi = max_amplitude
            for nid in node_ids:
                pnode = prev_table[nid]
                used, cost, parent_rank = pnode[0], pnode[1], pnode[4]
                for v in range(lo, hi + 1):
                    used2 = used + (1 if v else 0)
                    if used2 > max_events:
                        continue
                    if constrain_total:
                        sum2 = amp_sum + v
                        if sum2 > total_hi:
                            continue
                        deficit = total_lo - sum2
                        if deficit > 0:
                            if deficit > max_amplitude * remaining:
                                continue
                            if used2 + (-((-deficit) // max_amplitude)) > max_events:
                                continue
                    else:
                        sum2 = 0
                    point_cost = 0 if clipped else abs(k0 * v + ts - sp)
                    new_tail = tail[1:] + (v,)
                    skey = (new_tail, sum2)
                    per_key = nxt_best.get(skey)
                    cand = (cost + point_cost, parent_rank, v, nid)
                    if per_key is None:
                        nxt_best[skey] = {used2: cand}
                        if new_tail not in nxt_cache:
                            nxt_cache[new_tail] = _tail_sum(new_tail, w)
                    else:
                        prev = per_key.get(used2)
                        if prev is None or cand[:3] < prev[:3]:
                            per_key[used2] = cand
        if not nxt_best:  # radius below the feasible minimum; solve() prevents this
            raise RuntimeError("no candidate within the given residual radius")

        # Build the layer's node table: per state key, sweep event counts
        # ascending and keep a candidate only while its cost lowers the running
        # minimum -- the survivors are exactly the (events, cost) Pareto
        # frontier, and objective 4 is already settled by parent rank/v.
        table: list[tuple[int, int, int, int, int]] = []
        nxt: dict[tuple[tuple[int, ...], int], list[int]] = {}
        for skey, by_used in nxt_best.items():
            ids = []
            min_cost: Optional[int] = None
            for used2 in sorted(by_used):
                cost2, parent_rank, v, prev_id = by_used[used2]
                if min_cost is None or cost2 < min_cost:
                    ids.append(len(table))
                    table.append((used2, cost2, v, prev_id, 0))  # rank below
                    min_cost = cost2
            nxt[skey] = ids
        # Assign lexicographic path ranks: order survivors by (parent rank, v).
        order = sorted(
            range(len(table)),
            key=lambda i: (prev_table[table[i][3]][4], table[i][2]),
        )
        for rank, i in enumerate(order):
            used2, cost2, v, prev_id, _ = table[i]
            table[i] = (used2, cost2, v, prev_id, rank)
        layers.append(table)
        states = nxt
        ts_cache = nxt_cache

    # Pick the best terminal node: (cost, events, lexicographic path).
    best_id = None
    best_key = None
    for (tail, amp_sum), node_ids in states.items():
        if constrain_total and not (total_lo <= amp_sum <= total_hi):
            continue
        for nid in node_ids:
            used, cost, v, prev_id, rank = layers[n][nid]
            key = (cost, used, rank)
            if best_key is None or key < best_key:
                best_key = key
                best_id = nid
    # Reconstruct the amplitude sequence through the parent links.
    path_rev: list[int] = []
    cur = best_id
    for p in range(n, 0, -1):
        used, cost, v, prev_id, rank = layers[p][cur]
        path_rev.append(v)
        cur = prev_id
    path_rev.reverse()
    return best_key[0], best_key[1], tuple(path_rev)


def solve(
    samples: Sequence[int],
    kernel: Sequence[int],
    max_residual: int,
    max_amplitude: int,
    max_events: int,
    saturation_level: Optional[int] = None,
    total_amplitude_range: Optional[tuple[int, int]] = None,
) -> SolveResult:
    """Jointly deconvolve ``samples`` under the given limits.

    Returns a :class:`SolveResult`; when infeasible, only ``feasible`` and
    ``first_unexplainable_position`` are populated.

    When ``saturation_level`` is ``None`` the request follows the original
    (fully two-sided residual) semantics exactly, and
    ``total_amplitude_range`` must also be ``None``.  Otherwise samples equal
    to the level are clipped readings (``prediction >= saturation_level``);
    samples strictly above the level are illegal input and must be rejected by
    the caller.  ``total_amplitude_range``, when given, constrains the sum of
    all amplitudes to a closed interval inside the same dynamic program.
    """
    n = len(samples)
    if saturation_level is None:
        if total_amplitude_range is not None:
            raise ValueError("totalAmplitudeRange requires saturationLevel")
        clipped_mask = [False] * n
        total_lo, total_hi = 0, max_amplitude * n
        constrain_total = False
    else:
        clipped_mask = [s == saturation_level for s in samples]
        constrain_total = total_amplitude_range is not None
        if constrain_total:
            total_lo, total_hi = total_amplitude_range  # type: ignore[misc]
        else:
            total_lo, total_hi = 0, max_amplitude * n

    def scan(radius: int) -> tuple[bool, Optional[int]]:
        return _scan(
            samples, kernel, radius, max_amplitude, max_events,
            clipped_mask, total_lo, total_hi, constrain_total,
        )

    # Residuals at non-clipped points are integers, so the optimal max
    # |residual| is an integer in [0, max_residual].  Probe radius 0 first
    # (the common exact-recovery case): that single scan both establishes
    # feasibility and fixes the radius.  Otherwise exponentially double the
    # radius towards max_residual -- infeasible probes are cheap because the
    # scan aborts at the first unexplainable position -- then binary-search
    # the bracket for the minimum feasible radius.
    exact_ok, exact_fail = scan(0)
    if exact_ok:
        radius = 0
    elif max_residual == 0:
        return SolveResult(
            feasible=False, first_unexplainable_position=exact_fail
        )
    else:
        lo, hi = 0, -1  # lo infeasible, hi feasible (-1 until found)
        probe = 1
        while probe < max_residual:
            if scan(probe)[0]:
                hi = probe
                break
            lo = probe
            probe *= 2
        if hi < 0:
            feasible, first_fail = scan(max_residual)
            if not feasible:
                return SolveResult(
                    feasible=False, first_unexplainable_position=first_fail
                )
            hi = max_residual
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if scan(mid)[0]:
                hi = mid
            else:
                lo = mid
        radius = hi

    sum_abs, event_count, path = _optimize(
        samples, kernel, radius, max_amplitude, max_events,
        clipped_mask, total_lo, total_hi, constrain_total,
    )
    amplitudes = list(path)
    m = len(kernel)
    prediction = []
    for i in range(n):
        acc = 0
        for j in range(m):
            t = i - j
            if t >= 0:
                acc += kernel[j] * amplitudes[t]
        prediction.append(acc)
    residuals = [samples[i] - prediction[i] for i in range(n)]
    # Clipped points carry a one-sided inequality, not a residual: the max
    # residual objective is taken over non-clipped points only.
    max_abs = max(
        (abs(r) for i, r in enumerate(residuals) if not clipped_mask[i]),
        default=0,
    )
    result = SolveResult(
        feasible=True,
        amplitudes=amplitudes,
        prediction=prediction,
        residuals=residuals,
        max_abs_residual=max_abs,
        sum_abs_residual=sum_abs,
        event_count=event_count,
    )
    if saturation_level is not None:
        result.saturation_evidence = SaturationEvidence(
            total_amplitude=sum(amplitudes),
            saturated_points=[
                SaturatedPoint(
                    position=i,
                    sample=samples[i],
                    prediction=prediction[i],
                )
                for i in range(n)
                if clipped_mask[i]
            ],
        )
    return result
