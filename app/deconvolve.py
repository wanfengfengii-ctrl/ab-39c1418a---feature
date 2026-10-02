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

Strong-pulse clipping (optional)
--------------------------------
When ``saturation_level`` is given together with a closed interval
``[total_lo, total_hi]`` for the sum of all event amplitudes (an independent
integrator reading), samples equal to the saturation level are *clipped*:
they only assert that the true prediction is no smaller than the level
(``prediction[p] >= saturation_level``, one-sided), while samples strictly
below the level keep the ordinary two-sided residual.  Samples above the
level are rejected upstream as invalid input.

The clipped-point semantics, the event-count limit and the closed total
interval are solved in one joint dynamic program -- its states carry the
tail and events used, and every state is annotated with the union of
integer intervals of amplitude sums reachable through it (forward) and still
achievable over the remaining positions (backward); a state survives only
while those interval sets can meet the closed total interval.  The
clipping-aware total constraint is therefore never applied as a post-hoc
filter on solutions of the clipped waveform.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


# Per-layer state budget for the clipping dynamic program.  Weak one-sided
# clipping constraints (many plateau samples together with loose event/total
# limits) make the exact joint DP grow combinatorially (millions of
# states); the bound keeps the service responsive while staying far above
# genuinely invertible instances -- a tight integrator interval prunes the
# sum-aware layers to a handful of states, and even heavily under-determined
# but realisable instances stay well under this size.
CLIPPED_STATE_BUDGET = 300_000


class _BudgetExceeded(RuntimeError):
    """Raised inside the clipping optimiser when its state layer overflows."""

    def __init__(self, position: int) -> None:
        super().__init__("clipping DP state budget exceeded")
        self.position = position


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
    total_amplitude: Optional[int] = None
    saturation_positions: Optional[list[int]] = None
    saturation_excess: Optional[list[int]] = None
    constraint: Optional[str] = None
    budget_position: Optional[int] = None


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


def _clipped_allowed_range(
    sample: int,
    radius: int,
    tail_sum: int,
    k0: int,
    max_amplitude: int,
    saturated: bool,
) -> tuple[int, int]:
    """Admissible amplitudes v in [0, max_amplitude] at one sample.

    Below the saturation level the ordinary two-sided residual applies:
    ``|k0*v + tail_sum - sample| <= radius``.  At a clipped (saturation-level)
    sample only the one-sided assertion ``k0*v + tail_sum >= sample`` binds
    (the true prediction is no smaller than the plateau); an upper residual
    radius there is vacuous.
    """
    if saturated:
        lo = _ceil_div(sample - tail_sum, k0)
        hi = max_amplitude
    else:
        lo = _ceil_div(sample - radius - tail_sum, k0)
        hi = (sample + radius - tail_sum) // k0
    if lo < 0:
        lo = 0
    if hi > max_amplitude:
        hi = max_amplitude
    return lo, hi


def _merge_intervals(parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of closed integer intervals given as ``[lo, hi]`` pairs."""
    if not parts:
        return []
    parts = sorted(parts)
    merged: list[tuple[int, int]] = []
    cur_lo, cur_hi = parts[0]
    for lo, hi in parts[1:]:
        if lo <= cur_hi + 1:
            if hi > cur_hi:
                cur_hi = hi
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return merged


def _shift_intervals(
    intervals: list[tuple[int, int]], shift: int, cap: int
) -> list[tuple[int, int]]:
    """Shift every interval by ``shift`` and drop parts above ``cap``."""
    out: list[tuple[int, int]] = []
    for lo, hi in intervals:
        lo += shift
        hi = min(hi + shift, cap)
        if lo <= hi:
            out.append((lo, hi))
    return out


def _intervals_meet(
    a_intervals: list[tuple[int, int]],
    b_intervals: list[tuple[int, int]],
    total_lo: int,
    total_hi: int,
) -> bool:
    """Whether some a, b from the two interval unions have a+b in the closed
    total interval."""
    for alo, ahi in a_intervals:
        for blo, bhi in b_intervals:
            if alo + blo <= total_hi and ahi + bhi >= total_lo:
                return True
    return False


def _intersect_intervals(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Set intersection of two unions of closed integer intervals."""
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        alo, ahi = a[i]
        blo, bhi = b[j]
        lo, hi = max(alo, blo), min(ahi, bhi)
        if lo <= hi:
            out.append((lo, hi))
        if ahi < bhi:
            i += 1
        else:
            j += 1
    return out


def _scan_clipped(
    samples: Sequence[int],
    kernel: Sequence[int],
    radius: int,
    max_amplitude: int,
    max_events: int,
    saturation_level: int,
    total_lo: int,
    total_hi: int,
) -> tuple[Optional[dict], Optional[int], Optional[str]]:
    """Feasibility with pointwise clipping, event limit and total-amplitude
    interval solved jointly.

    DP states are just ``(tail, events used)``; each state carries the *set*
    of amplitude sums reachable on paths through it, represented as a union
    of disjoint integer intervals (shifting a reachable interval by a
    contiguous admissible-amplitude interval keeps it a single interval, so
    the unions stay compact).

    Three passes share the per-state transition structure:

    * ``R`` is a cheap forward reachability set ignoring the total
      interval -- it decides the residual/event-limited first unexplainable
      position;
    * ``B`` is a backward table over ``R`` states: additional-sum intervals
      still achievable over the remaining positions (capped at
      ``total_hi``);
    * ``F`` is a forward table of used-sum intervals, pruned at every layer
      against ``B``: a state (and individual sum intervals) survives only
      while some completion lands in ``[total_lo, total_hi]``.

    The closed total interval thus prunes prefixes *during* the scan; it is
    never a post-hoc filter.  Failure attribution: "residual" when ``R``
    itself dies, otherwise "total".

    Returns ``(tables, first_fail, constraint)`` with ``tables`` holding
    ``(F, B)`` for the optimiser; ``tables`` is ``None`` when infeasible.
    """
    n = len(samples)
    m = len(kernel)
    k0 = kernel[0]
    w = _tail_weights(kernel)
    saturated = [s == saturation_level for s in samples]
    zero_tail = (0,) * (m - 1)
    State = tuple[tuple[int, ...], int]
    # Any single amplitude above the total upper bound is hopeless (sums are
    # non-negative), which also keeps large maxAmplitude requests tractable.
    amp_cap = min(max_amplitude, total_hi)

    tail_sum_cache: dict[tuple[int, ...], int] = {}

    def tail_sum_of(tail: tuple[int, ...]) -> int:
        ts = tail_sum_cache.get(tail)
        if ts is None:
            ts = 0
            for i in range(m - 1):
                ti = tail[i]
                if ti:
                    ts += w[i] * ti
            tail_sum_cache[tail] = ts
        return ts

    # Pass 1: sum-independent forward reachability R.
    R: list[set[State]] = [{(zero_tail, 0)}]
    for p in range(n):
        sp = samples[p]
        sat = saturated[p]
        rcur: set[State] = set()
        for tail, used in R[p]:
            ts = tail_sum_of(tail)
            # R ignores the total interval: use the full amplitude bound so
            # the residual/event attribution stays independent of it.
            lo, hi = _clipped_allowed_range(
                sp, radius, ts, k0, max_amplitude, sat
            )
            new_tail_prefix = tail[1:]
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 <= max_events:
                    rcur.add((new_tail_prefix + (v,), used2))
            if len(rcur) > CLIPPED_STATE_BUDGET:
                break
        if len(rcur) > CLIPPED_STATE_BUDGET:
            # Weak one-sided constraints: the exact DP is intractable.
            return None, p, "budget"
        if not rcur:
            return None, p, "residual"
        R.append(rcur)

    # Pass 2: backward achievable-additional-sum table B over R states.
    B: list[dict[State, list[tuple[int, int]]]] = [{} for _ in range(n + 1)]
    for state in R[n]:
        B[n][state] = [(0, 0)]
    for p in range(n - 1, -1, -1):
        sp = samples[p]
        sat = saturated[p]
        b_next = B[p + 1]
        table: dict[State, list[tuple[int, int]]] = {}
        for tail, used in R[p]:
            ts = tail_sum_of(tail)
            lo, hi = _clipped_allowed_range(
                sp, radius, ts, k0, amp_cap, sat
            )
            new_tail_prefix = tail[1:]
            parts: list[tuple[int, int]] = []
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                child = b_next.get((new_tail_prefix + (v,), used2))
                if child is not None:
                    parts.extend(_shift_intervals(child, v, total_hi))
            if parts:
                table[(tail, used)] = _merge_intervals(parts)
        B[p] = table

    def prune_used_sums(
        layer: dict[State, list[tuple[int, int]]], q: int
    ) -> dict[State, list[tuple[int, int]]]:
        """Keep only used-sum intervals extendable into the total interval."""
        out: dict[State, list[tuple[int, int]]] = {}
        for state, parts in layer.items():
            b_intervals = B[q].get(state)
            if not b_intervals:
                continue
            # Per completion interval [blo, bhi], a used sum a survives iff
            # total_lo <= a + b <= total_hi for some b, i.e.
            # a in [total_lo - bhi, total_hi - blo].
            allowed = _merge_intervals([
                (total_lo - bhi, total_hi - blo) for blo, bhi in b_intervals
            ])
            kept = _intersect_intervals(_merge_intervals(parts), allowed)
            if kept:
                out[state] = kept
        return out

    # Pass 3: forward used-sum table F, pruned layer by layer against B.
    F0 = prune_used_sums({(zero_tail, 0): [(0, 0)]}, 0)
    if not F0:
        return None, 0, "total"
    F: list[dict[State, list[tuple[int, int]]]] = [F0]
    for p in range(n):
        sp = samples[p]
        sat = saturated[p]
        fcur: dict[State, list[tuple[int, int]]] = {}
        for (tail, used), intervals in F[p].items():
            ts = tail_sum_of(tail)
            lo, hi = _clipped_allowed_range(
                sp, radius, ts, k0, amp_cap, sat
            )
            new_tail_prefix = tail[1:]
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                nxt: State = (new_tail_prefix + (v,), used2)
                parts = fcur.setdefault(nxt, [])
                for alo, ahi in intervals:
                    nlo = alo + v
                    nhi = min(ahi + v, total_hi)
                    if nlo <= nhi:
                        parts.append((nlo, nhi))
            if len(fcur) > CLIPPED_STATE_BUDGET:
                break
        if len(fcur) > CLIPPED_STATE_BUDGET:
            return None, p, "budget"
        pruned = prune_used_sums(fcur, p + 1)
        if not pruned:
            # Samples 0..p have just been assigned yet no state extends into
            # the total interval; R[p+1] is non-empty, so the interval
            # removed the last extendable candidate at sample p.
            return None, p, "total"
        F.append(pruned)
    return {"F": F, "B": B}, None, None


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


def _optimize_clipped(
    samples: Sequence[int],
    kernel: Sequence[int],
    radius: int,
    max_amplitude: int,
    max_events: int,
    saturation_level: int,
    total_lo: int,
    total_hi: int,
    tables: dict,
) -> tuple[int, int, int, tuple[int, ...]]:
    """Minimise (sum |residual|, events, amplitude sequence) at fixed radius.

    Residual cost is counted only at samples strictly below the saturation
    level; clipped samples contribute the one-sided feasibility assertion but
    no residual.  The terminal state additionally has to land in the closed
    total-amplitude interval ``[total_lo, total_hi]``.

    States are ``(tail, events, sum used)`` mapping to the best
    ``(residual cost, amplitude path)``.  The feasibility tables ``F``/``B``
    prune prefix states that can no longer reach the closed interval, so the
    event limit and the total bound are enforced jointly along the way.
    """
    n = len(samples)
    m = len(kernel)
    k0 = kernel[0]
    w = _tail_weights(kernel)
    saturated = [s == saturation_level for s in samples]
    B = tables["B"]
    # Any single amplitude above the total upper bound is hopeless.
    amp_cap = min(max_amplitude, total_hi)
    # (tail, events used, sum used) -> (sum |residual| so far, path so far)
    states: dict[
        tuple[tuple[int, ...], int, int], tuple[int, tuple[int, ...]]
    ] = {((0,) * (m - 1), 0, 0): (0, ())}
    for p in range(n):
        sp = samples[p]
        sat = saturated[p]
        nxt: dict[
            tuple[tuple[int, ...], int, int], tuple[int, tuple[int, ...]]
        ] = {}
        for (tail, used, sm), (cost, path) in states.items():
            tail_sum = 0
            for i in range(m - 1):
                ti = tail[i]
                if ti:
                    tail_sum += w[i] * ti
            lo, hi = _clipped_allowed_range(
                sp, radius, tail_sum, k0, amp_cap, sat
            )
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                sm2 = sm + v
                if sm2 > total_hi:
                    continue
                key = (tail[1:] + (v,), used2, sm2)
                point_cost = 0 if sat else abs(k0 * v + tail_sum - sp)
                cand = (cost + point_cost, path + (v,))
                cur = nxt.get(key)
                if cur is None or cand < cur:
                    nxt[key] = cand
            if len(nxt) > CLIPPED_STATE_BUDGET:
                raise _BudgetExceeded(p)
        if len(nxt) > CLIPPED_STATE_BUDGET:
            raise _BudgetExceeded(p)
        # Drop states that cannot be extended into the total interval anymore.
        alive: dict[
            tuple[tuple[int, ...], int, int], tuple[int, tuple[int, ...]]
        ] = {}
        for (tail, used, sm), value in nxt.items():
            b_intervals = B[p + 1].get((tail, used))
            if b_intervals is None:
                continue
            if _intervals_meet([(sm, sm)], b_intervals, total_lo, total_hi):
                alive[(tail, used, sm)] = value
        if not alive:  # radius below the feasible minimum; solve() prevents this
            raise RuntimeError("no candidate within the given residual radius")
        states = alive
    return min(
        (cost, used, sm, path)
        for (tail, used, sm), (cost, path) in states.items()
        if total_lo <= sm <= total_hi
    )


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
    ``first_unexplainable_position`` (plus ``constraint`` in clipping mode)
    are populated.

    When both ``saturation_level`` and ``total_amplitude_range``
    ``(lo, hi)`` are supplied, samples equal to the saturation level are
    treated as clipped (prediction no smaller than the level) and the event
    amplitude sum must land in the closed interval.  Samples above the
    saturation level are the caller's responsibility to reject beforehand.
    Supplying only one of the two is likewise a caller-side validation error.
    """
    n = len(samples)

    clipped = saturation_level is not None
    if clipped:
        if total_amplitude_range is None:
            raise ValueError(
                "total_amplitude_range is required when saturation_level is set"
            )
        total_lo, total_hi = total_amplitude_range
        m = len(kernel)

        # Any single amplitude above the total upper bound is hopeless (sums
        # are non-negative); it also bounds the largest possible prediction.
        amp_cap = min(max_amplitude, total_hi)

        # At a radius of at most P + max|sample| every amplitude in
        # [0, amp_cap] is admissible at every point regardless of the tail
        # (|s - pred| <= |s| + pred <= |s| + P), so radii beyond that are
        # indistinguishable and the optimal radius can never exceed it.
        max_pred_bound = amp_cap * sum(kernel)
        r_limit = min(
            max_residual,
            max_pred_bound + max((abs(s) for s in samples), default=0),
        )

        def scan(rad: int):
            return _scan_clipped(
                samples, kernel, rad, max_amplitude, max_events,
                saturation_level, total_lo, total_hi,
            )

        tables, first_fail, constraint = scan(r_limit)
        if constraint == "budget":
            return SolveResult(
                feasible=False,
                first_unexplainable_position=first_fail,
                constraint="budget",
                budget_position=first_fail,
            )
        if first_fail is not None:
            return SolveResult(
                feasible=False,
                first_unexplainable_position=first_fail,
                constraint=constraint,
            )
        # A feasible train at r_limit gives a (usually much tighter) upper
        # bound on the minimal radius: its own largest un-clipped residual.
        try:
            _, _, _, ub_path = _optimize_clipped(
                samples, kernel, r_limit, max_amplitude, max_events,
                saturation_level, total_lo, total_hi, tables,
            )
        except _BudgetExceeded as exc:
            return SolveResult(
                feasible=False,
                first_unexplainable_position=exc.position,
                constraint="budget",
            )
        ub_pred = []
        for i in range(n):
            acc = 0
            for j in range(m):
                t = i - j
                if t >= 0:
                    acc += kernel[j] * ub_path[t]
            ub_pred.append(acc)
        r_ub = max(
            (
                abs(samples[i] - ub_pred[i])
                for i in range(n)
                if samples[i] != saturation_level
            ),
            default=0,
        )

        # Residuals at un-clipped samples are integers, so the optimal max
        # |residual| there is an integer in [0, r_ub].
        lo_r, hi_r, hi_tables = -1, r_ub, tables  # hi feasible, lo infeasible
        while hi_r - lo_r > 1:
            mid = (lo_r + hi_r) // 2
            mid_tables, fail, _ = scan(mid)
            if fail is None:
                hi_r = mid
                hi_tables = mid_tables
            else:
                lo_r = mid
        radius = hi_r

        try:
            sum_abs, event_count, total_amp, path = _optimize_clipped(
                samples, kernel, radius, max_amplitude, max_events,
                saturation_level, total_lo, total_hi, hi_tables,
            )
        except _BudgetExceeded as exc:
            return SolveResult(
                feasible=False,
                first_unexplainable_position=exc.position,
                constraint="budget",
            )
        amplitudes = list(path)
        m = len(kernel)
        saturated_mask = [s == saturation_level for s in samples]
        prediction = []
        for i in range(n):
            acc = 0
            for j in range(m):
                t = i - j
                if t >= 0:
                    acc += kernel[j] * amplitudes[t]
            prediction.append(acc)
        residuals = [samples[i] - prediction[i] for i in range(n)]
        # Objectives are defined on genuinely observed (un-clipped) points.
        max_abs = max(
            (abs(r) for i, r in enumerate(residuals) if not saturated_mask[i]),
            default=0,
        )
        return SolveResult(
            feasible=True,
            amplitudes=amplitudes,
            prediction=prediction,
            residuals=residuals,
            max_abs_residual=max_abs,
            sum_abs_residual=sum_abs,
            event_count=event_count,
            total_amplitude=total_amp,
            saturation_positions=[i for i, sat in enumerate(saturated_mask) if sat],
            saturation_excess=[
                prediction[i] - saturation_level
                for i, sat in enumerate(saturated_mask) if sat
            ],
        )

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
