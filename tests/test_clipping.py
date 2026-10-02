"""Solver tests for strong-pulse clipping with an integrator total interval."""
from __future__ import annotations

import itertools
import random

import pytest

from app.deconvolve import solve
from tests.test_deconvolve import convolve


KERNEL = [3, 2, 1]


def brute_force_clipped(samples, kernel, max_residual, max_amplitude, max_events,
                        saturation_level, total_lo, total_hi):
    """Exhaustive reference under pointwise clipping and a total-sum interval.

    Returns ((max_abs, sum_abs, events, amplitudes) or None, first_fail,
    constraint) where first_fail is the first prefix position unexplainable by
    ANY train satisfying all constraints (including the total interval).
    """
    n = len(samples)
    m = len(kernel)
    saturated = [s == saturation_level for s in samples]
    best_key = None
    explainable = [False] * n
    residual_explainable = [False] * n  # ignoring the total interval
    for x in itertools.product(range(max_amplitude + 1), repeat=n):
        events = sum(1 for v in x if v)
        total = sum(x)
        total_ok = total_lo <= total <= total_hi
        pred = [
            sum(kernel[j] * x[i - j] for j in range(m) if i - j >= 0)
            for i in range(n)
        ]
        prefix_ok = True
        r_prefix_ok = True
        for i in range(n):
            if saturated[i]:
                point_ok = pred[i] >= saturation_level
            else:
                point_ok = abs(samples[i] - pred[i]) <= max_residual
            if not point_ok or events > max_events:
                prefix_ok = False
                r_prefix_ok = False
            if prefix_ok and total_ok:
                explainable[i] = True
            if r_prefix_ok:
                residual_explainable[i] = True
        feasible_point = all(
            (pred[i] >= saturation_level) if saturated[i]
            else abs(samples[i] - pred[i]) <= max_residual
            for i in range(n)
        )
        if feasible_point and events <= max_events and total_ok:
            unclipped_res = [
                abs(samples[i] - pred[i]) for i in range(n) if not saturated[i]
            ]
            key = (
                max(unclipped_res, default=0),
                sum(unclipped_res),
                events,
                x,
            )
            if best_key is None or key < best_key:
                best_key = key
    first_fail = next((i for i in range(n) if not explainable[i]), None)
    residual_first = next(
        (i for i in range(n) if not residual_explainable[i]), None
    )
    if best_key is not None:
        constraint = None
        total_first = None
    elif residual_first is not None:
        # The residual/clipping/event limits alone are unsatisfiable.
        constraint = "residual"
        total_first = residual_first
    else:
        # Residual limits admit complete trains; the total interval removes
        # the last extendable candidates.
        constraint = "total"
        total_first = first_fail
    return best_key, total_first, constraint


class TestClippingRecovery:
    def test_clipped_plateau_pinned_by_integrator_total(self):
        # Only the last sample is observed, at the plateau: prediction must be
        # >= 15.  Without a total reading the lex-smallest choice is amplitude
        # 5; the closed interval [8, 8] pins it to 8.
        samples = [0] * 11 + [15]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=15, total_amplitude_range=(8, 8),
        )
        assert result.feasible
        assert result.amplitudes == [0] * 11 + [8]
        assert result.prediction[11] == 24
        assert result.total_amplitude == 8
        assert result.saturation_positions == [11]
        assert result.saturation_excess == [9]
        assert (result.max_abs_residual, result.sum_abs_residual) == (0, 0)

    def test_clipped_point_inside_overlap_recovered(self):
        # Events at 4 (amp 5) and 6 (amp 6); the ADC clips at 20 while the
        # true prediction at sample 6 is 23.  Surrounding tails remain
        # observable and pin both amplitudes.
        samples = [0, 0, 0, 0, 15, 10, 20, 12, 6, 0, 0, 0]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=20, total_amplitude_range=(11, 11),
        )
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 0, 0]
        assert result.prediction[6] == 23
        assert result.saturation_positions == [6]
        assert result.saturation_excess == [3]
        assert result.residuals[6] == -3  # full un-clipped residual preserved
        assert result.total_amplitude == 11

    def test_exact_plateau_prediction_has_zero_excess(self):
        # A train whose prediction equals the level at the clipped point is
        # admissible with excess 0.
        samples = convolve([0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0], KERNEL)
        samples = [min(v, 15) for v in samples]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=15, total_amplitude_range=(5, 5),
        )
        assert result.feasible
        assert result.saturation_excess == [0]
        assert result.amplitudes[4] == 5

    def test_total_interval_drives_objective_tie_break(self):
        # Two trailing plateau points admit several trains whose un-clipped
        # residual sum is zero; the closed interval [8, 8] fixes the total.
        samples = [0] * 10 + [15, 15]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=15, total_amplitude_range=(8, 8),
        )
        assert result.feasible
        assert result.total_amplitude == 8
        for i, s in enumerate(samples):
            assert result.prediction[i] >= s
        # Minimising events before the amplitude sequence prefers a single
        # event (x10 = 8): it covers both plateau points.
        assert result.event_count == 1
        assert result.amplitudes[10] == 8


class TestClippingInfeasible:
    def test_total_conflict_is_identifiable(self):
        # Plateau 15 at the last point needs amplitude >= 5, so total [3, 4]
        # conflicts with every explainable train.
        samples = [0] * 11 + [15]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=15, total_amplitude_range=(3, 4),
        )
        assert not result.feasible
        assert result.constraint == "total"
        assert result.first_unexplainable_position == 0

    def test_residual_failure_not_attributed_to_total(self):
        # Amplitude cap 4 cannot reach the plateau (needs >= 5): this is a
        # residual/clipping failure at the clipped sample itself.
        samples = [0] * 11 + [15]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=4, max_events=4,
            saturation_level=15, total_amplitude_range=(8, 8),
        )
        assert not result.feasible
        assert result.constraint == "residual"
        assert result.first_unexplainable_position == 11

    def test_event_limit_failure_inside_clipped_wave(self):
        # The clipped overlap needs two events; with max_events=1 the
        # observable tail at sample 8 cannot be explained by anyone.
        samples = [0, 0, 0, 0, 15, 10, 20, 12, 6, 0, 0, 0]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=1,
            saturation_level=20, total_amplitude_range=(11, 11),
        )
        assert not result.feasible
        assert result.constraint == "residual"

    def test_total_upper_bound_prunes_during_scan(self):
        # Two plateau points, each followed by its observable tail; each
        # plateau forces an amplitude of at least 5, so the minimum total
        # is 10.  The tails are zeros consistent with the kernel, so no
        # residual conflict exists -- only the interval can fail.
        samples = [0, 0, 0, 0, 15, 10, 5, 0, 15, 10, 5, 0]
        result = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=15, total_amplitude_range=(10, 10),
        )
        assert result.feasible
        assert result.total_amplitude == 10
        assert result.amplitudes[4] == 5 and result.amplitudes[8] == 5
        tight = solve(
            samples, KERNEL, max_residual=0, max_amplitude=10, max_events=4,
            saturation_level=15, total_amplitude_range=(9, 9),
        )
        assert not tight.feasible
        assert tight.constraint == "total"

    def test_underconstrained_input_hits_state_budget(self):
        # Every point clipped with loose limits: the exact joint DP exceeds
        # its per-layer state budget instead of running away combinatorially.
        result = solve(
            [15] * 36, KERNEL, max_residual=0, max_amplitude=1000, max_events=36,
            saturation_level=15, total_amplitude_range=(0, 3000),
        )
        assert not result.feasible
        assert result.constraint == "budget"
        assert 0 <= result.first_unexplainable_position < 36


class TestClippingBruteForceCrossCheck:
    @pytest.mark.parametrize("seed", range(80))
    def test_matches_exhaustive_search(self, seed):
        rng = random.Random(seed)
        n = 6
        kernel = [rng.randint(1, 3) for _ in range(3)]
        max_amplitude = 2
        max_events = rng.randint(0, 4)
        max_residual = rng.randint(0, 2)
        # Generate observations below/at a level, with at least one clipped.
        level = rng.randint(4, 9)
        samples = []
        clipped_forced = False
        for _ in range(n):
            if rng.random() < 0.3:
                samples.append(level)
                clipped_forced = True
            else:
                samples.append(rng.randint(-1, level))
        if not clipped_forced:
            samples[rng.randrange(n)] = level
        total_lo = rng.randint(0, 6)
        total_hi = rng.randint(total_lo, 7)

        result = solve(
            samples, kernel, max_residual, max_amplitude, max_events,
            saturation_level=level,
            total_amplitude_range=(total_lo, total_hi),
        )
        best_key, first_fail, constraint = brute_force_clipped(
            samples, kernel, max_residual, max_amplitude, max_events,
            level, total_lo, total_hi,
        )

        if best_key is None:
            assert not result.feasible
            assert result.constraint == constraint
            assert result.first_unexplainable_position == first_fail
            return

        assert result.feasible
        exp_max, exp_sum, exp_events, exp_x = best_key
        assert result.max_abs_residual == exp_max
        assert result.sum_abs_residual == exp_sum
        assert result.event_count == exp_events
        assert tuple(result.amplitudes) == exp_x
        assert result.total_amplitude == sum(exp_x)
        assert total_lo <= result.total_amplitude <= total_hi
        assert result.prediction == convolve(result.amplitudes, kernel)
        for i, s in enumerate(samples):
            if s == level:
                assert result.prediction[i] >= level
            else:
                assert abs(s - result.prediction[i]) <= max_residual
