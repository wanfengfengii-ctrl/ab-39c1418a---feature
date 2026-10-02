"""Unit tests for the joint integer deconvolution solver."""
from __future__ import annotations

import itertools
import random

import pytest

from app.deconvolve import solve


def convolve(amplitudes, kernel):
    """First len(amplitudes) outputs of the zero-padded discrete convolution."""
    n = len(amplitudes)
    return [
        sum(kernel[j] * amplitudes[i - j] for j in range(len(kernel)) if i - j >= 0)
        for i in range(n)
    ]


def brute_force(samples, kernel, max_residual, max_amplitude, max_events,
                saturation_level=None, total_amplitude_range=None):
    """Exhaustive reference: optimum key + first unexplainable position.

    Returns ((max_abs, sum_abs, events, amplitudes) or None, first_unexplainable).
    Residual objectives are taken over non-clipped points only; clipped points
    (sample == saturation_level) impose the one-sided constraint
    prediction >= saturation_level.  The total amplitude is constrained to the
    closed ``total_amplitude_range`` when given.
    """
    n = len(samples)
    m = len(kernel)
    total_lo, total_hi = (
        total_amplitude_range
        if total_amplitude_range is not None
        else (None, None)
    )
    best_key = None
    explainable = [False] * n

    def point_ok(i, pred):
        if saturation_level is not None and samples[i] == saturation_level:
            return pred >= saturation_level
        return abs(samples[i] - pred) <= max_residual

    for x in itertools.product(range(max_amplitude + 1), repeat=n):
        events = sum(1 for v in x if v)
        if events > max_events:
            continue
        if total_lo is not None and not (total_lo <= sum(x) <= total_hi):
            continue
        residuals = []
        for i in range(n):
            pred = sum(kernel[j] * x[i - j] for j in range(m) if i - j >= 0)
            residuals.append(samples[i] - pred)
        prefix_ok = True
        for i in range(n):
            pred = samples[i] - residuals[i]
            if not point_ok(i, pred):
                prefix_ok = False
            if prefix_ok:
                explainable[i] = True
        if all(
            point_ok(i, samples[i] - residuals[i]) for i in range(n)
        ):
            non_clipped = [
                i for i in range(n)
                if not (saturation_level is not None and samples[i] == saturation_level)
            ]
            key = (
                max((abs(residuals[i]) for i in non_clipped), default=0),
                sum(abs(residuals[i]) for i in non_clipped),
                events,
                x,
            )
            if best_key is None or key < best_key:
                best_key = key
    first_unexplainable = next(
        (i for i in range(n) if not explainable[i]), None
    )
    return best_key, first_unexplainable


# Overlapping ground truth used by several tests: events at 4 (amp 5) and 6 (amp 4)
# with kernel [3, 2, 1]; the tails overlap because 6 - 4 < len(kernel).
KERNEL = [3, 2, 1]
EXACT_SAMPLES = [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0]
NOISY_SAMPLES = [0, 0, 0, 0, 15, 11, 17, 7, 4, 0, 0, 0, 0, 0, 0, 0]


class TestExactRecovery:
    def test_overlapping_pulses_recovered_exactly(self):
        result = solve(EXACT_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=4)
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert result.prediction == EXACT_SAMPLES
        assert result.residuals == [0] * 16
        assert result.max_abs_residual == 0
        assert result.sum_abs_residual == 0
        assert result.event_count == 2

    def test_zero_samples_give_zero_events(self):
        result = solve([0] * 12, [2, 1, 1], max_residual=0, max_amplitude=5, max_events=3)
        assert result.feasible
        assert result.event_count == 0
        assert result.amplitudes == [0] * 12
        assert result.prediction == [0] * 12

    def test_single_event_at_last_position(self):
        samples = convolve([0, 0, 0, 0, 0, 3], [2, 1, 1])
        result = solve(samples, [2, 1, 1], max_residual=0, max_amplitude=9, max_events=1)
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 0, 3]
        assert result.prediction == samples


class TestObjectiveOrder:
    def test_max_residual_is_minimised(self):
        # Exact reconstruction is impossible for the noisy samples, so the
        # smallest achievable max |residual| is 1 and must be reported.
        result = solve(NOISY_SAMPLES, KERNEL, max_residual=1, max_amplitude=10, max_events=6)
        assert result.feasible
        assert result.max_abs_residual == 1
        assert not solve(NOISY_SAMPLES, KERNEL, 0, 10, 6).feasible

    def test_lexicographic_tie_break_on_amplitudes(self):
        # kernel [1,1,1], samples [0,2,0,0,0], radius 1: both [0,1,0,0,0] and
        # [1,0,0,0,0] give (max=1, sum=3, events=1); the lexicographically
        # smaller amplitude sequence must win.
        result = solve([0, 2, 0, 0, 0], [1, 1, 1], max_residual=1, max_amplitude=5, max_events=5)
        assert result.feasible
        assert result.amplitudes == [0, 1, 0, 0, 0]
        assert (result.max_abs_residual, result.sum_abs_residual, result.event_count) == (1, 3, 1)

    def test_sum_of_residuals_minimised_before_event_count(self):
        # samples [3]*6 with kernel [2,1,1], radius 1: a 3-event fit exists
        # ([2,0,0,1,1,0], residual sum 5) but the 4-event fit
        # [1,1,0,1,1,0] has residual sum 3.  The residual sum is minimised
        # before the event count, so the 4-event fit must win.
        result = solve([3] * 6, [2, 1, 1], max_residual=1, max_amplitude=9, max_events=6)
        assert result.feasible
        assert (result.max_abs_residual, result.sum_abs_residual) == (1, 3)
        assert result.event_count == 4
        assert result.amplitudes == [1, 1, 0, 1, 1, 0]


class TestInfeasible:
    def test_residual_limit_reports_first_unexplainable_position(self):
        # NOISY_SAMPLES[5] = 11 cannot be matched exactly: position 4 forces
        # amplitude 5 (15/3), leaving tail 10 at position 5, and 11-10 is not
        # divisible by kernel[0]=3.
        result = solve(NOISY_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=6)
        assert not result.feasible
        assert result.first_unexplainable_position == 5

    def test_event_limit_reports_first_unexplainable_position(self):
        # One event explains samples up to position 5; at position 6 a second
        # event becomes unavoidable, so with max_events=1 position 6 is the
        # first point no complete candidate can explain.
        result = solve(EXACT_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=1)
        assert not result.feasible
        assert result.first_unexplainable_position == 6

    def test_amplitude_bound_violation_at_first_sample(self):
        result = solve([9, 0, 0, 0, 0, 0], KERNEL, max_residual=0, max_amplitude=2, max_events=6)
        assert not result.feasible
        assert result.first_unexplainable_position == 0

    def test_negative_sample_beyond_tolerance(self):
        samples = [0] * 12
        samples[3] = -5
        result = solve(samples, KERNEL, max_residual=2, max_amplitude=5, max_events=3)
        assert not result.feasible
        assert result.first_unexplainable_position == 3

    def test_zero_events_allowed(self):
        result = solve([1, -1, 0, 0, 0, 0], KERNEL, max_residual=1, max_amplitude=5, max_events=0)
        assert result.feasible
        assert result.event_count == 0
        assert result.amplitudes == [0] * 6


class TestBruteForceCrossCheck:
    @pytest.mark.parametrize("seed", range(60))
    def test_matches_exhaustive_search(self, seed):
        rng = random.Random(seed)
        n, m = 7, 3
        kernel = [rng.randint(1, 3) for _ in range(m)]
        max_amplitude = 2
        max_events = rng.randint(0, 4)
        max_residual = rng.randint(0, 3)
        if seed % 2 == 0:
            true_x = [rng.randint(0, max_amplitude) for _ in range(n)]
            samples = [v + rng.randint(-1, 1) for v in convolve(true_x, kernel)]
        else:
            samples = [rng.randint(-2, 15) for _ in range(n)]

        result = solve(samples, kernel, max_residual, max_amplitude, max_events)
        best_key, first_unexplainable = brute_force(
            samples, kernel, max_residual, max_amplitude, max_events
        )

        if best_key is None:
            assert not result.feasible
            assert result.first_unexplainable_position == first_unexplainable
            return

        assert result.feasible
        exp_max, exp_sum, exp_events, exp_x = best_key
        assert result.max_abs_residual == exp_max
        assert result.sum_abs_residual == exp_sum
        assert result.event_count == exp_events
        assert tuple(result.amplitudes) == exp_x
        # Response consistency: prediction and residuals match the amplitudes.
        assert result.prediction == convolve(result.amplitudes, kernel)
        assert result.residuals == [samples[i] - result.prediction[i] for i in range(n)]
        assert result.event_count == sum(1 for v in result.amplitudes if v)


class TestScaling:
    def test_full_size_exact_instance(self):
        rng = random.Random(7)
        n, m = 36, 6
        kernel = [rng.randint(1, 4) for _ in range(m)]
        true_x = [rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(n)]
        samples = convolve(true_x, kernel)
        result = solve(samples, kernel, max_residual=0, max_amplitude=9, max_events=n)
        assert result.feasible
        assert result.max_abs_residual == 0
        assert result.prediction == samples
        # Zero-residual solutions of a lower-triangular system are unique.
        assert result.amplitudes == true_x

    def test_full_size_noisy_instance(self):
        rng = random.Random(11)
        n, m = 36, 6
        kernel = [rng.randint(2, 4) for _ in range(m)]
        true_x = [rng.choice([0, 0, 1, 2]) for _ in range(n)]
        samples = [v + rng.randint(-1, 1) for v in convolve(true_x, kernel)]
        result = solve(samples, kernel, max_residual=2, max_amplitude=9, max_events=n)
        assert result.feasible
        assert result.max_abs_residual <= 2
        assert len(result.prediction) == n
        assert result.residuals == [samples[i] - result.prediction[i] for i in range(n)]


# --- Strong pulses: ADC clipping with an independent integrating readout ----

CLIP_KERNEL = [3, 2, 1]
# Ground truth: one event of amplitude 5 at position 4; waveform peaks at 15.
CLIP_SAMPLES = [0, 0, 0, 0, 12, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CLIP_LEVEL = 12
# Ground truth: events at 4 (amp 5) and 6 (amp 4); positions 4 and 6 clip.
CLIP2_SAMPLES = [0, 0, 0, 0, 12, 10, 12, 8, 4, 0, 0, 0, 0, 0, 0, 0]


class TestSaturationRecovery:
    def test_single_clipped_platform_recovers_unclipped_pulse(self):
        result = solve(
            CLIP_SAMPLES, CLIP_KERNEL, max_residual=0,
            max_amplitude=10, max_events=4, saturation_level=CLIP_LEVEL,
        )
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # The full *unclipped* prediction is retained, including above-platform.
        assert result.prediction == [0, 0, 0, 0, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert result.prediction[4] > CLIP_LEVEL
        # The clipped residual does not count against the residual objectives.
        assert result.max_abs_residual == 0
        assert result.sum_abs_residual == 0
        assert result.event_count == 1
        evidence = result.saturation_evidence
        assert evidence is not None
        assert evidence.total_amplitude == 5
        assert [(p.position, p.sample, p.prediction, p.excess)
                for p in evidence.saturated_points] == [(4, 12, 15, 3)]

    def test_overlapping_clipped_pulses_recovered_with_total(self):
        result = solve(
            CLIP2_SAMPLES, CLIP_KERNEL, max_residual=0,
            max_amplitude=10, max_events=4,
            saturation_level=CLIP_LEVEL, total_amplitude_range=(9, 9),
        )
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert result.prediction == [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0]
        evidence = result.saturation_evidence
        assert evidence.total_amplitude == 9
        assert [(p.position, p.excess) for p in evidence.saturated_points] == [(4, 3), (6, 5)]
        assert all(p.prediction >= CLIP_LEVEL for p in evidence.saturated_points)

    def test_clipped_points_excluded_from_residuals_objective(self):
        # Add noise at a non-clipped point: the max residual is still measured
        # over non-clipped samples only.
        samples = list(CLIP_SAMPLES)
        samples[5] = 11  # was 10
        result = solve(
            samples, CLIP_KERNEL, max_residual=1,
            max_amplitude=10, max_events=4, saturation_level=CLIP_LEVEL,
        )
        assert result.feasible
        assert result.max_abs_residual == 1
        clipped_residual = result.residuals[4]
        assert clipped_residual < 0  # prediction tops the clipped reading

    def test_no_saturation_keeps_evidence_absent(self):
        result = solve(
            EXACT_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=4
        )
        assert result.feasible
        assert result.saturation_evidence is None

    def test_total_range_without_saturation_is_programming_error(self):
        with pytest.raises(ValueError):
            solve(
                EXACT_SAMPLES, KERNEL, max_residual=0, max_amplitude=10,
                max_events=4, total_amplitude_range=(1, 9),
            )


class TestSaturationTotalConstraint:
    def test_upper_bound_conflict_reports_first_unextendible_position(self):
        # Clip forces x[4] >= 4; total <= 4 permits x[4] = 4, but then sample 5
        # (= 10, two-sided) cannot be matched (tail 2*4 = 8, need 10 +/-0),
        # so position 5 is the first point no extendible candidate survives.
        result = solve(
            CLIP_SAMPLES, CLIP_KERNEL, max_residual=0,
            max_amplitude=10, max_events=4,
            saturation_level=CLIP_LEVEL, total_amplitude_range=(0, 4),
        )
        assert not result.feasible
        assert result.first_unexplainable_position == 5

    def test_lower_bound_conflict_consumes_all_positions(self):
        # The prefix is explainable with small amplitudes, but the required
        # total >= 6 can never be accumulated with a valid waveform; the first
        # point lacking an extendible candidate is the final sample.
        result = solve(
            CLIP_SAMPLES, CLIP_KERNEL, max_residual=0,
            max_amplitude=10, max_events=4,
            saturation_level=CLIP_LEVEL, total_amplitude_range=(6, 10),
        )
        assert not result.feasible
        assert result.first_unexplainable_position == len(CLIP_SAMPLES) - 1

    def test_closed_interval_endpoints_are_feasible(self):
        for endpoint in (5,):
            result = solve(
                CLIP_SAMPLES, CLIP_KERNEL, max_residual=0,
                max_amplitude=10, max_events=4,
                saturation_level=CLIP_LEVEL, total_amplitude_range=(endpoint, endpoint),
            )
            assert result.feasible
            assert result.saturation_evidence.total_amplitude == endpoint

    def test_total_constraint_breaks_tie_toward_required_sum(self):
        # With samples[5] = 9 (radius 1) both x[4] = 4 (tail 8, residual sum 2)
        # and x[4] = 5 (tail 10, residual sum 1) fit; the residual objective
        # alone prefers total 5.  The closed interval [4, 4] must steer the
        # single joint solve to total 4 instead.
        samples = list(CLIP_SAMPLES)
        samples[5] = 9  # loose two-sided window around the tail
        result = solve(
            samples, CLIP_KERNEL, max_residual=1,
            max_amplitude=10, max_events=4,
            saturation_level=CLIP_LEVEL, total_amplitude_range=(4, 4),
        )
        assert result.feasible
        assert result.saturation_evidence.total_amplitude == 4
        assert result.amplitudes[4] == 4
        assert result.sum_abs_residual == 2


class TestSaturationBruteForceCrossCheck:
    @pytest.mark.parametrize("seed", range(60))
    def test_matches_exhaustive_search_with_clipping(self, seed):
        rng = random.Random(1000 + seed)
        n, m = 6, 3
        kernel = [rng.randint(1, 3) for _ in range(m)]
        max_amplitude = 2
        max_events = rng.randint(1, 4)
        max_residual = rng.randint(0, 2)
        true_x = [rng.randint(0, max_amplitude) for _ in range(n)]
        clean = convolve(true_x, kernel)
        level = rng.choice([v for v in range(1, 7)])
        # Build a waveform with some samples clipped at the platform; ensure
        # none sits above it (illegal input, tested separately).
        samples = [min(v, level) if rng.random() < 0.35 else v
                   for v in (vv + rng.randint(-1, 1) for vv in clean)]
        samples = [min(v, level) for v in samples]
        if not any(s == level for s in samples):
            samples[rng.randrange(n)] = level
        if seed % 3 == 0:
            total_range = None
        else:
            tight = rng.random() < 0.5
            total = sum(true_x)
            if tight:
                total_range = (total, total)
            else:
                total_range = (max(0, total - 2), total + 2)

        result = solve(
            samples, kernel, max_residual, max_amplitude, max_events,
            saturation_level=level, total_amplitude_range=total_range,
        )
        best_key, first_unexplainable = brute_force(
            samples, kernel, max_residual, max_amplitude, max_events,
            saturation_level=level, total_amplitude_range=total_range,
        )

        if best_key is None:
            assert not result.feasible
            assert result.first_unexplainable_position == first_unexplainable
            return

        assert result.feasible
        exp_max, exp_sum, exp_events, exp_x = best_key
        assert result.max_abs_residual == exp_max
        assert result.sum_abs_residual == exp_sum
        assert result.event_count == exp_events
        assert tuple(result.amplitudes) == exp_x
        assert result.prediction == convolve(result.amplitudes, kernel)
        assert result.saturation_evidence is not None
        assert result.saturation_evidence.total_amplitude == sum(exp_x)
        for point in result.saturation_evidence.saturated_points:
            assert samples[point.position] == level
            assert point.prediction == result.prediction[point.position]
            assert point.excess == point.prediction - level
            assert point.excess >= 0
