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


def brute_force(samples, kernel, max_residual, max_amplitude, max_events):
    """Exhaustive reference: optimum key + first unexplainable position.

    Returns ((max_abs, sum_abs, events, amplitudes) or None, first_unexplainable).
    """
    n = len(samples)
    m = len(kernel)
    best_key = None
    explainable = [False] * n
    for x in itertools.product(range(max_amplitude + 1), repeat=n):
        events = sum(1 for v in x if v)
        if events > max_events:
            continue
        residuals = []
        for i in range(n):
            pred = sum(kernel[j] * x[i - j] for j in range(m) if i - j >= 0)
            residuals.append(samples[i] - pred)
        prefix_ok = True
        for i in range(n):
            if abs(residuals[i]) > max_residual:
                prefix_ok = False
            if prefix_ok:
                explainable[i] = True
        if all(abs(r) <= max_residual for r in residuals):
            key = (
                max(abs(r) for r in residuals),
                sum(abs(r) for r in residuals),
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
