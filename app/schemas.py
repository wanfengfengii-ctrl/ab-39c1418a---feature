"""Request/response schemas for the deconvolution API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DeconvolveRequest(BaseModel):
    """Input for POST /api/pulses/deconvolve."""

    samples: List[int] = Field(
        ..., min_length=12, max_length=36,
        description="Observed integer samples (12..36 values).",
    )
    kernel: List[int] = Field(
        ..., min_length=3, max_length=6,
        description="Positive integer impulse-response kernel (3..6 values).",
    )
    maxResidual: int = Field(
        ..., ge=0, le=1_000_000_000,
        description="Uniform per-sample absolute residual limit.",
    )
    maxAmplitude: int = Field(
        ..., ge=1, le=1_000_000,
        description="Upper bound for each event's integer amplitude.",
    )
    maxEvents: int = Field(
        ..., ge=0, le=10_000,
        description="Maximum number of events (non-zero amplitudes).",
    )
    saturationLevel: Optional[int] = Field(
        default=None, ge=0, le=1_000_000_000,
        description=(
            "ADC clipping level; samples equal to it are clipped (the true "
            "prediction is no smaller than the level). Must be supplied "
            "together with totalAmplitudeRange."
        ),
    )
    totalAmplitudeRange: Optional[List[int]] = Field(
        default=None,
        description=(
            "Closed interval [min, max] constraining the sum of all event "
            "amplitudes, from an independent integrator reading; must be "
            "supplied together with saturationLevel."
        ),
    )

    @field_validator("kernel")
    @classmethod
    def _kernel_positive(cls, value: List[int]) -> List[int]:
        for k in value:
            if k < 1:
                raise ValueError("kernel entries must be positive integers")
            if k > 1_000_000:
                raise ValueError("kernel entries must be <= 1000000")
        return value

    @field_validator("totalAmplitudeRange")
    @classmethod
    def _range_shape(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("totalAmplitudeRange must be a [min, max] pair")
        lo, hi = value
        if lo < 0:
            raise ValueError("totalAmplitudeRange min must be >= 0")
        if hi > 1_000_000_000:
            raise ValueError("totalAmplitudeRange max must be <= 1000000000")
        if lo > hi:
            raise ValueError("totalAmplitudeRange min must be <= max")
        return value

    @model_validator(mode="after")
    def _clipping_cross_fields(self) -> "DeconvolveRequest":
        has_level = self.saturationLevel is not None
        has_range = self.totalAmplitudeRange is not None
        if has_level != has_range:
            raise ValueError(
                "saturationLevel and totalAmplitudeRange must be supplied together"
            )
        if has_level:
            above = [i for i, s in enumerate(self.samples) if s > self.saturationLevel]
            if above:
                raise ValueError(
                    "samples must not exceed saturationLevel "
                    f"(first offending index: {above[0]})"
                )
        return self


class Event(BaseModel):
    position: int = Field(..., description="Sample index where the event starts.")
    amplitude: int = Field(..., description="Positive integer amplitude of the event.")


class Objectives(BaseModel):
    maxAbsResidual: int
    sumAbsResidual: int
    eventCount: int


class SaturationEvidence(BaseModel):
    position: int = Field(..., description="Index of a clipped (saturation-level) sample.")
    observed: int = Field(..., description="Observed plateau value (the saturation level).")
    predicted: int = Field(..., description="Un-clipped predicted value at that position.")
    excess: int = Field(
        ..., ge=0,
        description="Amount by which the prediction exceeds the saturation level.",
    )


class DeconvolveSuccessResponse(BaseModel):
    status: str = "ok"
    events: List[Event] = Field(..., description="Recovered events (position/amplitude).")
    amplitudes: List[int] = Field(..., description="Full amplitude sequence (one entry per sample).")
    prediction: List[int] = Field(..., description="Predicted waveform, aligned with the samples.")
    residuals: List[int] = Field(..., description="Pointwise residuals: samples - prediction.")
    objectives: Objectives
    # The following fields appear only when clipping mode is enabled, so
    # responses stay byte-for-byte unchanged for legacy requests.
    totalAmplitude: Optional[int] = Field(
        default=None,
        description="Sum of all recovered event amplitudes (integrator reading).",
    )
    totalAmplitudeRange: Optional[List[int]] = Field(
        default=None,
        description="Closed interval the total amplitude was constrained to.",
    )
    saturationEvidence: Optional[List[SaturationEvidence]] = Field(
        default=None,
        description="Per-clipped-point evidence: predicted value beyond the plateau.",
    )


class InfeasibleResponse(BaseModel):
    status: str = "infeasible"
    detail: str = (
        "no event train within the amplitude and event-count limits can "
        "reproduce the samples within the residual limit"
    )
    firstUnexplainablePosition: int = Field(
        ...,
        description=(
            "First sample index p for which no complete candidate (amplitudes "
            "within [0, maxAmplitude], at most maxEvents events) can keep every "
            "residual of samples[0..p] within +/-maxResidual."
        ),
    )
    # Present only in clipping mode.
    constraint: Optional[str] = Field(
        default=None,
        description=(
            "Which constraint removes the last extendable candidate: "
            "'totalAmplitudeRange', 'residualLimit', or 'stateBudgetExceeded' "
            "(too many admissible trains for exact joint inversion -- tighten "
            "maxEvents or totalAmplitudeRange)."
        ),
    )
    totalAmplitudeRange: Optional[List[int]] = Field(
        default=None,
        description="Closed total-amplitude interval in force when clipping is enabled.",
    )
