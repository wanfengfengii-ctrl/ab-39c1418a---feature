"""Request/response schemas for the deconvolution API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("kernel")
    @classmethod
    def _kernel_positive(cls, value: List[int]) -> List[int]:
        for k in value:
            if k < 1:
                raise ValueError("kernel entries must be positive integers")
            if k > 1_000_000:
                raise ValueError("kernel entries must be <= 1000000")
        return value


class Event(BaseModel):
    position: int = Field(..., description="Sample index where the event starts.")
    amplitude: int = Field(..., description="Positive integer amplitude of the event.")


class Objectives(BaseModel):
    maxAbsResidual: int
    sumAbsResidual: int
    eventCount: int


class DeconvolveSuccessResponse(BaseModel):
    status: str = "ok"
    events: List[Event] = Field(..., description="Recovered events (position/amplitude).")
    amplitudes: List[int] = Field(..., description="Full amplitude sequence (one entry per sample).")
    prediction: List[int] = Field(..., description="Predicted waveform, aligned with the samples.")
    residuals: List[int] = Field(..., description="Pointwise residuals: samples - prediction.")
    objectives: Objectives


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
