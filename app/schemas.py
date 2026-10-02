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
            "ADC platform value. Optional: when omitted the request follows "
            "the original two-sided residual semantics. When given, a sample "
            "equal to it is a clipped reading (true prediction >= level); any "
            "sample strictly above it is illegal."
        ),
    )
    totalAmplitudeRange: Optional[List[int]] = Field(
        default=None,
        description=(
            "Closed [min, max] interval for the sum of all event amplitudes, "
            "from an independent integrating readout. Optional and only "
            "meaningful together with saturationLevel."
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
    def _total_range_shape(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("totalAmplitudeRange must contain exactly [min, max]")
        low, high = value
        if low < 0:
            raise ValueError("totalAmplitudeRange min must be >= 0")
        if high > 1_000_000_000:
            raise ValueError("totalAmplitudeRange max must be <= 1000000000")
        if low > high:
            raise ValueError("totalAmplitudeRange min must be <= max")
        return value

    @model_validator(mode="after")
    def _validate_saturation(self) -> "DeconvolveRequest":
        if self.totalAmplitudeRange is not None and self.saturationLevel is None:
            raise ValueError(
                "totalAmplitudeRange requires saturationLevel to be given"
            )
        if self.saturationLevel is not None:
            over = [
                i for i, s in enumerate(self.samples)
                if s > self.saturationLevel
            ]
            if over:
                raise ValueError(
                    "samples must not exceed saturationLevel; offending positions: "
                    + ", ".join(str(i) for i in over)
                )
        return self


class Event(BaseModel):
    position: int = Field(..., description="Sample index where the event starts.")
    amplitude: int = Field(..., description="Positive integer amplitude of the event.")


class Objectives(BaseModel):
    maxAbsResidual: int
    sumAbsResidual: int
    eventCount: int


class SaturatedPointEvidence(BaseModel):
    position: int = Field(..., description="Index of the clipped sample.")
    sample: int = Field(..., description="Clipped reading (equal to saturationLevel).")
    prediction: int = Field(
        ..., description="Recovered unclipped prediction at this position."
    )
    excess: int = Field(
        ...,
        ge=0,
        description="Amount by which the prediction exceeds the clipped reading.",
    )


class SaturationEvidenceResponse(BaseModel):
    totalAmplitude: int = Field(
        ..., description="Sum of all recovered event amplitudes."
    )
    saturatedPoints: List[SaturatedPointEvidence] = Field(
        ..., description="One evidence entry per clipped sample, in position order."
    )


class DeconvolveSuccessResponse(BaseModel):
    status: str = "ok"
    events: List[Event] = Field(..., description="Recovered events (position/amplitude).")
    amplitudes: List[int] = Field(..., description="Full amplitude sequence (one entry per sample).")
    prediction: List[int] = Field(..., description="Predicted waveform, aligned with the samples.")
    residuals: List[int] = Field(..., description="Pointwise residuals: samples - prediction.")
    objectives: Objectives
    saturation: Optional[SaturationEvidenceResponse] = Field(
        default=None,
        description=(
            "Present only when saturationLevel was given: total amplitude and "
            "per-clipped-point excess evidence."
        ),
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
            "within [0, maxAmplitude], at most maxEvents events, and total "
            "amplitude within totalAmplitudeRange when given) can keep every "
            "residual of samples[0..p] within +/-maxResidual under the "
            "per-point saturation semantics."
        ),
    )
    totalAmplitudeRange: Optional[List[int]] = Field(
        default=None,
        description=(
            "Echoed [min, max] total-amplitude constraint when it was part of "
            "the request and no extendible candidate could satisfy it."
        ),
    )
