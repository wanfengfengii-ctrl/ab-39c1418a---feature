"""FastAPI surface for the scintillator pulse deconvolution service."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .deconvolve import solve
from .schemas import (
    DeconvolveRequest,
    DeconvolveSuccessResponse,
    Event,
    InfeasibleResponse,
    Objectives,
    SaturatedPointEvidence,
    SaturationEvidenceResponse,
)

app = FastAPI(
    title="Scintillator Pulse Deconvolution",
    version="1.1.0",
    description=(
        "Joint non-negative integer deconvolution of overlapping scintillator "
        "pulses: recovers event positions/amplitudes, the predicted waveform "
        "and pointwise residuals from discrete samples. Supports clipped "
        "(ADC-saturated) readings constrained by an independent integrating "
        "readout."
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed requests are reported as 400 so that 422 stays unambiguous
    (422 is reserved for 'valid request, but not invertible within limits')."""
    errors = [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={"status": "invalid_request", "detail": errors},
    )


@app.get("/healthz", tags=["meta"])
def healthz() -> dict:
    """Liveness/readiness probe used by the Docker health check."""
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "pulse-deconvolve",
        "version": "1.1.0",
        "docs": "/docs",
        "healthz": "/healthz",
    }


@app.post(
    "/api/pulses/deconvolve",
    response_model=DeconvolveSuccessResponse,
    # Optional evidence is omitted (not rendered as null) so requests without
    # saturationLevel keep the byte-for-byte original response shape.
    response_model_exclude_none=True,
    responses={
        422: {
            "model": InfeasibleResponse,
            "description": (
                "No extendible event train satisfies the residual/event-count "
                "limits (and the total-amplitude interval when given)."
            ),
        },
        400: {"description": "Request failed validation."},
    },
)
def deconvolve(req: DeconvolveRequest):
    result = solve(
        samples=req.samples,
        kernel=req.kernel,
        max_residual=req.maxResidual,
        max_amplitude=req.maxAmplitude,
        max_events=req.maxEvents,
        saturation_level=req.saturationLevel,
        total_amplitude_range=(
            tuple(req.totalAmplitudeRange)
            if req.totalAmplitudeRange is not None
            else None
        ),
    )
    if not result.feasible:
        body = InfeasibleResponse(
            firstUnexplainablePosition=result.first_unexplainable_position,
            totalAmplitudeRange=req.totalAmplitudeRange,
        )
        return JSONResponse(status_code=422, content=body.model_dump(exclude_none=True))
    events = [
        Event(position=i, amplitude=a)
        for i, a in enumerate(result.amplitudes)
        if a > 0
    ]
    saturation = None
    if result.saturation_evidence is not None:
        saturation = SaturationEvidenceResponse(
            totalAmplitude=result.saturation_evidence.total_amplitude,
            saturatedPoints=[
                SaturatedPointEvidence(
                    position=point.position,
                    sample=point.sample,
                    prediction=point.prediction,
                    excess=point.excess,
                )
                for point in result.saturation_evidence.saturated_points
            ],
        )
    return DeconvolveSuccessResponse(
        events=events,
        amplitudes=result.amplitudes,
        prediction=result.prediction,
        residuals=result.residuals,
        objectives=Objectives(
            maxAbsResidual=result.max_abs_residual,
            sumAbsResidual=result.sum_abs_residual,
            eventCount=result.event_count,
        ),
        saturation=saturation,
    )
