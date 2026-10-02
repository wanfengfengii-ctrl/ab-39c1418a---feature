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
    SaturationEvidence,
)

app = FastAPI(
    title="Scintillator Pulse Deconvolution",
    version="1.0.0",
    description=(
        "Joint non-negative integer deconvolution of overlapping scintillator "
        "pulses: recovers event positions/amplitudes, the predicted waveform "
        "and pointwise residuals from discrete samples."
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
        "version": "1.0.0",
        "docs": "/docs",
        "healthz": "/healthz",
    }


@app.post(
    "/api/pulses/deconvolve",
    response_model=DeconvolveSuccessResponse,
    responses={
        422: {
            "model": InfeasibleResponse,
            "description": "No event train satisfies the residual/event-count limits.",
        },
        400: {"description": "Request failed validation."},
    },
)
def deconvolve(req: DeconvolveRequest):
    clipping = req.saturationLevel is not None
    result = solve(
        samples=req.samples,
        kernel=req.kernel,
        max_residual=req.maxResidual,
        max_amplitude=req.maxAmplitude,
        max_events=req.maxEvents,
        saturation_level=req.saturationLevel,
        total_amplitude_range=(
            tuple(req.totalAmplitudeRange) if clipping else None
        ),
    )
    if not result.feasible:
        if result.constraint == "budget":
            body = InfeasibleResponse(
                detail=(
                    "the clipping constraints leave too many admissible event "
                    "trains for exact joint inversion within the solver state "
                    "budget; tighten maxEvents or totalAmplitudeRange"
                ),
                firstUnexplainablePosition=result.first_unexplainable_position,
                constraint="stateBudgetExceeded",
                totalAmplitudeRange=req.totalAmplitudeRange,
            )
        elif clipping:
            detail = (
                "no event train within the amplitude, event-count and "
                "total-amplitude limits can reproduce the samples "
                "(pointwise clipping at the saturation level) within the "
                "residual limit"
            )
            body = InfeasibleResponse(
                detail=detail,
                firstUnexplainablePosition=result.first_unexplainable_position,
                constraint=(
                    "totalAmplitudeRange"
                    if result.constraint == "total"
                    else "residualLimit"
                ),
                totalAmplitudeRange=req.totalAmplitudeRange,
            )
        else:
            body = InfeasibleResponse(
                firstUnexplainablePosition=result.first_unexplainable_position
            )
        # exclude_none keeps the legacy 422 body byte-for-byte unchanged.
        return JSONResponse(
            status_code=422, content=body.model_dump(exclude_none=True)
        )
    events = [
        Event(position=i, amplitude=a)
        for i, a in enumerate(result.amplitudes)
        if a > 0
    ]
    response = DeconvolveSuccessResponse(
        events=events,
        amplitudes=result.amplitudes,
        prediction=result.prediction,
        residuals=result.residuals,
        objectives=Objectives(
            maxAbsResidual=result.max_abs_residual,
            sumAbsResidual=result.sum_abs_residual,
            eventCount=result.event_count,
        ),
    )
    if clipping:
        response.totalAmplitude = result.total_amplitude
        response.totalAmplitudeRange = req.totalAmplitudeRange
        response.saturationEvidence = [
            SaturationEvidence(
                position=pos,
                observed=req.saturationLevel,
                predicted=result.prediction[pos],
                excess=result.saturation_excess[i],
            )
            for i, pos in enumerate(result.saturation_positions)
        ]
    # exclude_none keeps the legacy 200 body byte-for-byte unchanged.
    return JSONResponse(
        status_code=200, content=response.model_dump(exclude_none=True)
    )
