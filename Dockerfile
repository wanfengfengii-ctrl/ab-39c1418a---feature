# syntax=docker/dockerfile:1

# Shared base: runtime dependencies plus the application package.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home appuser
COPY --chown=appuser:appuser app ./app

# API service image.
FROM base AS api
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# One-shot verification image: dev dependencies, test suite and smoke checks.
FROM base AS verify
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pyproject.toml ./
COPY --chown=appuser:appuser tests ./tests
COPY --chown=appuser:appuser verify ./verify
USER appuser
CMD ["python", "-m", "verify.run_checks"]
