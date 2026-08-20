# Two stages: a build stage that compiles wheels, and a slim runtime that only
# carries the venv, the source and the trained artifacts.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-cache -e .

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CADENCE_LLM_PROVIDER=offline

# Non-root: the service needs no write access to anything it ships with.
RUN useradd --create-home --uid 10001 cadence
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=cadence:cadence src ./src
COPY --chown=cadence:cadence pyproject.toml README.md ./

# Artifacts and the processed catalog are produced by `make all` and mounted or
# baked in at deploy time; they are deliberately not built inside the image.
VOLUME ["/app/data/processed", "/app/artifacts"]

USER cadence
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "cadence.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
