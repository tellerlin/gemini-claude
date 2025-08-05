# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# Stage 2: Final Image
FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /install /install

# Add the /install/bin directory to the PATH
ENV PATH=/install/bin:$PATH

# Add the site-packages directory to PYTHONPATH
ENV PYTHONPATH=/install/lib/python3.11/site-packages

WORKDIR /app
COPY --chown=appuser:appuser ./src .

RUN mkdir -p logs && chown appuser:appuser logs

USER appuser

EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:8000", "--log-level", "info", "--access-logfile", "logs/access.log", "--error-logfile", "logs/error.log", "main:app"]
