# Use multi-stage builds to optimize the final image size
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies into a local directory for the non-root user
# Using --user installs packages in the user's home directory, which doesn't require root privileges.
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Production Stage ---
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Add the user's local bin to the PATH. This is where pip installs executables.
ENV PATH="/home/appuser/.local/bin:$PATH"

# Create a non-root user to run the application
RUN useradd --create-home --shell /bin/bash appuser

# Copy the installed packages from the builder stage
COPY --from=builder /home/appuser/.local /home/appuser/.local

# Set the working directory
WORKDIR /app

# +++ Correctly copy the 'src' directory and other necessary files +++
# Copy the source code, chown to the non-root user
COPY --chown=appuser:appuser ./src ./src
# Copy other root files that might be needed
COPY --chown=appuser:appuser requirements.txt .env.example ./

# Create the logs directory and set ownership
RUN mkdir -p logs && chown appuser:appuser logs

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# +++ Fix Healthcheck to use a library that is actually installed (httpx) +++
# The original used 'requests', which is not in requirements.txt.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# +++ The command now correctly points to the app object inside the src package +++
# Use Gunicorn with Uvicorn workers for better performance and concurrency.
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
