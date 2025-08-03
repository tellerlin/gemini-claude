# [cite_start]Use multi-stage builds to optimize the final image size [cite: 6]
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

# Install Python dependencies. [cite_start]When run as root, --user installs to /root/.local [cite: 7]
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Production Stage ---
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Add the user's local bin to the PATH. [cite_start]This is where pip installs executables. [cite: 8]
ENV PATH="/home/appuser/.local/bin:$PATH"

# Create a non-root user to run the application
RUN useradd --create-home --shell /bin/bash appuser

# Copy the installed packages from the builder stage
# [cite_start]The source path is /root/.local because the previous stage ran as root. [cite: 9]
COPY --from=builder /root/.local /home/appuser/.local

# Set the working directory
WORKDIR /app

# --- MODIFIED: Copy all source files from the build context root ---
# Copy the source code, chown to the non-root user
COPY --chown=appuser:appuser . .

# Create the logs directory and set ownership
RUN mkdir -p logs && chown appuser:appuser logs

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Fix Healthcheck to use a library that is guaranteed to be available (urllib)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=10).read()" || exit 1

# --- MODIFIED: The command now correctly points to the app object in main.py ---
# Use Gunicorn with Uvicorn workers, and read the number of workers from the
# [cite_start]SERVICE_WORKERS environment variable, defaulting to 1 if not set. [cite: 11]
# CMD gunicorn main:app -w ${SERVICE_WORKERS:-1} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# NEW, MORE PRECISE COMMAND FOR DEBUGGING
CMD ["python", "-c", "import main; print(main.app)"]
