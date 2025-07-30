# Dockerfile

# Use the official Python 3.11 slim image as a base.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the dependency file first to leverage Docker's caching.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code.
COPY . .

# Create a non-privileged user to run the application.
# This is a security best practice.
RUN useradd --create-home --shell /bin/bash appuser

# Switch to the non-privileged user.
USER appuser

# Expose the port the app will run on.
EXPOSE 8000

# The command to start the application using a production-ready server.
# Corrected "main:app" to match the actual file structure.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
