FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies natively
# We use minimal strict versions for the openenv environment
RUN pip install --no-cache-dir pydantic>=2.0.0 gymnasium>=0.28.1 openai>=1.0.0

# Copy application files
COPY . /app

# Ensure python paths are correct for local imports
ENV PYTHONPATH=/app

# Expose standard port for any openenv server interface
EXPOSE 7860

# CMD configuration for Hugging Face Spaces (default entry point)
CMD ["python", "scripts/baseline_inference.py"]
