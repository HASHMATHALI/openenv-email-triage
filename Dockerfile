FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH=/app     PYTHONUNBUFFERED=1

# Hugging Face Spaces routes traffic to port 7860.
EXPOSE 7860

# Serves the interactive dashboard. To run the headless benchmark instead:
#   docker run --rm openenv-email-triage python scripts/baseline_inference.py
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
