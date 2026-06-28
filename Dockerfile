# Single shared image for both services (Streamlit app + FastAPI). They run
# the same codebase against the same model artifacts, so one image with two
# different `command:` overrides in docker-compose.yml avoids building and
# maintaining two near-identical images.
FROM python:3.11-slim

WORKDIR /app

# libgomp1 provides the OpenMP runtime XGBoost/LightGBM need at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY credit_core.py app.py api.py monitoring.py ./
COPY models/ models/
COPY data/ data/

# Non-root user -- no reason to run either service as root.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8000

# Default to the Streamlit app; docker-compose overrides this for the API
# service. Runnable standalone too: `docker run -p 8501:8501 <image>`.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
