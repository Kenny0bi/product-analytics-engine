FROM python:3.11-slim

WORKDIR /app

# System dependencies required to build some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .
RUN chmod +x docker/entrypoint.sh

# The database is seeded on first boot by the entrypoint (see
# docker/entrypoint.sh), not at build time: the compose data volume would
# shadow a baked-in file, and runtime seeding keeps the image build fast.

# FastAPI on 8000, Streamlit on 8501
EXPOSE 8000 8501

ENTRYPOINT ["docker/entrypoint.sh"]

# Run both the API server and the Streamlit dashboard
CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
