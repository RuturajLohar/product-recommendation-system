FROM python:3.11-slim

WORKDIR /app

# Keep Python logs visible in Cloud Run and avoid writing pyc files into containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install minimal system dependencies for ML wheels and Postgres.
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install heavy dependencies to leverage Docker cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source.
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
