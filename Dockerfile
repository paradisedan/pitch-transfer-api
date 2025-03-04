FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Parselmouth
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app