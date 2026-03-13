FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Download YOLO model at build time (so it's cached in the image)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose port (Railway sets PORT env var)
EXPOSE 8080

# Run with gunicorn for production
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 2 app:app
