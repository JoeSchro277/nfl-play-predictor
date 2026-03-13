FROM python:3.12-slim

# Install system dependencies (Tesseract for OCR, libGL for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
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
# Railway sets PORT env var, default to 8080 for local Docker testing
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 2 app:app
