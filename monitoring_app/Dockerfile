# AI Vision Monitoring System - Docker Configuration
# Multi-stage build for optimized image size

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Builder stage for compiling
FROM base AS builder

# Copy application code
COPY . .

# Train the audio model
RUN python audio_train.py --synthetic --samples-per-class 50 --output models/audio_classifier.pkl

# Production stage
FROM base AS production

# Copy from builder
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create data directories
RUN mkdir -p /app/data/{events,clips,logs} && \
    chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development stage
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy

USER appuser

# Override command for development
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]
