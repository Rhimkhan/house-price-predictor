# ═══════════════════════════════════════════════════════════
#  House Price Prediction — Dockerfile
#  Build:  docker build -t house-price-prediction .
#  Run:    docker run -p 5000:5000 house-price-prediction
# ═══════════════════════════════════════════════════════════

FROM python:3.11-slim

# Metadata
LABEL maintainer="Data Science Portfolio"
LABEL version="2.0.0"
LABEL description="House Price Prediction — Production ML App"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5000

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (layer cache optimisation)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create required directories
RUN mkdir -p outputs/eda_plots outputs/model_results outputs/shap_plots data

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Entry point
CMD ["python", "app.py"]
