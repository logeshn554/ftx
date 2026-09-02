FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY dashboard/ dashboard/

# Install Python dependencies
RUN pip install --no-cache-dir -e "."

# Create logs directory
RUN mkdir -p logs checkpoints

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run trading system
CMD ["python", "scripts/trade.py", "--config", "config/production.yaml"]
