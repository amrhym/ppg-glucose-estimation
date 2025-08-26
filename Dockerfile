# Multi-stage build for PPG Glucose Estimation

# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY cli/ ./cli/
COPY configs/ ./configs/
COPY models/ ./models/

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 ppguser && \
    chown -R ppguser:ppguser /app

USER ppguser

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/healthz')"

# Default command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]