FROM python:3.11-slim

WORKDIR /app

# Set memory-efficient environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV FORCE_CPU=true
ENV DEPLOYMENT_ENV=docker
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_cache

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Use minimal requirements for deployment
COPY requirements-minimal.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

COPY . .

# Create cache directories with proper permissions
RUN mkdir -p /tmp/transformers_cache /tmp/hf_cache \
    && chmod 777 /tmp/transformers_cache /tmp/hf_cache

EXPOSE 8000

# Use optimized startup command with memory limits
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--max-requests", "100", "--max-requests-jitter", "10"]