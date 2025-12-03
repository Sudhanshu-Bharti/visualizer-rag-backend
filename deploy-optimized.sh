#!/bin/bash
# Memory-optimized deployment script

# Set memory-efficient environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export FORCE_CPU="true"
export DEPLOYMENT_ENV="production"
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export HF_HOME="/tmp/hf_cache"
export OMP_NUM_THREADS="2"
export OPENBLAS_NUM_THREADS="2"
export MKL_NUM_THREADS="2"

# Create cache directories
mkdir -p /tmp/transformers_cache /tmp/hf_cache

# Install minimal requirements
pip install --no-cache-dir -r requirements-minimal.txt

# Clear pip cache
pip cache purge

# Start with memory optimizations
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --max-requests 50 \
    --max-requests-jitter 10 \
    --timeout-keep-alive 30