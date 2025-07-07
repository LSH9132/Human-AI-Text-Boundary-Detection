# AI Text Detection Docker Image
# Optimized for H100 GPUs with CUDA 12.1 support
# 🚀 Ready-to-deploy image with all dependencies pre-installed

# Multi-stage build for optimized image size
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (critical for H100)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=true
ENV HF_HOME=/app/cache/huggingface

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p data logs models submissions cache/huggingface

# Copy project files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py ./
COPY main_cuda.py ./
COPY requirements.txt ./
COPY docker_startup.sh ./
COPY README.md ./

# Make startup script executable
RUN chmod +x docker_startup.sh

# Create non-root user for security
RUN useradd -m -u 1000 aiuser && \
    chown -R aiuser:aiuser /app

# Switch to non-root user
USER aiuser

# Expose port for monitoring (if needed)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["./docker_startup.sh"]