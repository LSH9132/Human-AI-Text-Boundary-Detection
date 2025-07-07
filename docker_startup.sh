#!/bin/bash
# AI Text Detection Docker Container Startup Script
# 🚀 Automated initialization and training execution
# Author: Claude AI Assistant

set -e  # Exit on any error

echo "🚀 AI Text Detection Container Starting..."
echo "⏰ Container started at: $(date)"
echo "📦 Environment: ${ENVIRONMENT:-h100}"

# Display system information
echo ""
echo "🖥️  System Information:"
echo "   OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    echo "   GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "   GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'Unknown')"
fi
echo ""

# GPU Information
echo "📊 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
else
    echo "   nvidia-smi not available"
fi
echo ""

# Check data files
echo "🔍 Data Files Check:"
if [ -f "data/train.csv" ] && [ -f "data/test.csv" ]; then
    echo "   ✅ Training data found: $(wc -l < data/train.csv) lines"
    echo "   ✅ Test data found: $(wc -l < data/test.csv) lines"
    if [ -f "data/sample_submission.csv" ]; then
        echo "   ✅ Sample submission found: $(wc -l < data/sample_submission.csv) lines"
    fi
else
    echo "   ⚠️  Data files not found! Creating dummy files for testing..."
    mkdir -p data
    echo "id,title,full_text,generated" > data/train.csv
    echo "1,Test Title,Test content for training,0" >> data/train.csv
    echo "2,Test Title 2,Another test content,1" >> data/train.csv
    echo "id,title,full_text" > data/test.csv
    echo "1,Test Title,Test content for prediction" >> data/test.csv
    echo "id,generated" > data/sample_submission.csv
    echo "1,0" >> data/sample_submission.csv
    echo "   ⚠️  Using dummy data for testing purposes"
fi
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs submissions cache/huggingface
echo "   ✅ Directories created"
echo ""

# Environment-specific configuration
case "${ENVIRONMENT:-h100}" in
    "h100")
        echo "🏎️  H100 Environment Configuration:"
        echo "   Batch Size: 256"
        echo "   Max Length: 512"
        echo "   Mixed Precision: Enabled"
        echo "   Workers: 16"
        TRAINING_COMMAND="python main.py --env h100"
        ;;
    "gpu")
        echo "🎮 GPU Environment Configuration:"
        echo "   Batch Size: 32"
        echo "   Max Length: 256"
        echo "   Mixed Precision: Enabled"
        echo "   Workers: 8"
        TRAINING_COMMAND="python main_cuda.py"
        ;;
    "debug")
        echo "🐛 Debug Environment Configuration:"
        echo "   Epochs: 1"
        echo "   Splits: 2"
        echo "   Batch Size: 16"
        TRAINING_COMMAND="python main.py --env debug"
        ;;
    *)
        echo "🔧 Default Environment Configuration"
        TRAINING_COMMAND="python main.py"
        ;;
esac
echo ""

# Pre-training checks
echo "🔍 Pre-training System Check:"
python -c "
import torch
import transformers
from src.config import get_config_for_environment
import os

print(f'   PyTorch version: {torch.__version__}')
print(f'   Transformers version: {transformers.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')

# Test configuration
config = get_config_for_environment(os.getenv('ENVIRONMENT', 'h100'))
print(f'   Config loaded: {config.training.batch_size} batch size')
print(f'   Device: {config.system.device}')
"

if [ $? -eq 0 ]; then
    echo "   ✅ All systems ready"
else
    echo "   ❌ System check failed"
    exit 1
fi
echo ""

# Training execution
echo "🚀 Starting Training..."
echo "   Command: $TRAINING_COMMAND"
echo "   Expected completion: 1-2 hours (H100) / 4-6 hours (GPU)"
echo "   Monitor progress: tail -f logs/training.log"
echo ""

# Execute training with logging
exec $TRAINING_COMMAND 2>&1 | tee -a logs/container_training.log