#!/bin/bash
# vast.ai H100 인스턴스 자동 설정 및 훈련 시작 스크립트
# 작성자: Claude AI Assistant
# 용도: AI Text Detection 프로젝트 자동 배포

set -e  # 오류 발생 시 스크립트 중단

echo "🚀 Starting AI Text Detection training on H100..."
echo "⏰ Started at: $(date)"

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=true
export HF_HOME=/tmp/huggingface_cache

echo "🔧 Environment variables set"

# GPU 정보 확인
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# 프로젝트 클론 (이미 존재하면 pull)
if [ ! -d "Human-AI-Text-Boundary-Detection" ]; then
    echo "📁 Cloning repository..."
    git clone https://github.com/LSH9132/Human-AI-Text-Boundary-Detection.git
else
    echo "📁 Repository exists, pulling latest changes..."
    cd Human-AI-Text-Boundary-Detection
    git pull origin master
    cd ..
fi

cd Human-AI-Text-Boundary-Detection

# PyTorch GPU 버전 설치
echo "📦 Installing PyTorch GPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성 설치
echo "📦 Installing other dependencies..."
pip install transformers>=4.35.0 datasets>=2.14.0 scikit-learn>=1.3.0 pandas>=2.0.0 numpy>=1.24.0 tqdm>=4.65.0 tokenizers>=0.15.0

# 데이터 디렉토리 생성
mkdir -p data logs models submissions

echo "✅ Setup completed successfully!"
echo ""

# 훈련 시작 전 최종 확인
echo "🔍 Final system check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
echo ""

# 데이터 파일 존재 확인
if [ ! -f "data/train.csv" ] || [ ! -f "data/test.csv" ]; then
    echo "⚠️  Warning: Data files not found in data/ directory"
    echo "Please upload your data files (train.csv, test.csv, sample_submission.csv) to the data/ directory"
    echo "You can use: scp your_data_files user@instance_ip:/path/to/project/data/"
    echo ""
    echo "For now, creating dummy data files for testing..."
    echo "id,title,full_text,generated" > data/train.csv
    echo "1,Test Title,Test content for training,0" >> data/train.csv
    echo "id,title,full_text" > data/test.csv
    echo "1,Test Title,Test content for prediction" >> data/test.csv
    echo "id,generated" > data/sample_submission.csv
    echo "1,0" >> data/sample_submission.csv
fi

echo "🚀 Starting H100 optimized training..."
echo "Expected completion time: 1-2 hours"
echo "Monitor progress with: tail -f gpu_optimized_training.log"
echo ""

# H100 최적화 훈련 시작
python main.py --env h100

echo "🎉 Training completed successfully!"
echo "⏰ Finished at: $(date)"
echo "📁 Results saved in: submission.csv"
echo "📁 Model checkpoints in: models/"