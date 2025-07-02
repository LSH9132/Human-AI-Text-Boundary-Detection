#!/usr/bin/env python3
"""
CUDA PyTorch 설치 스크립트
GPU 환경에서 최적의 성능을 위한 CUDA 버전 PyTorch 설치
"""

import subprocess
import sys
import platform

def check_cuda_version():
    """시스템의 CUDA 버전 확인"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            if "CUDA Version:" in output:
                cuda_line = [line for line in output.split('\n') if 'CUDA Version:' in line][0]
                cuda_version = cuda_line.split('CUDA Version: ')[1].split()[0]
                return cuda_version
    except FileNotFoundError:
        pass
    return None

def install_cuda_pytorch(cuda_version=None):
    """CUDA 버전에 맞는 PyTorch 설치"""
    print("🔍 CUDA 환경을 확인합니다...")
    
    detected_cuda = check_cuda_version()
    if detected_cuda:
        print(f"   ✅ CUDA 버전 감지: {detected_cuda}")
    else:
        print("   ⚠️  CUDA가 설치되지 않았거나 감지할 수 없습니다.")
        print("   CPU 버전으로 진행하려면 install_packages.py를 사용하세요.")
        return False
    
    # CUDA 버전에 따른 PyTorch 설치 URL 결정
    if cuda_version is None:
        if detected_cuda:
            major_version = detected_cuda.split('.')[0]
            minor_version = detected_cuda.split('.')[1]
            
            if major_version == "12":
                if int(minor_version) >= 1:
                    cuda_version = "cu121"
                else:
                    cuda_version = "cu118"
            elif major_version == "11":
                if int(minor_version) >= 8:
                    cuda_version = "cu118"
                else:
                    cuda_version = "cu117"
            else:
                print(f"   ⚠️  지원되지 않는 CUDA 버전: {detected_cuda}")
                print("   CPU 버전으로 설치합니다.")
                cuda_version = "cpu"
        else:
            cuda_version = "cpu"
    
    print(f"📦 PyTorch CUDA 버전 설치: {cuda_version}")
    
    # 기본 패키지 먼저 설치
    basic_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "transformers>=4.20.0"
    ]
    
    print("📦 기본 패키지를 설치합니다...")
    for package in basic_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ {package.split('>=')[0]} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"   ❌ {package} 설치 실패")
            return False
    
    # PyTorch CUDA 버전 설치
    if cuda_version != "cpu":
        torch_url = f"https://download.pytorch.org/whl/{cuda_version}"
        torch_packages = ["torch", "torchvision", "torchaudio"]
        
        print(f"🚀 PyTorch CUDA ({cuda_version}) 버전을 설치합니다...")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + torch_packages + ["--index-url", torch_url]
            subprocess.check_call(cmd)
            print("   ✅ PyTorch CUDA 버전 설치 완료!")
        except subprocess.CalledProcessError:
            print("   ❌ PyTorch CUDA 설치 실패, CPU 버전으로 대체합니다.")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
                print("   ✅ PyTorch CPU 버전 설치 완료")
            except subprocess.CalledProcessError:
                print("   ❌ PyTorch 설치 실패")
                return False
    else:
        print("🔧 PyTorch CPU 버전을 설치합니다...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
            print("   ✅ PyTorch CPU 버전 설치 완료")
        except subprocess.CalledProcessError:
            print("   ❌ PyTorch 설치 실패")
            return False
    
    return True

def verify_installation():
    """설치 검증"""
    print("\n🔍 설치를 검증합니다...")
    
    try:
        import torch
        import transformers
        import pandas as pd
        import numpy as np
        import sklearn
        
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ Transformers: {transformers.__version__}")
        print(f"   ✅ Pandas: {pd.__version__}")
        print(f"   ✅ NumPy: {np.__version__}")
        print(f"   ✅ Scikit-learn: {sklearn.__version__}")
        
        # CUDA 확인
        if torch.cuda.is_available():
            print(f"   🚀 CUDA 사용 가능!")
            print(f"   📱 GPU: {torch.cuda.get_device_name()}")
            print(f"   💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            print(f"   🔢 CUDA 버전: {torch.version.cuda}")
            return True
        else:
            print(f"   ⚠️  CUDA 사용 불가 (CPU 모드)")
            return False
            
    except ImportError as e:
        print(f"   ❌ 패키지 import 오류: {e}")
        return False

def main():
    print("=" * 60)
    print("🤖 AI 텍스트 판별 프로젝트 - CUDA 환경 설정")
    print("=" * 60)
    
    # 운영체제 확인
    os_name = platform.system()
    print(f"🖥️  운영체제: {os_name}")
    print(f"🐍 Python: {sys.version}")
    
    # CUDA PyTorch 설치
    if install_cuda_pytorch():
        # 설치 검증
        cuda_available = verify_installation()
        
        print("\n" + "=" * 60)
        if cuda_available:
            print("🎉 CUDA 환경 설정 완료!")
            print("   다음 명령어로 GPU 가속 모델을 실행하세요:")
            print("   python main_cuda.py")
        else:
            print("⚠️  CPU 환경으로 설정됨")
            print("   다음 명령어로 CPU 모델을 실행하세요:")
            print("   python main.py")
        print("=" * 60)
        
        return cuda_available
    else:
        print("\n❌ 설치 실패!")
        return False

if __name__ == "__main__":
    main()