#!/usr/bin/env python3
"""
패키지 설치 스크립트
사용법: python3 install_packages.py
"""

import subprocess
import sys

def install_package(package):
    """패키지 설치"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 설치 완료")
    except subprocess.CalledProcessError:
        print(f"✗ {package} 설치 실패")
        return False
    return True

def main():
    packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "transformers>=4.20.0"
    ]
    
    print("필요한 패키지들을 설치합니다...")
    failed = []
    
    for package in packages:
        if not install_package(package):
            failed.append(package)
    
    if failed:
        print(f"\n설치 실패한 패키지들: {failed}")
        print("수동으로 설치해주세요.")
        return False
    
    print("\n모든 패키지 설치 완료!")
    return True

if __name__ == "__main__":
    main()