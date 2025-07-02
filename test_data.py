#!/usr/bin/env python3
"""
데이터 검증 스크립트 - 패키지 없이도 실행 가능
"""
import csv
import sys

def test_data_structure():
    """데이터 구조 검증"""
    print("데이터 구조를 검증합니다...")
    
    # 1. train.csv 검증
    try:
        with open('data/train.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"Train CSV 헤더: {headers}")
            
            sample_rows = []
            for i, row in enumerate(reader):
                if i < 3:  # 첫 3행만 확인
                    sample_rows.append(row)
                if i >= 100:  # 100행만 확인
                    break
            
            print(f"Train 데이터 샘플:")
            for i, row in enumerate(sample_rows):
                print(f"  행 {i+1}: 제목={row['title'][:30]}..., 라벨={row['generated']}")
                
    except Exception as e:
        print(f"Train 데이터 오류: {e}")
        return False
    
    # 2. test.csv 검증
    try:
        with open('data/test.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"\nTest CSV 헤더: {headers}")
            
            sample_rows = []
            for i, row in enumerate(reader):
                if i < 3:
                    sample_rows.append(row)
                if i >= 10:
                    break
            
            print(f"Test 데이터 샘플:")
            for i, row in enumerate(sample_rows):
                print(f"  행 {i+1}: ID={row['ID']}, 제목={row['title'][:30]}...")
                
    except Exception as e:
        print(f"Test 데이터 오류: {e}")
        return False
    
    # 3. sample_submission.csv 검증
    try:
        with open('data/sample_submission.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"\nSubmission CSV 헤더: {headers}")
            
            row_count = sum(1 for _ in reader)
            print(f"Submission 파일 행 수: {row_count}")
                
    except Exception as e:
        print(f"Submission 데이터 오류: {e}")
        return False
    
    print("\n✓ 모든 데이터 파일 구조가 올바릅니다!")
    return True

def count_lines():
    """각 파일의 행 수 확인"""
    files = ['data/train.csv', 'data/test.csv', 'data/sample_submission.csv']
    
    print("\n파일별 행 수:")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8-sig') as f:
                line_count = sum(1 for _ in f) - 1  # 헤더 제외
                print(f"  {file}: {line_count:,}행")
        except Exception as e:
            print(f"  {file}: 오류 - {e}")

def main():
    print("=" * 50)
    print("AI 텍스트 판별 프로젝트 - 데이터 검증")
    print("=" * 50)
    
    if test_data_structure():
        count_lines()
        print("\n모든 검증 완료!")
        return True
    else:
        print("\n검증 실패!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)