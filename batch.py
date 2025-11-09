"""
CJ Claim Batch Pipeline
통합 배치 처리 CLI - 학습, 예측, 보정, 월별 증분학습 자동화
"""
import argparse
import sys
from pathlib import Path
import subprocess
import json

def main():
    parser = argparse.ArgumentParser(description="CJ Claim Batch Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train models up to a given year")
    train_parser.add_argument("--train-until", type=int, required=True)
    train_parser.add_argument("--engine", type=str, default="pandas")
    train_parser.add_argument("--series", type=str, default="all")

    # forecast 서브커맨드
    forecast_parser = subparsers.add_parser('forecast', help='예측 생성')
    forecast_parser.add_argument('--year', type=int, help='예측 대상 연도 (기존 방식)')
    forecast_parser.add_argument('--month-new', type=str, help='대상 월 YYYY-MM (새로운 월별 예측)')
    forecast_parser.add_argument('--series', type=str, default='all')
    forecast_parser.add_argument('--engine', type=str, default='pandas')

    # reconcile
    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile predictions with actuals")
    reconcile_parser.add_argument("--year", type=int, help="연도 (기존 방식)")
    reconcile_parser.add_argument("--series", type=str, default="all")
    reconcile_parser.add_argument("--month-new", type=str, help="대상 월 YYYY-MM (새로운 월별 Reconcile)")
    reconcile_parser.add_argument("--stage-new", choices=['bias', 'seasonal', 'optuna', 'all'],
                                 default='all', help="보정 단계 (새로운 월별용)")

    # roll
    roll_parser = subparsers.add_parser("roll", help="Run full rolling pipeline")
    roll_parser.add_argument("--start", type=int, required=True)
    roll_parser.add_argument("--end", type=int, required=True)
    roll_parser.add_argument("--series", type=str, default="all")
    roll_parser.add_argument("--engine", type=str, default="pandas")
    
    # process - 월별 증분학습 파이프라인
    process_parser = subparsers.add_parser("process", help="Process monthly incremental data")
    process_parser.add_argument("--upload", type=str, required=True, help="업로드된 월별 CSV 파일")
    process_parser.add_argument("--month", type=str, required=True, help="대상 월 (YYYY-MM)")
    
    # retrain - 증분 재학습
    retrain_parser = subparsers.add_parser("retrain", help="Incremental model retraining")
    retrain_parser.add_argument("--month", type=str, required=True, help="대상 월 (YYYY-MM)")
    retrain_parser.add_argument("--workers", type=int, default=4, help="병렬 처리 worker 수")
    retrain_parser.add_argument("--series-list", type=str, help="재학습 대상 시리즈 리스트(txt)")

    args = parser.parse_args()

    # process와 retrain은 별도의 스크립트를 호출하므로 import 불필요
    if args.command in ["process", "retrain"]:
        pass  # 아래에서 subprocess로 실행
    else:
        # 기존 파이프라인용 import (주간 기반)
        from pipeline_train import train_until
        from pipeline_forecast import forecast_year
        from pipeline_reconcile import reconcile_year
    
    curated = Path("data/curated/claims.parquet")

    if args.command == "train":
        train_until(curated, args.train_until)
    elif args.command == "forecast":
        # 예측 생성
        if hasattr(args, 'month_new') and args.month_new:
            # 새로운 월별 예측
            print("=" * 80)
            print("월별 예측 생성")
            print("=" * 80)

            year, month = args.month_new.split('-')

            # Check for cycle_features.parquet and generate if missing
            features_parquet = Path("data/features/cycle_features.parquet")
            if not features_parquet.exists():
                print(f"[INFO] cycle_features.parquet not found. Generating...")
                gen_cmd = [sys.executable, "tools/generate_cycle_features_parquet.py"]
                gen_result = subprocess.run(gen_cmd)
                if gen_result.returncode != 0:
                    print("❌ Failed to generate cycle_features.parquet.")
                    sys.exit(1)

            cmd = [
                sys.executable, "generate_monthly_forecast.py",
                "--year", year,
                "--month", month,
                "--max-workers", "4"
            ]

            print(f"\n명령: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
        else:
            # 기존 forecast (연도 전체)
            forecast_year(curated, args.year)
    elif args.command == "reconcile":
        # 기존 reconcile (연도 기반) - 기존 로직 유지
        if hasattr(args, 'month_new') and args.month_new:
            # 새로운 월별 Reconcile 파이프라인
            print("=" * 80)
            print("Reconcile 보정 (월별)")
            print("=" * 80)
            
            year, month = args.month_new.split('-')
            
            cmd = [
                sys.executable, "reconcile_pipeline.py",
                "--year", year,
                "--month", month,
                "--stage", getattr(args, 'stage_new', 'all')
            ]
            
            print(f"\n명령: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
        else:
            # 기존 reconcile (연도 전체)
            reconcile_year(curated, args.year)
    elif args.command == "roll":
        # Rolling backtest: 연도별 학습 → 예측 → 검증
        print(f"\n🔄 Rolling 백테스트 시작: {args.start} ~ {args.end}")
        
        from pipeline_train import train_until
        from pipeline_forecast import forecast_year
        
        for year in range(args.start, args.end):
            print(f"\n{'='*60}")
            print(f"[YEAR] {year} → {year+1} 파이프라인")
            print(f"{'='*60}")
            
            print(f"\n[1/3] {year}까지 학습...")
            train_until(curated, year)
            
            print(f"\n[2/3] {year+1} 예측...")
            forecast_year(curated, year+1)
            
            print(f"\n[3/3] {year+1} 보정 (Bias Map)...")
            reconcile_year(curated, year+1)
        
        print("\n✅ Rolling 백테스트 완료!")
        print(f"📊 결과 위치: artifacts/forecasts/, artifacts/adjustments/")
    
    elif args.command == "process":
        # 월별 증분학습 파이프라인
        print("=" * 80)
        print("월별 데이터 처리 (Incremental Update)")
        print("=" * 80)
        
        year, month = args.month.split('-')
        print(f"\n입력 파일: {args.upload}")
        print(f"대상 월: {year}년 {month}월")
        
        # process_monthly_incremental.py 실행
        cmd = [
            sys.executable, "process_monthly_incremental.py",
            "--new-csv", args.upload,
            "--year", year,
            "--month", month,
            "--output-list", f"artifacts/temp/updated_series_{year}{month}.txt"
        ]
        
        print(f"\n명령: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("[SUCCESS] Monthly incremental update completed!")
            print("=" * 80)
        
        sys.exit(result.returncode)
    
    elif args.command == "retrain":
        # 증분 재학습
        print("=" * 80)
        print("증분 재학습")
        print("=" * 80)

        year, month = args.month.split('-')
        print(f"\n대상 월: {year}년 {month}월")
        print(f"Workers: {args.workers}")

        # Check for cycle_features.parquet and generate if missing
        features_parquet = Path("data/features/cycle_features.parquet")
        if not features_parquet.exists():
            print(f"[INFO] cycle_features.parquet not found. Generating...")
            gen_cmd = [sys.executable, "tools/generate_cycle_features_parquet.py"]
            gen_result = subprocess.run(gen_cmd)
            if gen_result.returncode != 0:
                print("❌ Failed to generate cycle_features.parquet.")
                sys.exit(1)

        cmd = [
            sys.executable, "train_incremental_models.py",
            "--year", year,
            "--month", month,
            "--max-workers", str(args.workers)
        ]

        print(f"\n명령: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        sys.exit(result.returncode)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
