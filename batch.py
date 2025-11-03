import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="CJ Claim Batch Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train models up to a given year")
    train_parser.add_argument("--train-until", type=int, required=True)
    train_parser.add_argument("--engine", type=str, default="pandas")
    train_parser.add_argument("--series", type=str, default="all")

    # forecast
    forecast_parser = subparsers.add_parser("forecast", help="Forecast next year")
    forecast_parser.add_argument("--year", type=int, required=True)
    forecast_parser.add_argument("--series", type=str, default="all")
    forecast_parser.add_argument("--engine", type=str, default="pandas")

    # reconcile
    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile predictions with actuals")
    reconcile_parser.add_argument("--year", type=int, required=True)
    reconcile_parser.add_argument("--series", type=str, default="all")

    # roll
    roll_parser = subparsers.add_parser("roll", help="Run full rolling pipeline")
    roll_parser.add_argument("--start", type=int, required=True)
    roll_parser.add_argument("--end", type=int, required=True)
    roll_parser.add_argument("--series", type=str, default="all")
    roll_parser.add_argument("--engine", type=str, default="pandas")

    args = parser.parse_args()

    from pipeline_train import train_until
    from pipeline_forecast import forecast_year
    from pipeline_reconcile import reconcile_year
    curated = Path("data/curated/claims.parquet")

    if args.command == "train":
        train_until(curated, args.train_until)
    elif args.command == "forecast":
        forecast_year(curated, args.year)
    elif args.command == "reconcile":
        reconcile_year(curated, args.year)
    elif args.command == "roll":
        # Use dedicated rolling backtest script for comprehensive analysis
        from roll_backtest import run_rolling_backtest, analyze_trends, generate_rolling_report
        
        print(f"\n🔄 Rolling 백테스트 시작: {args.start} ~ {args.end}")
        
        metrics_df = run_rolling_backtest(
            curated_path=curated,
            start_year=args.start,
            end_year=args.end,
            series_filter=args.series,
            engine=args.engine
        )
        
        if not metrics_df.empty:
            trends_df = analyze_trends(metrics_df)
            
            # 결과 저장
            metrics_path = Path('artifacts/metrics')
            metrics_path.mkdir(parents=True, exist_ok=True)
            metrics_df.to_parquet(metrics_path / f'rolling_metrics_{args.start}_{args.end}.parquet', index=False)
            
            if not trends_df.empty:
                trends_df.to_parquet(metrics_path / f'rolling_trends_{args.start}_{args.end}.parquet', index=False)
            
            # 보고서 생성
            generate_rolling_report(metrics_df, trends_df, args.start, args.end, Path('reports'))
            
            print(f"\n✅ Rolling 백테스트 완료!")
        else:
            print(f"\n⚠️  메트릭이 계산되지 않았습니다.")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
