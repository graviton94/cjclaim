"""
월별 증분학습 파이프라인
발생일자 기준 1개월 데이터 처리:
1. Lag 필터링 (Normal + Borderline만)
2. 주간 집계 및 JSON 업데이트
3. 기존 예측 vs 실제 비교
4. 모델 재학습 (append_fit)
5. 로그 기록
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import argparse
import sys
import subprocess

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

class MonthlyIncrementalPipeline:
    def __init__(self, year, month, lag_stats_path="artifacts/metrics/lag_stats_from_raw.csv"):
        self.year = year
        self.month = month
        self.lag_stats = pd.read_csv(lag_stats_path)
        self.month_key = f"{year}{month:02d}"
        # 경로 설정
        self.base_dir = Path("artifacts/incremental")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.month_dir = self.base_dir / self.month_key
        self.month_dir.mkdir(parents=True, exist_ok=True)
        # Standardized file paths
        self.eval_path = self.month_dir / f"{self.month_key}_predict_vs_actual.csv"
        self.summary_path = self.month_dir / f"{self.month_key}_summary.json"
        self.retrain_results_path = self.month_dir / f"{self.month_key}_incremental_training_results.csv"
        print("=" * 80)
        print(f"월별 증분학습: {year}년 {month}월")
        print("=" * 80)
    
    def step1_filter_data(self, input_csv):
        """Step 1: Lag 필터링 (Normal + Borderline만)"""
        print(f"\n[Step 1] Lag 필터링")
        print(f"  입력: {input_csv}")
        # lag_analyzer 호출
        filtered_path = self.month_dir / f"candidates_{self.month_key}.csv"
        cmd = [
            "python", "tools/lag_analyzer.py",
            "--input", str(input_csv),
            "--ref", "artifacts/metrics/lag_stats_from_raw.csv",
            "--policy-out", str(filtered_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ❌ 오류: {result.stderr}")
            return None
        print(f"  ✅ 필터링 완료: {filtered_path}")
        return filtered_path

    def step2_monthly_aggregate_and_json(self, filtered_csv):
        print(f"\n[Step 2] 월간 집계 및 JSON 업데이트")
        df = pd.read_csv(filtered_csv, encoding='utf-8-sig')
        # 발생일자로 연월 계산
        df['발생일자'] = pd.to_datetime(df['발생일자'])
        df['year'] = df['발생일자'].dt.year
        df['month'] = df['발생일자'].dt.month
        # 시리즈별 그룹화 (원본 컬럼 유지)
        df['series_id'] = df['플랜트'] + '|' + df['제품범주2'] + '|' + df['중분류']
        # 월간 집계 (plant, product_cat2, mid_category 포함)
        monthly = df.groupby(['series_id', '플랜트', '제품범주2', '중분류', 'year', 'month']).agg({
            'count': 'sum',
            'lag_class': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'normal',
            'sample_weight': 'mean'
        }).reset_index()
        # summary 예시 (실제 summary 생성 로직 필요)
        summary = {
            "series_count": int(monthly['series_id'].nunique()),
            "record_count": int(len(df)),
        }
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 요약 저장: {self.summary_path}")
        print(f"  영향 받은 시리즈: {monthly['series_id'].nunique():,}개")
        # JSON 업데이트 (data/features 디렉토리 사용)
        json_dir = Path("data/features")
        json_dir.mkdir(parents=True, exist_ok=True)
        updated_count = 0
        created_count = 0

        for series_id in monthly['series_id'].unique():
            # 시리즈 정보 추출
            series_data = monthly[monthly['series_id'] == series_id].iloc[0]
            plant = series_data['플랜트']
            product_cat2 = series_data['제품범주2']
            mid_category = series_data['중분류']

            # 안전한 파일명
            safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                           .replace('|', '_').replace('?', '_').replace('*', '_')
                           .replace('<', '_').replace('>', '_').replace('"', '_'))
            json_path = json_dir / f"{safe_filename}.json"

            # 기존 데이터 로드 또는 신규 생성
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            else:
                # 신규 JSON 생성
                json_data = {"data": []}
                created_count += 1

            # 기존 데이터를 dict로 변환 (year, month를 키로)
            existing_data = {(row['year'], row['month']): row for row in json_data.get('data', [])}

            # 새 데이터 추가/업데이트
            series_monthly = monthly[monthly['series_id'] == series_id]
            for _, row in series_monthly.iterrows():
                year_month_key = (int(row['year']), int(row['month']))
                # 새 레코드 생성
                new_record = {
                    'year': int(row['year']),
                    'month': int(row['month']),
                    'claim_count': float(row.get('claim_count', row.get('count', 0))),
                    'lag_class': row['lag_class'],
                    'sample_weight': float(row['sample_weight'])
                }
                # 업데이트 또는 추가
                existing_data[year_month_key] = new_record

            # 정렬하여 저장
            json_data['data'] = sorted(existing_data.values(), key=lambda x: (x['year'], x['month']))

            # 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            updated_count += 1

        print(f"  ✅ {updated_count}개 JSON 파일 업데이트 (신규: {created_count}개)")

        # 증분 재학습 결과 파일 복사 (if exists)
        orig_results_path = Path("artifacts/models/base_monthly/incremental_training_results.csv")
        if orig_results_path.exists():
            import shutil
            shutil.copy(orig_results_path, self.retrain_results_path)
            print(f"  ✅ 증분 재학습 결과 저장: {self.retrain_results_path}")

        return monthly
    
    def step3_compare_forecast(self, monthly_actual):
        """Step 3: 기존 예측 vs 실제 비교 (월단위)"""
        print(f"\n[Step 3] 예측-실측 비교 (월단위)")

        # 예측 파일 경로 (월별)
        forecast_dir = Path("artifacts/forecasts")
        forecast_file = forecast_dir / f"forecast_{self.year}{self.month:02d}.parquet"

        if not forecast_file.exists():
            print(f"  ⚠️  예측 파일 없음: {forecast_file}")
            return None

        # 예측 로드
        df_forecast = pd.read_parquet(forecast_file)
        df_forecast = df_forecast[
            (df_forecast['year'] == self.year) &
            (df_forecast['month'] == self.month)
        ]

        # 실측 데이터와 병합 (월단위)
        comparison = monthly_actual.merge(
            df_forecast[['series_id', 'year', 'month', 'y_pred']],
            on=['series_id', 'year', 'month'],
            how='left'
        )

        # 오차 계산
        comparison['error'] = comparison['claim_count'] - comparison['y_pred'].fillna(0)
        comparison['abs_error'] = comparison['error'].abs()
        comparison['pct_error'] = np.where(
            comparison['claim_count'] > 0,
            comparison['error'] / comparison['claim_count'] * 100,
            0
        )

        # 로그 저장
        log_path = self.month_dir / f"predict_vs_actual_{self.month_key}.csv"
        comparison.to_csv(log_path, index=False, encoding='utf-8-sig')

        # 통계
        print(f"  비교 레코드: {len(comparison):,}개")
        print(f"  평균 오차: {comparison['error'].mean():.2f}")
        print(f"  절대 오차 평균: {comparison['abs_error'].mean():.2f}")
        print(f"  ✅ 로그 저장: {log_path}")

        return comparison
    
    def step4_retrain_models(self, updated_series, monthly_actual):
        """Step 4: 모델 재학습 (증분학습 - warm start, 월단위)"""
        print(f"\n[Step 4] 모델 재학습 (증분, 월단위)")
        print(f"  재학습 대상: {len(updated_series):,}개 시리즈")

        # train_incremental_models.py 호출
        cmd = [
            sys.executable, "train_incremental_models.py",
            "--year", str(self.year),
            "--month", str(self.month),
            "--max-workers", "4"
        ]

        print(f"  실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ❌ 재학습 실패:")
            print(result.stderr)
            return None

        print(result.stdout)
        print(f"  ✅ 재학습 완료")

        # 재학습 결과 로드 (train_incremental_models.py가 생성한 summary)
        summary_path = Path(f"artifacts/incremental/{self.year}{self.month:02d}") / "retrain_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                retrain_summary = json.load(f)
            print(f"  성공: {retrain_summary.get('success_count', 0)}개")
            print(f"  실패: {retrain_summary.get('fail_count', 0)}개")
            return retrain_summary

        return {"status": "completed"}
    
    def step5_generate_forecast(self):
        """Step 5: 새 예측 생성 (+6개월)"""
        print(f"\n[Step 5] 새 예측 생성 (+6개월)")
        
        # generate_monthly_forecast.py 호출
        cmd = [
            sys.executable, "generate_monthly_forecast.py",
            "--year", str(self.year),
            "--month", str(self.month),
            "--max-workers", "4"
        ]
        
        print(f"  실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ 예측 생성 실패:")
            print(result.stderr)
            return None
        
        print(result.stdout)
        print(f"  ✅ 예측 생성 완료")
        
        # 예측 결과 확인
        forecast_path = Path(f"artifacts/forecasts/forecast_{self.year}{self.month:02d}.parquet")
        if forecast_path.exists():
            import pandas as pd
            df_forecast = pd.read_parquet(forecast_path)
            print(f"  예측 레코드: {len(df_forecast):,}개")
            print(f"  시리즈 수: {df_forecast['series_id'].nunique():,}개")
            return {"status": "completed", "n_forecasts": len(df_forecast)}
        
        return {"status": "completed"}
    
    def step6_log_results(self, comparison):
        """Step 6: 결과 로그 기록"""
        print(f"\n[Step 6] 결과 로그")
        
        summary = {
            "year": self.year,
            "month": self.month,
            "processed_at": datetime.now().isoformat(),
            "total_records": int(len(comparison)) if comparison is not None else 0,
            "series_count": int(comparison['series_id'].nunique()) if comparison is not None else 0,
            "mean_error": float(comparison['error'].mean()) if comparison is not None else None,
            "mae": float(comparison['abs_error'].mean()) if comparison is not None else None,
        }
        
        log_path = self.month_dir / f"summary_{self.month_key}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 요약 저장: {log_path}")
        
        return summary
    
    def run(self, input_csv):
        """전체 파이프라인 실행"""
        try:
            # Step 1: Lag 필터링
            filtered_csv = self.step1_filter_data(input_csv)
            if filtered_csv is None:
                return False
            
            # Step 2: 월간 집계 및 JSON 업데이트
            monthly_actual = self.step2_monthly_aggregate_and_json(filtered_csv)
            
            # Step 3: 예측-실측 비교
            comparison = self.step3_compare_forecast(monthly_actual)
            
            # Step 4: 모델 재학습
            updated_series = monthly_actual['series_id'].unique()
            retrain_results = self.step4_retrain_models(updated_series, monthly_actual)
            
            # Step 5: 새 예측 생성
            forecast_results = self.step5_generate_forecast()
            
            # Step 6: 결과 로그
            self.step6_log_results(comparison)
            
            print(f"\n{'='*80}")
            print(f"✅ {self.year}년 {self.month}월 증분학습 완료!")
            print(f"{'='*80}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="월별 증분학습 파이프라인")
    parser.add_argument("--input", required=True, help="입력 CSV 파일 (발생일자 기준 1개월 데이터)")
    parser.add_argument("--year", type=int, required=True, help="연도")
    parser.add_argument("--month", type=int, required=True, help="월")
    args = parser.parse_args()
    
    pipeline = MonthlyIncrementalPipeline(args.year, args.month)
    success = pipeline.run(args.input)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
