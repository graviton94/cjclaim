"""
예측 가능성 분석 도구

학습된 모델들을 분석하여:
1. 예측 가능한 시리즈 vs 불가능한 시리즈 분류
2. 집중해야 할 시리즈 추천
3. 상세 분석 보고서 생성

실행 예시:
    python tools/analyze_predictability.py
    python tools/analyze_predictability.py --threshold 0.6
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_all_models(models_dir: Path) -> pd.DataFrame:
    """모든 학습된 모델 메타데이터 로드"""
    
    model_files = list(models_dir.glob("*.json"))
    
    if not model_files:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {models_dir}")
    
    models_data = []
    
    for model_file in model_files:
        with open(model_file, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
            # 실패한 모델 스킵
            if model_data.get('status') == 'failed':
                continue
            
            models_data.append({
                'series_id': model_data['series_id'],
                'model_type': model_data['model_type'],
                'n_train_points': model_data['n_train_points'],
                'predictability_score': model_data.get('predictability_score', 0.5),
                'is_sparse': model_data['guard_results']['is_sparse'],
                'zero_ratio': model_data['guard_results']['zero_ratio'],
                'has_drift': model_data['guard_results']['has_drift'],
                'has_seasonality': model_data['guard_results']['has_seasonality'],
                'seasonality_strength': model_data['guard_results']['seasonality_strength'],
                'mean': model_data['historical_stats']['mean'],
                'std': model_data['historical_stats']['std'],
                'nonzero_pct': model_data['historical_stats']['nonzero_pct'],
            })
    
    return pd.DataFrame(models_data)


def classify_series(df: pd.DataFrame, 
                    high_threshold: float = 0.7,
                    low_threshold: float = 0.4) -> pd.DataFrame:
    """
    시리즈를 예측 가능성에 따라 분류
    
    Args:
        df: 모델 데이터프레임
        high_threshold: 높은 예측 가능성 임계값
        low_threshold: 낮은 예측 가능성 임계값
    
    Returns:
        분류 결과가 추가된 데이터프레임
    """
    df = df.copy()
    
    def get_category(score):
        if score >= high_threshold:
            return 'high'
        elif score >= low_threshold:
            return 'medium'
        else:
            return 'low'
    
    df['predictability_category'] = df['predictability_score'].apply(get_category)
    
    # 집중 권장 플래그
    df['focus_recommended'] = (
        (df['predictability_category'] == 'high') &
        (df['mean'] > df['mean'].quantile(0.25))  # 충분한 볼륨
    )
    
    return df


def generate_analysis_report(df: pd.DataFrame, 
                             output_path: Path,
                             high_threshold: float,
                             low_threshold: float):
    """예측 가능성 분석 보고서 생성"""
    
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / 'predictability_analysis.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 예측 가능성 분석 보고서\n\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. 전체 요약
        f.write("## 📊 전체 요약\n\n")
        
        total = len(df)
        high_count = (df['predictability_category'] == 'high').sum()
        medium_count = (df['predictability_category'] == 'medium').sum()
        low_count = (df['predictability_category'] == 'low').sum()
        focus_count = df['focus_recommended'].sum()
        
        f.write(f"- **총 시리즈 수**: {total:,}개\n")
        f.write(f"- **평균 예측 가능성 스코어**: {df['predictability_score'].mean():.3f}\n\n")
        
        f.write("### 예측 가능성 분포\n\n")
        f.write(f"- 🟢 **높음** (≥{high_threshold}): {high_count:,}개 ({high_count/total*100:.1f}%)\n")
        f.write(f"- 🟡 **중간** ({low_threshold}~{high_threshold}): {medium_count:,}개 ({medium_count/total*100:.1f}%)\n")
        f.write(f"- 🔴 **낮음** (<{low_threshold}): {low_count:,}개 ({low_count/total*100:.1f}%)\n\n")
        
        f.write(f"### 🎯 집중 권장 시리즈\n\n")
        f.write(f"**{focus_count:,}개** 시리즈에 집중하는 것을 권장합니다.\n")
        f.write("(높은 예측 가능성 + 충분한 볼륨)\n\n")
        
        # 2. 모델 타입별 분석
        f.write("## 🔧 모델 타입별 예측 가능성\n\n")
        
        model_analysis = df.groupby('model_type').agg({
            'predictability_score': ['mean', 'count'],
            'series_id': 'count'
        }).round(3)
        
        f.write("| 모델 타입 | 시리즈 수 | 평균 스코어 |\n")
        f.write("|----------|-----------|-------------|\n")
        
        for model_type in df['model_type'].unique():
            model_df = df[df['model_type'] == model_type]
            count = len(model_df)
            avg_score = model_df['predictability_score'].mean()
            f.write(f"| {model_type} | {count} | {avg_score:.3f} |\n")
        
        f.write("\n")
        
        # 3. 집중 권장 시리즈 목록
        f.write("## 🎯 집중 권장 시리즈 (Top 50)\n\n")
        
        focus_series = df[df['focus_recommended'] == True].sort_values(
            'predictability_score', ascending=False
        ).head(50)
        
        if len(focus_series) > 0:
            f.write("| 순위 | Series ID | 스코어 | 모델 | 평균값 | 계절성 |\n")
            f.write("|------|-----------|--------|------|--------|--------|\n")
            
            for rank, (_, row) in enumerate(focus_series.iterrows(), 1):
                seasonality = "✓" if row['has_seasonality'] else "✗"
                f.write(f"| {rank} | {row['series_id']} | {row['predictability_score']:.3f} | {row['model_type']} | {row['mean']:.1f} | {seasonality} |\n")
            
            f.write("\n")
        else:
            f.write("집중 권장 시리즈가 없습니다.\n\n")
        
        # 4. 문제 시리즈 (낮은 예측 가능성)
        f.write("## ⚠️ 예측 어려운 시리즈 (낮은 스코어 Top 30)\n\n")
        
        problem_series = df[df['predictability_category'] == 'low'].sort_values(
            'predictability_score'
        ).head(30)
        
        if len(problem_series) > 0:
            f.write("| Series ID | 스코어 | 주요 문제 | 모델 |\n")
            f.write("|-----------|--------|-----------|------|\n")
            
            for _, row in problem_series.iterrows():
                issues = []
                if row['is_sparse']:
                    issues.append(f"희소({row['zero_ratio']*100:.0f}%)")
                if row['has_drift']:
                    issues.append("드리프트")
                if not row['has_seasonality']:
                    issues.append("계절성↓")
                
                issue_str = ", ".join(issues) if issues else "-"
                f.write(f"| {row['series_id']} | {row['predictability_score']:.3f} | {issue_str} | {row['model_type']} |\n")
            
            f.write("\n")
        
        # 5. 특성별 분석
        f.write("## 📈 시리즈 특성 분석\n\n")
        
        f.write("### 희소도 영향\n\n")
        sparse_df = df.groupby('is_sparse')['predictability_score'].agg(['mean', 'count'])
        f.write(f"- 희소 시리즈: 평균 스코어 {sparse_df.loc[True, 'mean']:.3f} ({sparse_df.loc[True, 'count']}개)\n")
        f.write(f"- 밀집 시리즈: 평균 스코어 {sparse_df.loc[False, 'mean']:.3f} ({sparse_df.loc[False, 'count']}개)\n\n")
        
        f.write("### 계절성 영향\n\n")
        seasonal_df = df.groupby('has_seasonality')['predictability_score'].agg(['mean', 'count'])
        f.write(f"- 계절성 있음: 평균 스코어 {seasonal_df.loc[True, 'mean']:.3f} ({seasonal_df.loc[True, 'count']}개)\n")
        f.write(f"- 계절성 없음: 평균 스코어 {seasonal_df.loc[False, 'mean']:.3f} ({seasonal_df.loc[False, 'count']}개)\n\n")
        
        f.write("### 드리프트 영향\n\n")
        drift_df = df.groupby('has_drift')['predictability_score'].agg(['mean', 'count'])
        if True in drift_df.index and False in drift_df.index:
            f.write(f"- 드리프트 있음: 평균 스코어 {drift_df.loc[True, 'mean']:.3f} ({drift_df.loc[True, 'count']}개)\n")
            f.write(f"- 드리프트 없음: 평균 스코어 {drift_df.loc[False, 'mean']:.3f} ({drift_df.loc[False, 'count']}개)\n\n")
        
        # 6. 추천 사항
        f.write("## 💡 추천 사항\n\n")
        
        f.write(f"### 즉시 활용 가능 ({high_count}개 시리즈)\n")
        f.write(f"높은 예측 가능성 시리즈는 즉시 프로덕션 예측에 활용하세요.\n\n")
        
        if medium_count > 0:
            f.write(f"### 개선 후 활용 ({medium_count}개 시리즈)\n")
            f.write("중간 예측 가능성 시리즈는 다음 방법으로 개선:\n")
            f.write("- Bias 보정 적용\n")
            f.write("- Seasonal Recalibration\n")
            f.write("- Optuna 하이퍼파라미터 튜닝\n\n")
        
        if low_count > 0:
            f.write(f"### 대체 방법 고려 ({low_count}개 시리즈)\n")
            f.write("낮은 예측 가능성 시리즈는:\n")
            f.write("- 단순 Naive 예측 사용\n")
            f.write("- 도메인 전문가 의견 활용\n")
            f.write("- 추가 외부 변수 수집 고려\n\n")
        
        f.write("---\n\n")
        f.write("## 다음 단계\n\n")
        f.write("1. **집중 시리즈 CSV 생성**: 높은 예측 가능성 시리즈 목록\n")
        f.write("2. **문제 시리즈 분석**: 예측 어려운 시리즈 원인 파악\n")
        f.write("3. **선택적 튜닝**: 중간 카테고리 시리즈만 Optuna 적용\n\n")
    
    print(f"\n✅ 보고서 생성 완료: {report_file}")


def save_category_lists(df: pd.DataFrame, output_path: Path):
    """카테고리별 시리즈 목록 CSV 저장"""
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 집중 권장 시리즈
    focus_series = df[df['focus_recommended'] == True].sort_values(
        'predictability_score', ascending=False
    )
    focus_file = output_path / 'focus_series.csv'
    focus_series[['series_id', 'predictability_score', 'model_type', 'mean', 'has_seasonality']].to_csv(
        focus_file, index=False
    )
    print(f"✓ 집중 권장 시리즈: {focus_file} ({len(focus_series)}개)")
    
    # 문제 시리즈
    problem_series = df[df['predictability_category'] == 'low'].sort_values(
        'predictability_score'
    )
    problem_file = output_path / 'problem_series.csv'
    problem_series[['series_id', 'predictability_score', 'is_sparse', 'has_drift', 'has_seasonality']].to_csv(
        problem_file, index=False
    )
    print(f"✓ 문제 시리즈: {problem_file} ({len(problem_series)}개)")
    
    # 전체 분류 결과
    all_file = output_path / 'all_series_classified.csv'
    df.to_csv(all_file, index=False)
    print(f"✓ 전체 분류 결과: {all_file} ({len(df)}개)")


def main():
    parser = argparse.ArgumentParser(description='예측 가능성 분석')
    parser.add_argument('--models-dir', type=str, default='artifacts/models', help='모델 디렉토리')
    parser.add_argument('--output', type=str, default='reports', help='보고서 출력 디렉토리')
    parser.add_argument('--high-threshold', type=float, default=0.7, help='높은 예측 가능성 임계값')
    parser.add_argument('--low-threshold', type=float, default=0.4, help='낮은 예측 가능성 임계값')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_path = Path(args.output)
    
    print(f"\n{'='*70}")
    print("예측 가능성 분석")
    print(f"{'='*70}\n")
    
    # 1. 모델 로드
    print("📂 모델 로드 중...")
    df = load_all_models(models_dir)
    print(f"   ✓ {len(df)}개 시리즈 로드 완료\n")
    
    # 2. 분류
    print("🔍 시리즈 분류 중...")
    df = classify_series(df, args.high_threshold, args.low_threshold)
    print(f"   ✓ 분류 완료\n")
    
    # 3. 통계 출력
    print("📊 분류 결과:")
    print(f"   🟢 높음 (≥{args.high_threshold}): {(df['predictability_category']=='high').sum()}개")
    print(f"   🟡 중간 ({args.low_threshold}~{args.high_threshold}): {(df['predictability_category']=='medium').sum()}개")
    print(f"   🔴 낮음 (<{args.low_threshold}): {(df['predictability_category']=='low').sum()}개")
    print(f"   🎯 집중 권장: {df['focus_recommended'].sum()}개\n")
    
    # 4. 보고서 생성
    print("📝 보고서 생성 중...")
    generate_analysis_report(df, output_path, args.high_threshold, args.low_threshold)
    
    # 5. 카테고리별 목록 저장
    print("\n💾 카테고리별 목록 저장 중...")
    save_category_lists(df, output_path)
    
    print(f"\n{'='*70}")
    print("✅ 분석 완료!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
