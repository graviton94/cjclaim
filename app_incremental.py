"""
Streamlit 통합 품질 클레임 관리 시스템
EWS 조기경보 | 월별 데이터 업로드 | Lag 필터링 | 예측 비교 | 재학습
"""
import streamlit as st
import pandas as pd
from src.io_utils import _read_csv_any_encoding
import numpy as np
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="품질 클레임 관리 시스템", layout="wide", page_icon="📊")

st.title("📊 품질 클레임 예측 관리 시스템")
st.markdown("**EWS 조기경보 | 예측 대시보드 | 월별 데이터 업로드 | 증분학습 | Reconcile 보정**")

# 사이드바 - 설정
st.sidebar.header("⚙️ 시스템 정보")

# Lag 통계 파일 확인
lag_stats_path = Path("artifacts/metrics/lag_stats_from_raw.csv")
if lag_stats_path.exists():
    st.sidebar.success(f"✅ Lag 통계 파일 존재")
    lag_stats = pd.read_csv(lag_stats_path)
    st.sidebar.caption(f"제품범주2: {len(lag_stats):,}개")
else:
    st.sidebar.error("❌ Lag 통계 파일 없음")
    st.error("lag_stats_from_raw.csv 파일이 필요합니다. tools/lag_analyzer.py를 먼저 실행하세요.")
    st.stop()

# 메인 영역
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚠️ EWS 조기경보", 
    "📤 데이터 업로드", 
    "📈 처리 결과", 
    "🔧 Reconcile 보정", 
    "📊 통계"
])

# Tab 1: EWS 조기경보
with tab1:
    st.header("⚠️ EWS 조기경보 시스템")
    st.markdown("**6개월 예측 기반 고위험 클레임 시리즈 식별**")
    
    # EWS 데이터 로드
    import re
    def get_latest_file(folder, pattern, ext):
        files = list(Path(folder).glob(pattern))
        # Extract YYYY_MM from filename
        def extract_ym(f):
            m = re.search(r'(\d{4})[_.-]?(\d{2})', f.name)
            return int(m.group(1)) * 100 + int(m.group(2)) if m else 0
        files = [f for f in files if f.suffix == ext]
        files.sort(key=extract_ym, reverse=True)
        return files[0] if files else None

    @st.cache_data
    def load_ews_scores():
        latest_ews = get_latest_file("artifacts/metrics", "ews_scores_*.csv", ".csv")
        if not latest_ews or not latest_ews.exists():
            return None
        return pd.read_csv(latest_ews)

    @st.cache_data
    def load_forecast_data():
        latest_forecast = get_latest_file("artifacts/forecasts/2024", "forecast_2024_*.parquet", ".parquet")
        if not latest_forecast or not latest_forecast.exists():
            return None
        return pd.read_parquet(latest_forecast)

    df_ews = load_ews_scores()
    df_forecast_ews = load_forecast_data()
    
    if df_ews is None:
        st.error("❌ EWS 스코어 파일이 없습니다.")
        st.info("먼저 EWS 스코어링을 실행하세요:")
        st.code("python -m src.ews_scoring_v2 --forecast artifacts/forecasts/2024/forecast_2024_01.parquet --output artifacts/metrics/ews_scores_2024_01.csv", language="bash")
    else:
        # 예측 위험 월 계산
        @st.cache_data
        def calculate_risk_months(df_ews, df_forecast):
            """각 시리즈의 최대 위험 월 계산"""
            if df_forecast is None:
                return {}
            
            risk_months = {}
            for _, row in df_ews.iterrows():
                series_id = row['series_id']
                series_forecast = df_forecast[df_forecast['series_id'] == series_id]
                
                if len(series_forecast) > 0:
                    # 예측값이 가장 높은 월 찾기
                    max_idx = series_forecast['y_pred'].idxmax()
                    max_row = series_forecast.loc[max_idx]
                    risk_month = f"{int(max_row['year'])}-{int(max_row['month']):02d}"
                    max_value = max_row['y_pred']
                    risk_months[series_id] = {'month': risk_month, 'value': max_value}
                else:
                    risk_months[series_id] = {'month': 'N/A', 'value': 0}
            
            return risk_months
        
        risk_months_dict = calculate_risk_months(df_ews, df_forecast_ews)
        
        # 위험 월 정보를 DataFrame에 추가
        df_ews['위험_월'] = df_ews['series_id'].map(lambda x: risk_months_dict.get(x, {}).get('month', 'N/A'))
        df_ews['위험_월_예측값'] = df_ews['series_id'].map(lambda x: risk_months_dict.get(x, {}).get('value', 0))
        
        # LOW_CONF 설명 추가
        st.info("💡 **레벨 설명**: HIGH(신뢰도 높음), MEDIUM(중간), LOW(낮은 위험), **LOW_CONF(증가율 높지만 신뢰도 낮음 - 데이터 부족)**")
        # 필터 컨트롤
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
        
        with col_filter1:
            level_filter = st.multiselect(
                "EWS 레벨",
                ["전체"] + sorted(df_ews['level'].unique().tolist()),
                default=["전체"]
            )
        
        with col_filter2:
            conf_min = st.slider("최소 신뢰도 (F2)", 0.0, 1.0, 0.0, 0.1)
        
        with col_filter3:
            ratio_min = st.slider("최소 증가율 (F1)", 0.0, 5.0, 0.0, 0.1)
        
        with col_filter4:
            score_min = st.slider("최소 EWS 스코어", 0.0, 1.0, 0.0, 0.1)
        
        # 필터 적용
        df_ews_filtered = df_ews.copy()
        
        if "전체" not in level_filter:
            df_ews_filtered = df_ews_filtered[df_ews_filtered['level'].isin(level_filter)]
        
        df_ews_filtered = df_ews_filtered[
            (df_ews_filtered['f2_conf'] >= conf_min) &
            (df_ews_filtered['f1_ratio'] >= ratio_min) &
            (df_ews_filtered['ews_score'] >= score_min)
        ]
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 시리즈", f"{len(df_ews):,}개")
        
        with col2:
            high_count = (df_ews['level'] == 'HIGH').sum()
            st.metric("HIGH 위험", f"{high_count:,}개", 
                      delta=f"{high_count/len(df_ews)*100:.1f}%")
        
        with col3:
            valid_count = df_ews['candidate'].sum()
            st.metric("유효 후보", f"{valid_count:,}개",
                      delta=f"{valid_count/len(df_ews)*100:.1f}%")
        
        with col4:
            st.metric("필터 결과", f"{len(df_ews_filtered):,}개")
        
        st.markdown("---")
        
        # Top 위험 시리즈 (단일 화면)
        st.markdown("### 🏆 고위험 시리즈 (변별력 개선)")
        
        # 변별력 개선 옵션
        col_option1, col_option2 = st.columns(2)
        
        with col_option1:
            use_weighted_score = st.checkbox(
                "가중 스코어 사용 (신뢰도×증가율 반영)",
                value=True,
                help="EWS 스코어에 신뢰도와 증가율을 곱하여 변별력 향상"
            )
        
        with col_option2:
            exclude_low_conf = st.checkbox(
                "LOW_CONF 제외",
                value=True,
                help="신뢰도가 낮은 시리즈 제외"
            )
        
        # 필터 적용
        df_display = df_ews_filtered.copy()
        
        if exclude_low_conf:
            df_display = df_display[df_display['level'] != 'LOW_CONF']
        
        # 가중 스코어 계산
        if use_weighted_score:
            df_display['변별_스코어'] = (
                df_display['ews_score'] * 0.4 + 
                df_display['f1_ratio'] * 0.3 + 
                df_display['f2_conf'] * 0.3
            )
            # 변별 스코어로 재정렬하고 순위 재계산
            df_display = df_display.sort_values('변별_스코어', ascending=False).reset_index(drop=True)
            df_display['rank'] = range(1, len(df_display) + 1)
            sort_default = '변별_스코어'
        else:
            sort_default = 'ews_score'
        
        top_n = st.slider("표시할 시리즈 수", 10, 100, 20, 10, key="ews_top_n")
        
        sort_by = st.selectbox(
            "정렬 기준",
            ["변별 스코어", "EWS 스코어", "증가율", "신뢰도", "계절성", "진폭", "변곡"],
            key="ews_sort_by"
        )
        
        sort_col_map = {
            "변별 스코어": "변별_스코어" if use_weighted_score else "ews_score",
            "EWS 스코어": "ews_score",
            "증가율": "f1_ratio",
            "신뢰도": "f2_conf",
            "계절성": "f3_season",
            "진폭": "f4_ampl",
            "변곡": "f5_inflect"
        }
        
        df_ews_top = df_display.nlargest(top_n, sort_col_map[sort_by])
        
        # 컬럼명 한글화
        display_columns = {
            'rank': '순위',
            'series_id': '시리즈',
            'ews_score': 'EWS점수',
            '변별_스코어': '변별점수',
            'level': '레벨',
            'f1_ratio': '증가율',
            'f2_conf': '신뢰도',
            'f3_season': '계절성',
            'f4_ampl': '진폭',
            'f5_inflect': '변곡',
            '위험_월': '위험월',
            '위험_월_예측값': '예측값'
        }
        
        # 표시할 컬럼 선택
        if use_weighted_score:
            show_cols = ['rank', 'series_id', 'ews_score', '변별_스코어', 'level', 
                        'f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect',
                        '위험_월', '위험_월_예측값']
        else:
            show_cols = ['rank', 'series_id', 'ews_score', 'level', 
                        'f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect',
                        '위험_월', '위험_월_예측값']
        
        df_display_table = df_ews_top[show_cols].copy()
        df_display_table = df_display_table.rename(columns=display_columns)
        
        # 스타일 적용
        styled_df = df_display_table.style.background_gradient(
            subset=['EWS점수'] if not use_weighted_score else ['변별점수'], 
            cmap='YlOrRd'
        ).format({
            'EWS점수': '{:.3f}',
            '변별점수': '{:.3f}' if use_weighted_score else None,
            '증가율': '{:.2f}x',
            '신뢰도': '{:.2f}',
            '계절성': '{:.2f}',
            '진폭': '{:.2f}',
            '변곡': '{:.2f}',
            '예측값': '{:.1f}건'
        })
        
        st.dataframe(styled_df, width='stretch', height=400)

        # 시리즈 선택 및 상세 정보 + 예측 추이 그래프
        if len(df_ews_top) > 0:
            st.markdown("---")
            st.subheader("📋 시리즈 상세 정보 및 예측 추이")
            
            selected_series_ews = st.selectbox(
                "시리즈 선택 (아래 상세 정보 및 그래프 확인)",
                df_ews_top['series_id'].tolist(),
                key="ews_detail_select"
            )
            
            if selected_series_ews:
                series_info = df_ews_top[df_ews_top['series_id'] == selected_series_ews].iloc[0]
                
                # 상세 정보 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**기본 정보**")
                    st.write(f"- **시리즈**: {series_info['series_id']}")
                    st.write(f"- **EWS 레벨**: {series_info['level']}")
                    st.write(f"- **EWS 스코어**: {series_info['ews_score']:.3f}")
                    if use_weighted_score:
                        st.write(f"- **변별 스코어**: {series_info['변별_스코어']:.3f}")
                    st.write(f"- **순위**: {int(series_info['rank'])}")
                
                with col2:
                    st.markdown("**5-Factor 점수**")
                    st.write(f"- **증가율 (F1)**: {series_info['f1_ratio']:.2f}x")
                    st.write(f"- **신뢰도 (F2)**: {series_info['f2_conf']:.2f}")
                    st.write(f"- **계절성 (F3)**: {series_info['f3_season']:.2f}")
                    st.write(f"- **진폭 (F4)**: {series_info['f4_ampl']:.2f}")
                    st.write(f"- **변곡 (F5)**: {series_info['f5_inflect']:.2f}")
                
                with col3:
                    st.markdown("**예측 위험 정보**")
                    st.write(f"- **위험 월**: {series_info['위험_월']}")
                    st.write(f"- **예상 클레임**: {series_info['위험_월_예측값']:.1f}건")
                    
                    # 위험도 표시
                    if series_info['level'] == 'HIGH':
                        st.error("🔴 **높은 위험**")
                    elif series_info['level'] == 'MEDIUM':
                        st.warning("🟡 **중간 위험**")
                    elif series_info['level'] == 'LOW_CONF':
                        st.info("🔵 **데이터 부족 (신뢰도 낮음)**")
                    else:
                        st.success("🟢 **낮은 위험**")
                
                # 월별 예측 추이 그래프 (바로 아래 표시)
                st.markdown("---")
                st.markdown("### 📈 월별 예측 추이")
                
                if df_forecast_ews is not None:
                    # 예측 데이터
                    series_forecast = df_forecast_ews[df_forecast_ews['series_id'] == selected_series_ews].copy()
                    
                    if len(series_forecast) > 0:
                        # month_label 생성 (year-month 형식)
                        series_forecast['month_label'] = series_forecast['year'].astype(str) + '-' + series_forecast['month'].astype(str).str.zfill(2)
                        series_forecast = series_forecast.sort_values(['year', 'month'])
                        
                        # 과거 데이터 로드
                        @st.cache_data
                        def load_historical_for_series(series_id, forecast_start_year, forecast_start_month):
                            """시리즈의 과거 12개월 데이터 로드 (예측 시작 전 12개월)"""
                            try:
                                # JSON 파일에서 과거 데이터 로드
                                json_dir = Path("data/features")
                                safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                                                .replace('|', '_').replace('?', '_').replace('*', '_')
                                                .replace('<', '_').replace('>', '_').replace('"', '_'))
                                json_path = json_dir / f"{safe_filename}.json"
                                
                                if json_path.exists():
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    
                                    df_hist = pd.DataFrame(data.get('data', []))
                                    if len(df_hist) > 0:
                                        # 예측 시작 전 12개월 계산
                                        from datetime import datetime
                                        from dateutil.relativedelta import relativedelta
                                        
                                        forecast_start = datetime(forecast_start_year, forecast_start_month, 1)
                                        hist_end = forecast_start - relativedelta(months=1)  # 예측 직전 월
                                        hist_start = hist_end - relativedelta(months=11)  # 12개월 전
                                        
                                        # 해당 기간의 데이터만 필터링
                                        df_hist = df_hist[
                                            ((df_hist['year'] > hist_start.year) | 
                                             ((df_hist['year'] == hist_start.year) & (df_hist['month'] >= hist_start.month))) &
                                            ((df_hist['year'] < hist_end.year) | 
                                             ((df_hist['year'] == hist_end.year) & (df_hist['month'] <= hist_end.month)))
                                        ].copy()
                                        
                                        if len(df_hist) > 0:
                                            df_hist['month_label'] = df_hist['year'].astype(str) + '-' + df_hist['month'].astype(str).str.zfill(2)
                                            return df_hist[['year', 'month', 'month_label', 'claim_count']].sort_values(['year', 'month'])
                            
                            except Exception as e:
                                st.warning(f"과거 데이터 로드 실패: {e}")
                            
                            return pd.DataFrame()
                        
                        # 예측 시작 시점 추출
                        forecast_start_year = int(series_forecast['year'].min())
                        forecast_start_month = int(series_forecast['month'].min())
                        
                        df_historical = load_historical_for_series(selected_series_ews, forecast_start_year, forecast_start_month)
                        
                        # 과거 데이터 로드 상태 표시
                        if len(df_historical) > 0:
                            hist_start = df_historical['month_label'].iloc[0]
                            hist_end = df_historical['month_label'].iloc[-1]
                            st.info(f"📊 과거 데이터: {hist_start} ~ {hist_end} ({len(df_historical)}개월)")
                        else:
                            st.warning("⚠️ 과거 12개월 데이터를 찾을 수 없습니다. 예측 데이터만 표시됩니다.")
                        
                        # 음수 클레임 제거
                        if len(df_historical) > 0:
                            df_historical = df_historical[df_historical['claim_count'] >= 0].copy()
                        
                        series_forecast = series_forecast[
                            (series_forecast['y_pred'] >= 0) &
                            (series_forecast['y_pred_lower'] >= 0) &
                            (series_forecast['y_pred_upper'] >= 0)
                        ].copy()
                        
                        # 데이터 유효성 체크
                        if len(series_forecast) == 0:
                            st.warning("⚠️ 모든 예측값이 음수로 필터링되었습니다. 이 시리즈는 데이터 오류가 있을 수 있습니다.")
                        else:
                            # 그래프
                            fig = go.Figure()
                            
                            # 과거 실제 데이터 (예측 전 12개월)
                            if len(df_historical) > 0:
                                fig.add_trace(go.Scatter(
                                    x=df_historical['month_label'],
                                    y=df_historical['claim_count'],
                                    mode='lines+markers',
                                    name=f'실제값 (과거 12개월)',
                                    line=dict(color='gray', width=2),
                                    marker=dict(size=8, symbol='circle')
                                ))
                            
                            # 예측값 (6개월)
                            fig.add_trace(go.Scatter(
                                x=series_forecast['month_label'],
                                y=series_forecast['y_pred'],
                                mode='lines+markers',
                                name=f'예측값 (6개월)',
                                line=dict(color='blue', width=2, dash='dash'),
                                marker=dict(size=8, symbol='diamond')
                            ))
                            
                            # 신뢰구간
                            fig.add_trace(go.Scatter(
                                x=series_forecast['month_label'],
                                y=series_forecast['y_pred_upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=series_forecast['month_label'],
                                y=series_forecast['y_pred_lower'],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(0,100,255,0.2)',
                                fill='tonexty',
                                name='95% 신뢰구간'
                            ))
                            
                            # 예측 시작 구분선
                            if len(df_historical) > 0 and len(series_forecast) > 0:
                                forecast_start = series_forecast['month_label'].iloc[0]
                                
                                fig.add_shape(
                                    type="line",
                                    x0=forecast_start,
                                    x1=forecast_start,
                                    y0=0,
                                    y1=1,
                                    yref="paper",
                                    line=dict(color="red", width=2, dash="dot")
                                )
                                
                                fig.add_annotation(
                                    x=forecast_start,
                                    y=1,
                                    yref="paper",
                                    text="예측 시작",
                                    showarrow=False,
                                    font=dict(size=10, color="red"),
                                    yshift=10
                                )
                            
                            fig.update_layout(
                                title=f'시계열 추이: {selected_series_ews}<br><sub>과거 12개월 + 예측 6개월</sub>',
                                xaxis_title='월',
                                yaxis_title='클레임 수',
                                height=450,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # 통계
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if len(df_historical) > 0:
                                    st.metric("과거 12개월 평균", f"{df_historical['claim_count'].mean():.2f}건")
                                else:
                                    st.metric("과거 평균", "N/A")
                            
                            with col2:
                                st.metric("예측 6개월 평균", f"{series_forecast['y_pred'].mean():.2f}건")
                            
                            with col3:
                                max_pred = series_forecast['y_pred'].max()
                                max_month = series_forecast.loc[series_forecast['y_pred'].idxmax(), 'month_label']
                                st.metric("최대 예측", f"{max_pred:.2f}건", delta=max_month)
                            
                            with col4:
                                st.metric("6개월 합계", f"{series_forecast['y_pred'].sum():.0f}건")
                            
                            # 상세 테이블
                            st.markdown("**� 상세 예측 데이터**")
                            detail_cols = st.columns([1, 1])
                            
                            with detail_cols[0]:
                                if len(df_historical) > 0:
                                    st.markdown("**과거 12개월 실제**")
                                    st.dataframe(
                                        df_historical[['month_label', 'claim_count']].rename(columns={
                                            'month_label': '월',
                                            'claim_count': '실제값'
                                        }).style.format({'실제값': lambda x: f"{int(x)}건" if x is not None else "N/A"}),
                                        width='stretch',
                                        height=300
                                    )
                            
                            with detail_cols[1]:
                                st.markdown("**예측 6개월**")
                                st.dataframe(
                                    series_forecast[['month_label', 'y_pred', 'y_pred_lower', 'y_pred_upper']].rename(columns={
                                        'month_label': '월',
                                        'y_pred': '예측값',
                                        'y_pred_lower': '하한',
                                        'y_pred_upper': '상한'
                                    }).style.format({
                                        '예측값': lambda x: f"{x:.2f}건" if x is not None else "N/A",
                                        '하한': lambda x: f"{x:.2f}건" if x is not None else "N/A",
                                        '상한': lambda x: f"{x:.2f}건" if x is not None else "N/A"
                                    }),
                                    width='stretch',
                                    height=300
                                )
                else:
                    st.warning("선택한 시리즈의 예측 데이터가 없습니다.")
            else:
                st.error("예측 파일이 없습니다.")

# Tab 2: 데이터 업로드
with tab2:
    st.header("1️⃣ 월별 데이터 업로드")
    
    # 학습 데이터 현황 테이블
    st.subheader("📊 학습 데이터 현황")
    
    features_dir = Path("data/features")
    if features_dir.exists() and any(features_dir.glob("*.json")):
        with st.spinner("학습 데이터 분석 중..."):
            # 년/월별 데이터 수집
            year_month_data = {}
            total_series = 0
            for json_file in features_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    records = data.get('data', [])
                    if not records:
                        continue
                    total_series += 1
                    for record in records:
                        year = record.get('year')
                        month = record.get('month')
                        if year and month:
                            key = (year, month)
                            if key not in year_month_data:
                                year_month_data[key] = 0
                            year_month_data[key] += 1
                except Exception as e:
                    continue
            if year_month_data:
                years = sorted(set(year for year, month in year_month_data.keys()))
                months = list(range(1, 13))
                table_data = []
                for month in months:
                    row = {'월': f"{month}월"}
                    for year in years:
                        count = year_month_data.get((year, month), 0)
                        threshold = total_series * 0.8
                        if count >= threshold:
                            status = "✅"
                        elif count >= threshold * 0.5:
                            status = "⚠️"
                        else:
                            status = "❌"
                        row[f"{year}년"] = f"{status} {count}"
                    table_data.append(row)
                df_status = pd.DataFrame(table_data)
                st.dataframe(
                    df_status.style.format({
                        col: (lambda x: f"{int(x)}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
                        for col in df_status.columns if col != '월'
                    }),
                    width='stretch',
                    hide_index=True
                )
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.caption(f"**총 시리즈**: {total_series:,}개")
                with col2:
                    st.caption("**✅ 충분**: ≥80% 시리즈")
                with col3:
                    st.caption("**⚠️ 보통**: 40-80% 시리즈")
                with col4:
                    st.caption("**❌ 부족**: <40% 시리즈")
            else:
                st.info("학습 데이터가 없습니다.")
    else:
        st.warning("Feature JSON 파일이 없습니다. 먼저 Base 학습을 실행하세요.")
    
    st.markdown("---")
    
    # 업로드할 데이터의 년/월은 CSV에서 자동 감지
    st.subheader("2️⃣ 월별 데이터 업로드")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            f"**CSV 파일 업로드**",
            type=['csv'],
            help="발생일자 기준 1개월 데이터 (플랜트, 제품범주2, 중분류, 발생일자, 제조일자, count 컬럼 필수)"
        )
    
    with col2:
        st.markdown("**필수 컬럼**")
        st.code("""
플랜트
제품범주2
중분류
발생일자
제조일자
count
        """, language="text")
    
    if uploaded_file is not None:
        # 임시 저장
        temp_dir = Path("artifacts/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 미리보기 및 년/월 자동 감지
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        try:
            df_preview = pd.read_csv(uploaded_file, encoding='utf-8-sig', nrows=10)
            df_preview.columns = df_preview.columns.str.strip()
            
            # 전체 데이터 로드
            uploaded_file.seek(0)  # 파일 포인터 초기화
            df_full = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            df_full.columns = df_full.columns.str.strip()
            
            # 발생일자에서 년/월 자동 감지
            if '발생일자' in df_full.columns:
                df_full['발생일자'] = pd.to_datetime(df_full['발생일자'])
                detected_year = int(df_full['발생일자'].dt.year.mode()[0])  # 최빈값
                detected_month = int(df_full['발생일자'].dt.month.mode()[0])  # 최빈값
                month_key = f"{detected_year}-{detected_month:02d}"
                
                st.info(f"📅 **감지된 대상 월:** {month_key}")
            else:
                st.error("'발생일자' 컬럼이 없습니다.")
                st.stop()
            
            # 임시 파일 저장
            temp_path = temp_dir / f"upload_{month_key.replace('-', '')}.csv"
            df_full.to_csv(temp_path, index=False, encoding='utf-8-sig')
            st.subheader("📋 데이터 미리보기")
            st.dataframe(df_preview, width='stretch')
            
            # 전체 데이터 통계
            df_full = pd.read_csv(temp_path, encoding='utf-8-sig')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                run_pipeline = st.button("🚀 파이프라인 자동 실행", help="업로드 후 라그 필터링, feature/parquet 생성, 예측, EWS, 증분 재학습까지 자동 실행")
            if run_pipeline:
                st.info(f"[자동화] {month_key} 파이프라인 실행 중...")
                # 1. 라그 필터링/주간 집계/feature JSON 생성
                process_cmd = [sys.executable, "process_monthly_incremental.py", "--new-csv", str(temp_path), "--year", str(detected_year), "--month", str(detected_month), "--output-list", f"artifacts/temp/updated_series_{detected_year}{detected_month:02d}.txt"]
                result1 = subprocess.run(process_cmd, capture_output=True, text=True)
                st.code(result1.stdout)
                if result1.returncode != 0:
                    st.error(f"[오류] 데이터 처리 실패: {result1.stderr}")
                    st.stop()
                # 2. feature parquet 생성
                feature_cmd = [sys.executable, "tools/generate_cycle_features_parquet.py"]
                result2 = subprocess.run(feature_cmd, capture_output=True, text=True)
                st.code(result2.stdout)
                if result2.returncode != 0:
                    st.error(f"[오류] feature/parquet 생성 실패: {result2.stderr}")
                    st.stop()
                # 3. 예측
                forecast_cmd = [sys.executable, "batch.py", "forecast", "--month-new", month_key]
                result3 = subprocess.run(forecast_cmd, capture_output=True, text=True)
                st.code(result3.stdout)
                if result3.returncode != 0:
                    st.error(f"[오류] 예측 실패: {result3.stderr}")
                    st.stop()
                # 4. EWS 스코어링
                forecast_parquet = f"artifacts/forecasts/{detected_year}/forecast_{detected_year}_{detected_month:02d}.parquet"
                ews_output = f"artifacts/metrics/ews_scores_{detected_year}_{detected_month:02d}.csv"
                ews_cmd = [sys.executable, "-m", "src.ews_scoring_v2", "--forecast", forecast_parquet, "--output", ews_output]
                result4 = subprocess.run(ews_cmd, capture_output=True, text=True)
                st.code(result4.stdout)
                if result4.returncode != 0:
                    st.error(f"[오류] EWS 스코어링 실패: {result4.stderr}")
                    st.stop()
                # 5. 증분 재학습
                retrain_cmd = [sys.executable, "batch.py", "retrain", "--month", month_key]
                result5 = subprocess.run(retrain_cmd, capture_output=True, text=True)
                st.code(result5.stdout)
                if result5.returncode != 0:
                    st.error(f"[오류] 증분 재학습 실패: {result5.stderr}")
                    st.stop()
                st.success(f"✅ {month_key} 파이프라인 자동 실행 완료!")
                st.metric("총 레코드", f"{len(df_full):,}건")
            with col2:
                series_count = int(df_full.groupby(['플랜트', '제품범주2', '중분류']).ngroups)
                st.metric("시리즈 수", f"{series_count:,}개")
            with col3:
                if '발생일자' in df_full.columns:
                    df_full['발생일자'] = pd.to_datetime(df_full['발생일자'])
                    date_range = f"{df_full['발생일자'].min().date()} ~ {df_full['발생일자'].max().date()}"
                    st.metric("발생일자 범위", date_range)
            with col4:
                if 'count' in df_full.columns:
                    total_claims = int(df_full['count'].sum())
                    st.metric("총 클레임 건수", f"{total_claims:,}건")
            
            st.markdown("---")
            
            # 처리 버튼
            st.subheader("2️⃣ 파이프라인 실행")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                run_pipeline = st.button("🚀 파이프라인 실행", type="primary", width='stretch', key="run_pipeline")
            
            with col2:
                show_command = st.checkbox("명령어 표시", value=False)
            
            if show_command:
                with col3:
                    st.code(f"python batch.py process --upload {temp_path} --month {month_key}", language="bash")
            
            if run_pipeline:
                st.markdown("---")
                
                # 처리 상태 표시 (동적 업데이트)
                status_placeholder = st.empty()
                status_placeholder.subheader("⏳ 처리 중...")
                
                # 진행 상황 표시
                status_text = st.empty()
                
                # 로그 출력 컨테이너
                log_container = st.expander("🔍 실시간 로그", expanded=True)
                log_output = log_container.empty()
                
                try:
                    # batch.py process 실행
                    status_text.info("🔄 파이프라인 시작...")
                    
                    cmd = [
                        sys.executable, "batch.py", "process",
                        "--upload", str(temp_path),
                        "--month", month_key
                    ]
                    
                    log_container.code(f"실행: {' '.join(cmd)}", language="bash")
                    
                    # 실시간 출력을 위한 프로세스 실행
                    import io
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=Path.cwd(),
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # 실시간 로그 수집
                    output_lines = []
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_lines.append(line.rstrip())
                            # 마지막 50줄만 표시
                            display_lines = output_lines[-50:]
                            log_output.code('\n'.join(display_lines), language="text")
                            
                            # 상태 업데이트
                            if "Step 1" in line or "Lag 필터링" in line:
                                status_text.info("🔄 Step 1/4: Lag 필터링 중...")
                            elif "Step 2" in line or "주간 집계" in line:
                                status_text.info("🔄 Step 2/4: 주간 집계 및 JSON 업데이트 중...")
                            elif "Step 3" in line or "예측 비교" in line:
                                status_text.info("🔄 Step 3/4: 예측 vs 실제 비교 중...")
                            elif "Step 4" in line or "재학습" in line:
                                status_text.info("🔄 Step 4/4: 모델 재학습 중...")
                    
                    process.wait()
                    result_code = process.returncode
                    
                    if result_code == 0:
                        status_placeholder.success("✅ 처리 완료!")
                        status_text.success("✅ 파이프라인 완료!")
                        st.success("🎉 월별 파이프라인 처리 완료!")
                        
                        # 결과 표시 (자동 감지된 년/월 사용)
                        month_dir = Path(f"artifacts/incremental/{detected_year}{detected_month:02d}")
                        summary_file = month_dir / f"summary_{detected_year}{detected_month:02d}.json"
                        
                        if summary_file.exists():
                            with open(summary_file, 'r', encoding='utf-8') as f:
                                summary = json.load(f)
                            
                            st.subheader("📊 처리 결과 요약")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("총 레코드", f"{summary.get('total_records', 0):,}건")
                            with col2:
                                st.metric("시리즈 수", f"{summary.get('series_count', 0):,}개")
                            with col3:
                                if summary.get('mean_error') is not None:
                                    st.metric("평균 오차", f"{summary['mean_error']:.2f}")
                            with col4:
                                if summary.get('mae') is not None:
                                    st.metric("MAE", f"{summary['mae']:.2f}")
                            
                            st.info(f"처리 시간: {summary.get('processed_at', 'N/A')}")
                    else:
                        status_placeholder.error("❌ 처리 실패")
                        status_text.error("❌ 파이프라인 실패")
                        st.error(f"파이프라인 실행 실패 (exit code: {result_code})")
                        st.info("💡 위의 로그를 확인하여 오류 원인을 파악하세요.")
                
                except Exception as e:
                    status_placeholder.error("❌ 오류 발생")
                    status_text.error("❌ 오류 발생")
                    st.error(f"오류 발생: {e}")
                    import traceback
                    with log_container:
                        st.code(traceback.format_exc(), language="text")
        
        except Exception as e:
            st.error(f"파일 읽기 실패: {e}")

# Tab 3: 처리 결과
with tab3:
    st.header("📈 처리 결과")
    
    # 서브탭 생성
    tabs_result = st.tabs(["📋 처리 결과 조회", "🎯 예측 정확도 평가", "🔄 증분 재학습"])
    result_tab1, result_tab2, result_tab3 = tabs_result
    
    # 처리 결과 조회
    with result_tab1:
        # 년/월 선택
        col_date1, col_date2, col_date3 = st.columns([1, 1, 2])
        with col_date1:
            current_year = datetime.now().year
            view_year = st.selectbox("연도", range(2024, current_year + 2), key="view_year")
        with col_date2:
            view_month = st.selectbox("월", range(1, 13), key="view_month")
        with col_date3:
            st.info(f"**조회 대상 월:** {view_year}-{view_month:02d}")
        
        month_key = f"{view_year}-{view_month:02d}"
        
        # 월별 결과 디렉토리
        month_dir = Path(f"artifacts/incremental/{view_year}{view_month:02d}")
        if month_dir.exists():
            st.success(f"✅ {month_key} 처리 결과 존재")
            files = list(month_dir.glob("*"))
            # 주요 파일 자동 분류
            summary_file = month_dir / f"summary_{view_year}{view_month:02d}.json"
            eval_file = month_dir / f"{view_year}{view_month:02d}_predict_vs_actual.csv"
            retrain_file = month_dir / f"{view_year}{view_month:02d}_incremental_training_results.csv"
            # 요약 정보 표시
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                st.subheader("📊 처리 결과 요약")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 레코드", f"{summary.get('total_records', 0):,}건")
                with col2:
                    st.metric("시리즈 수", f"{summary.get('series_count', 0):,}개")
                with col3:
                    if summary.get('mean_error') is not None:
                        st.metric("평균 오차", f"{summary['mean_error']:.2f}")
                with col4:
                    if summary.get('mae') is not None:
                        st.metric("MAE", f"{summary['mae']:.2f}")
                st.info(f"처리 시간: {summary.get('processed_at', 'N/A')}")
            # 평가 결과 미리보기
            if eval_file.exists():
                st.subheader("📈 예측 vs 실제 평가 결과")
                if st.button("평가 결과 미리보기", key="preview_eval"):
                    df_eval = _read_csv_any_encoding(eval_file)
                    st.dataframe(df_eval.head(100), width='stretch')
            # 재학습 결과 미리보기
            if retrain_file.exists():
                st.subheader("� 증분 재학습 결과")
                if st.button("재학습 결과 미리보기", key="preview_retrain"):
                    df_retrain = _read_csv_any_encoding(retrain_file)
                    st.dataframe(df_retrain.head(100), width='stretch')
            # 기타 파일 목록
            st.subheader("📁 기타 생성된 파일")
            for file in sorted(files):
                # 주요 파일은 위에서 이미 표시
                if file in [summary_file, eval_file, retrain_file]:
                    continue
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.caption(f"{file.stat().st_size / 1024:.1f} KB")
                with col3:
                    if file.suffix == '.csv':
                        if st.button(f"보기", key=f"view_{file.name}"):
                            df = _read_csv_any_encoding(file)
                            st.dataframe(df.head(100), width='stretch')
                    elif file.suffix == '.json':
                        if st.button(f"보기", key=f"view_{file.name}"):
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            st.json(data)
                    elif file.suffix == '.parquet':
                        if st.button(f"보기", key=f"view_{file.name}"):
                            df = pd.read_parquet(file)
                            st.dataframe(df.head(100), width='stretch')
                        elif file.suffix == '.json':
                            if st.button(f"보기", key=f"view_{file.name}"):
                                with open(file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                st.json(data)
                
                # 예측-실측 비교 파일 있으면 시각화
                predict_vs_actual = month_dir / f"predict_vs_actual_{view_year}{view_month:02d}.csv"
                if predict_vs_actual.exists():
                    st.markdown("---")
                    st.subheader("📊 예측 vs 실측 비교")
                    
                    df_compare = pd.read_csv(predict_vs_actual, encoding='utf-8-sig')
                    
                    # 통계
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("평균 오차", f"{df_compare['error'].mean():.2f}")
                    with col2:
                        st.metric("절대 오차 평균", f"{df_compare['abs_error'].mean():.2f}")
                    with col3:
                        st.metric("퍼센트 오차 평균", f"{df_compare['pct_error'].mean():.2f}%")
                    with col4:
                        st.metric("비교 레코드", f"{len(df_compare):,}건")
                    
                    # 상위 오차 시리즈
                    st.markdown("**상위 오차 시리즈 (Top 10)**")
                    display_cols = ['series_id']
                    if 'month' in df_compare.columns:
                        display_cols.append('month')
                    display_cols.extend(['claim_count', 'y_pred', 'error', 'abs_error'])
                    top_errors = df_compare.nlargest(10, 'abs_error')[display_cols]
                    st.dataframe(top_errors, width='stretch')
            else:
                st.info("파일이 없습니다.")
        else:
            st.info(f"{month_key} 처리 결과가 없습니다. 먼저 데이터를 업로드하고 파이프라인을 실행하세요.")
    
    # 서브탭 2: 예측 정확도 평가
    with result_tab2:
        st.markdown("### 🎯 예측 vs 실제 성능 평가")
        st.caption("업로드된 실제 데이터와 예측값을 비교하여 모델 성능을 평가합니다.")

        # 평가 대상 월 선택
        col1, col2 = st.columns(2)
        with col1:
            eval_year = st.selectbox("평가 연도", range(2024, current_year + 2), key="eval_year_accuracy")
        with col2:
            eval_month = st.selectbox("평가 월", range(1, 13), key="eval_month_accuracy")

        eval_month_key = f"{eval_year}-{eval_month:02d}"

        # 예측 파일은 (평가월-1)로 자동 선택, 단 2024-01월은 최초 베이스 파일 사용
        from dateutil.relativedelta import relativedelta
        if eval_year == 2024 and eval_month == 1:
            forecast_file = Path("artifacts/forecasts/base_monthly/forecast_base_monthly.parquet")
        else:
            eval_date = datetime(eval_year, eval_month, 1)
            forecast_date = eval_date - relativedelta(months=1)
            forecast_year = forecast_date.year
            forecast_month = forecast_date.month
            forecast_file = Path(f"artifacts/forecasts/{forecast_year}/forecast_{forecast_year}_{forecast_month:02d}.parquet")
        actual_file = Path("data/curated/claims_monthly.parquet")

        # 파일 존재 확인
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            if forecast_file.exists():
                st.success(f"✅ 예측 파일 존재: {forecast_file.name}")
            else:
                st.error(f"❌ 예측 파일 없음: {forecast_file.name}")

        with col_check2:
            if actual_file.exists():
                # 실제 데이터에 해당 월이 있는지 확인
                df_actual_check = pd.read_parquet(actual_file)
                has_month = ((df_actual_check['year'] == eval_year) & 
                            (df_actual_check['month'] == eval_month)).any()
                if has_month:
                    st.success(f"✅ 실제 데이터 존재: {eval_month_key}")
                else:
                    st.warning(f"⚠️ {eval_month_key} 실제 데이터 없음")
            else:
                st.error("❌ 실제 데이터 파일 없음")

        # 평가 실행 버튼
        if st.button("🎯 성능 평가 실행", type="primary", width='stretch', key="eval_accuracy_tab2"):
            if not forecast_file.exists():
                st.error("예측 파일이 없습니다. 먼저 예측을 생성하세요.")
            elif not actual_file.exists():
                st.error("실제 데이터 파일이 없습니다.")
            else:
                with st.spinner("평가 중..."):
                    try:
                        # 데이터 로드
                        df_forecast = pd.read_parquet(forecast_file)
                        df_actual = pd.read_parquet(actual_file)
                        # 해당 월 필터링
                        df_forecast_month = df_forecast[
                            (df_forecast['year'] == eval_year) & 
                            (df_forecast['month'] == eval_month)
                        ].copy()
                        df_actual_month = df_actual[
                            (df_actual['year'] == eval_year) & 
                            (df_actual['month'] == eval_month)
                        ].copy()
                        if len(df_actual_month) == 0:
                            st.error(f"{eval_month_key} 실제 데이터가 없습니다.")
                        else:
                            # 병합
                            df_compare = pd.merge(
                                df_actual_month[['series_id', 'year', 'month', 'claim_count']],
                                df_forecast_month[['series_id', 'year', 'month', 'y_pred', 'y_pred_lower', 'y_pred_upper']],
                                on=['series_id', 'year', 'month'],
                                how='inner'
                            )
                            if len(df_compare) == 0:
                                st.warning("예측과 실제 데이터 간 매칭되는 시리즈가 없습니다.")
                            else:
                                # 결과 전체를 컨테이너로 묶음
                                with st.container():
                                    # 오차 계산
                                    df_compare['error'] = df_compare['y_pred'] - df_compare['claim_count']
                                    df_compare['abs_error'] = df_compare['error'].abs()
                                    df_compare['abs_pct_error'] = (df_compare['abs_error'] / (df_compare['claim_count'] + 1)) * 100
                                    # 전체 메트릭 계산
                                    mae = df_compare['abs_error'].mean()
                                    rmse = np.sqrt((df_compare['error'] ** 2).mean())
                                    mape = df_compare['abs_pct_error'].mean()
                                    # WMAPE (Weighted MAPE)
                                    wmape = (df_compare['abs_error'].sum() / df_compare['claim_count'].sum()) * 100
                                    # Bias (평균 오차)
                                    bias = df_compare['error'].mean()
                                    # 결과 표시
                                    st.success(f"✅ 평가 완료: {len(df_compare)}개 시리즈 비교")
                                    # 메트릭 카드
                                    st.markdown("### 📊 전체 성능 메트릭")
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    with col1:
                                        st.metric("MAE", f"{mae:.2f}건", help="Mean Absolute Error (평균 절대 오차)")
                                    with col2:
                                        st.metric("RMSE", f"{rmse:.2f}건", help="Root Mean Square Error")
                                    with col3:
                                        st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
                                    with col4:
                                        st.metric("WMAPE", f"{wmape:.1f}%", help="Weighted MAPE (총량 기준)")
                                    with col5:
                                        bias_delta = "과대예측" if bias > 0 else "과소예측"
                                        st.metric("Bias", f"{bias:+.2f}건", delta=bias_delta, help="평균 오차 (+ = 과대예측)")
                                    # 오차 분포 히스토그램
                                    st.markdown("### 📈 오차 분포")
                                    fig_hist = go.Figure()
                                    fig_hist.add_trace(go.Histogram(
                                        x=df_compare['error'], nbinsx=50, name='오차 분포', marker_color='lightblue'))
                                    fig_hist.update_layout(
                                        title='예측 오차 분포 (예측 - 실제)', xaxis_title='오차 (건)', yaxis_title='빈도', height=400)
                                    st.plotly_chart(fig_hist, width='stretch')
                                    # Top 오차 시리즈
                                    st.markdown("### ⚠️ 오차가 큰 시리즈 (재학습 우선순위)")
                                    df_top_errors = df_compare.nlargest(20, 'abs_error')[
                                        ['series_id', 'claim_count', 'y_pred', 'error', 'abs_error', 'abs_pct_error']].copy()
                                    df_top_errors.columns = ['시리즈', '실제', '예측', '오차', '절대오차', '오차율(%)']
                                    st.dataframe(
                                        df_top_errors.style.format({
                                            '실제': lambda x: f"{int(x)}" if x is not None else "N/A",
                                            '예측': lambda x: f"{x:.1f}" if x is not None else "N/A",
                                            '오차': lambda x: f"{x:+.1f}" if x is not None else "N/A",
                                            '절대오차': lambda x: f"{x:.1f}" if x is not None else "N/A",
                                            '오차율(%)': lambda x: f"{x:.1f}%" if x is not None else "N/A"
                                        }).background_gradient(subset=['절대오차'], cmap='Reds'),
                                        width='stretch',
                                        height=400
                                    )
                                    # 재학습 필요 시리즈 식별 함수
                                    def identify_retrain_candidates(df_compare, mae, wmape):
                                        """재학습이 필요한 시리즈 식별"""
                                        return df_compare[
                                            ((df_compare['abs_error'] > mae * 2) |
                                             (df_compare['abs_pct_error'] > wmape * 1.5) |
                                             (df_compare['error'] > df_compare['claim_count'] * 0.5) |
                                             (df_compare['error'] < -df_compare['claim_count'] * 0.5) |
                                             ((df_compare['claim_count'] >= 10) & (df_compare['abs_pct_error'] > 30)))].copy()
                                    high_error_series = identify_retrain_candidates(df_compare, mae, wmape)
                                    if len(high_error_series) > 0:
                                        st.warning(f"⚠️ **재학습 권장**: {len(high_error_series)}개 시리즈가 과소, 과대 예측된 상태입니다.")
                                        st.session_state.high_error_series = high_error_series.copy()
                                        st.session_state.current_eval_metrics = {
                                            'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape), 'wmape': float(wmape), 'bias': float(bias)
                                        }
                                        st.info("👉 '증분 재학습' 탭에서 재학습을 실행할 수 있습니다.")
                    
                    except Exception as e:
                        st.error(f"평가 실패: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # 서브탭 3: 증분 재학습
    with result_tab3:
        st.markdown("### 🔄 증분 재학습")
        
        # 재학습 대상 확인
        if 'high_error_series' not in st.session_state or st.session_state.high_error_series is None:
            st.warning("⚠️ 먼저 '예측 정확도 평가' 탭에서 성능 평가를 실행하세요.")
            st.stop()
        
        # 현재 선택된 연월 표시
        st.info(f"**재학습 대상 월:** {eval_year}-{eval_month:02d}")
        
        # 재학습 대상 정보
        st.subheader("1️⃣ 재학습 대상")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            **선택된 시리즈:**
            - 과소/과대 예측된 시리즈: {len(st.session_state.high_error_series):,}개
            - 현재 메트릭:
              - WMAPE: {st.session_state.current_eval_metrics['wmape']:.1f}%
              - MAE: {st.session_state.current_eval_metrics['mae']:.2f}
              - Bias: {st.session_state.current_eval_metrics['bias']:+.2f}
            
            **처리 내용:**
            - 변곡점 감지 & 옵티마이저 재실행
            - 소요 시간: 시리즈당 약 2-3분
            """)
        
        with col2:
            rerun = st.button("🔄 재학습 시작", type="primary", width='stretch')
        
        if rerun:
            # 앱 전체 데이터 새로고침 함수
            def refresh_app_data():
                st.cache_data.clear()
                st.rerun()
            
            # 상태 표시 컨테이너
            status = st.empty()
            progress = st.empty()
            log = st.empty()
            
            try:
                # 재학습 대상 시리즈 저장
                retrain_dir = Path(f"artifacts/incremental/{eval_year}{eval_month:02d}")
                retrain_dir.mkdir(parents=True, exist_ok=True)
                
                retrain_file = retrain_dir / f"retrain_series_{eval_year}{eval_month:02d}.txt"
                st.session_state.high_error_series['series_id'].to_csv(
                    retrain_file,
                    index=False,
                    header=False
                )
                
                # 재학습 명령 실행
                cmd = [
                    sys.executable,
                    "train_incremental_models.py",
                    "--train-until", str(eval_year),
                    "--max-workers", "4"
                ]
                
                # 프로세스 시작
                with status.container():
                    st.info("🔄 재학습 프로세스 시작 중...")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 진행 상태 초기화
                progress_bar = progress.progress(0)
                log_output = []
                completed_series_set = set()
                skipped_series_set = set()
                total_series = None
                start_time = time.time()

                # 실시간 출력 처리
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break

                    if output:
                        log_output.append(output.strip())

                        import re
                        # 완료된 시리즈 집계
                        if "[PROGRESS] Completed training for series" in output:
                            match = re.search(r"series ([^\s]+)", output)
                            if match:
                                series_id = match.group(1)
                                completed_series_set.add(series_id)
                        # 스킵된 시리즈 집계
                        if "[PROGRESS] Skipped series" in output:
                            match = re.search(r"series ([^\s]+)", output)
                            if match:
                                series_id = match.group(1)
                                skipped_series_set.add(series_id)
                        # total_series가 None이면 추정 (최초 메시지에서 추출)
                        if total_series is None:
                            total_series = len(st.session_state.high_error_series)
                        # 진행률 계산: 완료/(전체-스킵)
                        processed_count = len(completed_series_set) + len(skipped_series_set)
                        current_progress = min(processed_count / max(total_series, 1), 1.0)
                        progress_bar.progress(current_progress)

                        with status.container():
                            st.info(f"""
                            🔄 재학습 진행 중...
                            - 완료: {processed_count}/{total_series} 시리즈
                            - 진행률: {current_progress*100:.1f}%
                            """)

                        # 로그 업데이트
                        log.code('\n'.join(log_output[-15:]))
                
                # 프로세스 완료 확인
                if process.returncode == 0:
                    # 메타데이터 저장
                    meta_file = retrain_dir / f"retrain_meta_{eval_year}{eval_month:02d}.json"
                    meta_data = {
                        "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "target_month": f"{eval_year}-{eval_month:02d}",
                        "total_series": len(st.session_state.high_error_series),
                        "retrain_series": len(st.session_state.high_error_series),
                        "metrics_before": st.session_state.current_eval_metrics
                    }
                    
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(meta_data, f, indent=2, ensure_ascii=False)
                    
                    # 완료 상태 표시
                    elapsed_time = time.time() - start_time
                    with status.container():
                        st.success(f"""
                        ✅ 재학습 완료!
                        - 총 {total_series}개 시리즈 재학습 완료
                        - 소요 시간: {elapsed_time:.1f}초
                        """)
                    
                    with st.spinner("재학습 후 예측 및 EWS 점수 생성 중..."):
                        # 예측 생성
                        forecast_cmd = [
                            sys.executable,
                            "batch.py",
                            "forecast",
                            "--month-new", f"{eval_year}-{eval_month+1:02d}"
                        ]
                        forecast_result = subprocess.run(forecast_cmd)
                        if forecast_result.returncode == 0:
                            # EWS 점수 계산
                            ews_cmd = [
                                sys.executable,
                                "batch.py",
                                "ews",
                                "--month", f"{eval_year}-{eval_month+1:02d}"
                            ]
                            ews_result = subprocess.run(ews_cmd)
                            if ews_result.returncode == 0:
                                st.success("✅ 재학습, 예측, EWS 점수 생성까지 모두 완료!")
                            else:
                                st.error("❌ EWS 점수 계산 실패")
                        else:
                            st.error("❌ 새로운 예측 생성 실패")
                        # 새로고침 제거: 성공/실패 메시지 유지
                else:
                    with status.container():
                        st.error("❌ 재학습 실패")
                        error_output = process.stderr.read()
                        st.code(error_output)
            
            except Exception as e:
                with status.container():
                    st.error(f"❌ 재학습 실행 오류: {e}")
                import traceback
                st.code(traceback.format_exc())

# Tab 4: Reconcile 보정
with tab4:
    st.header("🔧 Reconcile 보정")
    
    # 년/월 선택
    col_date1, col_date2, col_date3 = st.columns([1, 1, 2])
    with col_date1:
        current_year = datetime.now().year
        reconcile_year = st.selectbox("연도", range(2024, current_year + 2), key="reconcile_year")
    with col_date2:
        reconcile_month = st.selectbox("월", range(1, 13), key="reconcile_month")
    with col_date3:
        st.info(f"**Reconcile 대상 월:** {reconcile_year}-{reconcile_month:02d}")
    
    month_key = f"{reconcile_year}-{reconcile_month:02d}"
    
    month_dir = Path(f"artifacts/incremental/{reconcile_year}{reconcile_month:02d}")
    reconcile_dir = Path(f"artifacts/reconcile/{reconcile_year}{reconcile_month:02d}")
    
    # 처리 결과 확인
    predict_vs_actual = month_dir / f"predict_vs_actual_{reconcile_year}{reconcile_month:02d}.csv"
    
    if predict_vs_actual.exists():
        st.success(f"✅ {month_key} 처리 결과 존재")
        
        # KPI 확인
        df_compare = pd.read_csv(predict_vs_actual, encoding='utf-8-sig')
        
        # MAPE 계산
        valid_mask = df_compare['claim_count'] > 0
        if valid_mask.sum() > 0:
            mape = (df_compare[valid_mask]['abs_error'] / df_compare[valid_mask]['claim_count']).mean()
        else:
            mape = np.nan
        
        # Bias 계산
        bias = df_compare['error'].mean() / df_compare['claim_count'].mean() if df_compare['claim_count'].mean() > 0 else np.nan
        
        # KPI 통과 여부 계산
        mape_pass = mape < 0.20 if not np.isnan(mape) else False
        bias_pass = abs(bias) < 0.05 if not np.isnan(bias) else False
        kpi_pass = mape_pass and bias_pass
        
        st.subheader("📊 현재 KPI")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{mape:.2%}", delta=f"목표: <20%", delta_color="inverse" if not mape_pass else "normal")
        
        with col2:
            st.metric("|Bias|", f"{abs(bias):.4f}", delta=f"목표: <0.05", delta_color="inverse" if not bias_pass else "normal")
        
        with col3:
            st.metric("MAE", f"{df_compare['abs_error'].mean():.2f}")
        
        with col4:
            if kpi_pass:
                st.success("✅ KPI 통과")
            else:
                st.error("❌ KPI 미달")
        
        st.markdown("---")
        
        # Reconcile 실행
        st.subheader("🚀 보정 실행")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            stage = st.selectbox(
                "보정 단계",
                ['all', 'bias', 'seasonal', 'optuna'],
                index=0,
                help="all: 모든 단계 순차 실행, bias: Bias Map만, seasonal: 계절성 재추정만, optuna: Optuna 튜닝만"
            )
        
        with col2:
            run_reconcile = st.button("🔧 Reconcile 실행", type="primary", width='stretch', key="run_reconcile")
        
        stage_descriptions = {
            'all': '모든 단계 순차 실행 (Bias Map → Seasonal → Optuna)',
            'bias': 'Stage 1: Bias Map - 주간 평균 오차 보정',
            'seasonal': 'Stage 2: Seasonal Recalibration - 최근 2년 계절성 재추정',
            'optuna': 'Stage 3: Optuna Tuning - 하이퍼파라미터 최적화'
        }
        
        with col3:
            st.info(stage_descriptions[stage])
        
        if run_reconcile:
            st.markdown("---")
            st.subheader("⏳ Reconcile 실행 중...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander("🔍 상세 로그", expanded=True)
            
            try:
                status_text.text(f"실행 중: {stage} 단계...")
                progress_bar.progress(30)
                
                cmd = [
                    sys.executable, "reconcile_pipeline.py",
                    "--year", str(reconcile_year),
                    "--month", str(reconcile_month),
                    "--stage", stage
                ]
                
                with log_container:
                    st.code(f"실행: {' '.join(cmd)}", language="bash")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd()
                )
                
                with log_container:
                    if result.stdout:
                        st.text("STDOUT:")
                        st.code(result.stdout, language="text")
                    if result.stderr:
                        st.text("STDERR:")
                        st.code(result.stderr, language="text")
                
                if result.returncode == 0:
                    progress_bar.progress(100)
                    status_text.text("✅ 완료!")
                    st.success("🎉 Reconcile 보정 완료!")
                    
                    # 결과 표시
                    summary_file = reconcile_dir / f"reconcile_summary_{reconcile_year}{reconcile_month:02d}.json"
                    
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        
                        st.subheader("📊 보정 결과")
                        
                        # 초기 vs 최종 KPI
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**초기 KPI**")
                            initial = summary['initial_kpi']
                            st.metric("MAPE", f"{initial.get('MAPE', 0):.2%}")
                            st.metric("|Bias|", f"{abs(initial.get('Bias', 0)):.4f}")
                            st.metric("MAE", f"{initial.get('MAE', 0):.2f}")
                        
                        with col2:
                            st.markdown("**최종 KPI**")
                            if summary.get('final_kpi'):
                                final = summary['final_kpi']
                                st.metric("MAPE", f"{final.get('MAPE', 0):.2%}")
                                st.metric("|Bias|", f"{abs(final.get('Bias', 0)):.4f}")
                                st.metric("MAE", f"{final.get('MAE', 0):.2f}")
                        
                        # 통과 여부
                        if summary.get('pass'):
                            st.success("✅ KPI 목표 달성!")
                        else:
                            st.warning("⚠️ KPI 미달 - 추가 조치 필요")
                        
                        # 실행된 단계
                        if summary.get('stages_run'):
                            st.markdown("**실행된 단계**")
                            for stage_result in summary['stages_run']:
                                with st.expander(f"📌 {stage_result['stage']}"):
                                    st.json(stage_result['improvement'])
                else:
                    progress_bar.progress(0)
                    status_text.text("❌ 실패")
                    st.error(f"Reconcile 실행 실패 (exit code: {result.returncode})")
            
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ 오류")
                st.error(f"오류 발생: {e}")
                import traceback
                with log_container:
                    st.code(traceback.format_exc(), language="text")
        
        # 기존 Reconcile 결과 표시
        if reconcile_dir.exists():
            st.markdown("---")
            st.subheader("📁 Reconcile 결과 파일")
            
            files = list(reconcile_dir.glob("*"))
            if files:
                for file in sorted(files):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(f"📄 {file.name}")
                    with col2:
                        st.caption(f"{file.stat().st_size / 1024:.1f} KB")
                    with col3:
                        if file.suffix == '.csv':
                            if st.button(f"보기", key=f"view_rec_{file.name}"):
                                df = pd.read_csv(file, encoding='utf-8-sig')
                                st.dataframe(df.head(100), width='stretch')
                        elif file.suffix == '.json':
                            if st.button(f"보기", key=f"view_rec_{file.name}"):
                                with open(file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                st.json(data)
                        elif file.suffix == '.txt':
                            if st.button(f"보기", key=f"view_rec_{file.name}"):
                                import chardet
                                with open(file, 'rb') as f:
                                    raw = f.read()
                                    encoding = chardet.detect(raw)['encoding'] or 'utf-8'
                                    content = raw.decode(encoding, errors='replace')
                                st.text(content)
    else:
        st.info(f"{month_key} 처리 결과가 없습니다. 먼저 Tab 3에서 데이터를 처리하세요.")

# Tab 5: 통계
with tab5:
    st.header("📊 통계")
    
    # 증분 디렉토리 확인
    incremental_dir = Path("artifacts/incremental")
    
    # 서브탭: 업로드 히스토리 / 성능 트렌드
    stat_tab1, stat_tab2 = st.tabs(["📅 업로드 히스토리", "📈 성능 트렌드"])
    
    # 서브탭 1: 업로드 히스토리
    with stat_tab1:
        st.subheader("📥 데이터 업로드 이력")
        
        # 증분 디렉토리 전체 스캔
        incremental_dirs = list(incremental_dir.glob("*"))
        
        if not incremental_dirs:
            st.info("아직 업로드된 데이터가 없습니다.")
        else:
            # 업로드 이력 데이터 수집
            upload_history = []
            
            for inc_dir in sorted(incremental_dirs, reverse=True):
                if not inc_dir.is_dir():
                    continue
                
                try:
                    year = int(inc_dir.name[:4])
                    month = int(inc_dir.name[4:6])
                    
                    # 메타데이터 파일
                    meta_file = inc_dir / f"retrain_meta_{year}{month:02d}.json"
                    if meta_file.exists():
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                    else:
                        meta_data = None
                    
                    # 예측-실제 비교 파일
                    # 업로드한 csv의 총 시리즈 수

                    # 시리즈 수는 retrain_meta_YYYYMM.json의 total_series 값 사용
                    meta_file = inc_dir / f"retrain_meta_{year}{month:02d}.json"
                    if meta_file.exists():
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        n_series = meta_data.get('total_series', 0)
                        metrics = meta_data.get('metrics_before', {})
                        mae = metrics.get('mae', None)
                        wmape = metrics.get('wmape', None)
                    else:
                        n_series = 0
                        mae = None
                        wmape = None

                    # 전체 성능 메트릭에서 MAE, WMAPE 가져오기
                    meta_file = inc_dir / f"retrain_meta_{year}{month:02d}.json"
                    if meta_file.exists():
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        metrics = meta_data.get('metrics_before', {})
                        mae = metrics.get('mae', None)
                        wmape = metrics.get('wmape', None)
                    else:
                        mae = None
                        wmape = None
                    
                    # 재학습 리스트
                    retrain_file = inc_dir / f"retrain_series_{year}{month:02d}.txt"
                    if retrain_file.exists():
                        import chardet
                        with open(retrain_file, 'rb') as f:
                            raw = f.read()
                            encoding = chardet.detect(raw)['encoding'] or 'utf-8'
                            lines = raw.decode(encoding, errors='replace').splitlines()
                            n_retrain = len(lines)
                    else:
                        n_retrain = 0
                    
                    upload_history.append({
                        'year': year,
                        'month': month,
                        'date': meta_data['eval_date'] if meta_data else None,
                        'n_series': n_series,
                        'mae': mae,
                        'wmape': wmape,
                        'n_retrain': n_retrain
                    })
                
                except Exception as e:
                    st.error(f"디렉토리 {inc_dir.name} 처리 오류: {e}")
            
            if upload_history:
                # 데이터프레임 변환
                df_history = pd.DataFrame(upload_history)
                df_history['period'] = df_history.apply(
                    lambda x: f"{x['year']}-{x['month']:02d}", axis=1
                )
                # 최신순 정렬
                df_history = df_history.sort_values(['year', 'month'], ascending=[False, False])
                # session_state에 저장
                st.session_state.df_history = df_history
                # 테이블 표시
                st.dataframe(
                    df_history[['period', 'date', 'n_series', 'mae', 'wmape', 'n_retrain']].rename(columns={
                        'period': '대상 월',
                        'date': '평가 일시',
                        'n_series': '시리즈 수',
                        'mae': 'MAE',
                        'wmape': 'WMAPE(%)',
                        'n_retrain': '재학습'
                    }).style.format({
                        '시리즈 수': lambda x: f"{int(x)}" if x is not None else "N/A",
                        '재학습': lambda x: f"{int(x)}" if x is not None else "N/A",
                        'MAE': lambda x: f"{x:.2f}" if x is not None else "N/A",
                        'WMAPE(%)': lambda x: f"{x:.1f}%" if x is not None else "N/A"
                    }),
                    width='stretch'
                )
    
    # 서브탭 2: 성능 트렌드
    with stat_tab2:
        st.subheader("📈 모델 성능 추이")
        
        if not incremental_dirs:
            st.info("아직 성능 데이터가 없습니다.")
        elif 'df_history' not in st.session_state:
            st.warning("업로드 이력 데이터가 없습니다. 먼저 업로드 히스토리 탭을 확인하세요.")
        else:
            df_history = st.session_state.df_history
            # 시계열 그래프
            fig = go.Figure()
            # WMAPE 추이
            if not df_history['wmape'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df_history['period'],
                    y=df_history['wmape'],
                    name='WMAPE(%)',
                    line=dict(color='royalblue', width=2),
                    mode='lines+markers'
                ))
            # MAE 추이
            if not df_history['mae'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df_history['period'],
                    y=df_history['mae'],
                    name='MAE',
                    line=dict(color='firebrick', width=2, dash='dot'),
                    mode='lines+markers',
                    yaxis='y2'
                ))
            # 재학습 건수 (막대)
            fig.add_trace(go.Bar(
                x=df_history['period'],
                y=df_history['n_retrain'],
                name='재학습 건수',
                marker_color='lightgray',
                opacity=0.5,
                yaxis='y3'
            ))
            # 레이아웃
            fig.update_layout(
                title='월별 예측 성능 및 재학습 추이',
                xaxis=dict(title='대상 월'),
                yaxis=dict(
                    title=dict(text='WMAPE(%)', font=dict(color='royalblue')),
                    tickfont=dict(color='royalblue')
                ),
                yaxis2=dict(
                    title=dict(text='MAE', font=dict(color='firebrick')),
                    tickfont=dict(color='firebrick'),
                    overlaying='y',
                    side='right'
                ),
                yaxis3=dict(
                    title=dict(text='재학습 건수', font=dict(color='gray')),
                    tickfont=dict(color='gray'),
                    overlaying='y',
                    side='right',
                    position=0.85
                ),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # 요약 통계
            if len(df_history) > 1:
                st.markdown("### 📊 성능 요약")
                
                # 최근 2개월 비교
                latest = df_history.iloc[0]
                prev = df_history.iloc[1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    wmape_change = latest['wmape'] - prev['wmape']
                    st.metric(
                        "WMAPE 변화",
                        f"{latest['wmape']:.1f}%",
                        f"{wmape_change:+.1f}%",
                        delta_color="inverse"
                    )
                
                with col2:
                    mae_change = latest['mae'] - prev['mae']
                    st.metric(
                        "MAE 변화",
                        f"{latest['mae']:.2f}",
                        f"{mae_change:+.2f}",
                        delta_color="inverse"
                    )
                
                with col3:
                    retrain_change = latest['n_retrain'] - prev['n_retrain']
                    st.metric(
                        "재학습 건수 변화",
                        f"{latest['n_retrain']}건",
                        f"{retrain_change:+d}건",
                        delta_color="inverse"
                    )
    
    if incremental_dir.exists():
        month_dirs = [d for d in incremental_dir.iterdir() if d.is_dir()]
        
        if month_dirs:
            st.subheader(f"처리된 월: {len(month_dirs)}개")
            
            # 월별 요약 로드
            summaries = []
            for month_dir in sorted(month_dirs):
                summary_files = list(month_dir.glob("summary_*.json"))
                if summary_files:
                    with open(summary_files[0], 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        summaries.append(summary)
            
            if summaries:
                df_summary = pd.DataFrame(summaries)
                
                # 월별 트렌드
                if 'year' in df_summary.columns and 'month' in df_summary.columns:
                    df_summary['month_key'] = df_summary['year'].astype(str) + '-' + df_summary['month'].astype(str).str.zfill(2)
                    df_summary = df_summary.sort_values('month_key')
                    
                    st.line_chart(df_summary.set_index('month_key')[['total_records', 'series_count']])
                
                # 전체 통계
                st.dataframe(df_summary, width='stretch')
        else:
            st.info("처리된 월별 데이터가 없습니다.")
    else:
        st.info("증분학습 디렉토리가 없습니다.")

# 푸터
st.markdown("---")
st.caption("CJ Quality-Cycles - 월별 증분학습 시스템 v1.0")
