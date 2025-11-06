"""
Streamlit 월별 증분학습 UI
월별 데이터 업로드 → Lag 필터링 → 예측 비교 → 재학습
"""
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

st.set_page_config(page_title="품질 클레임 관리 시스템", layout="wide", page_icon="📊")

st.title("📊 품질 클레임 예측 관리 시스템")
st.markdown("**예측 대시보드 | 월별 데이터 업로드 | 증분학습 | Reconcile 보정**")

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 예측 대시보드", "📤 데이터 업로드", "📈 처리 결과", "🔧 Reconcile 보정", "📊 통계"])

# Tab 1: 예측 대시보드
with tab1:
    st.header("🔮 학습된 모델 기반 예측")
    st.markdown("**현재 학습된 모델로 향후 6개월 클레임 예측**")
    
    # 모델 및 데이터 디렉토리 확인
    models_dir = Path("artifacts/models/base_monthly")
    features_dir = Path("data/features")
    
    if not models_dir.exists() or not features_dir.exists():
        st.warning("⚠️ 학습된 모델 또는 데이터가 없습니다.")
        st.info("먼저 Base 학습을 실행하세요:")
        st.code("python batch.py train --mode base --workers 4", language="bash")
    else:
        # 모델 파일 개수 확인
        model_files = list(models_dir.glob("*.pkl"))
        
        if not model_files:
            st.error("❌ 모델 파일이 없습니다.")
        else:
            st.success(f"✅ {len(model_files)}개의 학습된 시리즈 모델 발견")
            
            # 시리즈 메타데이터 로드
            @st.cache_data
            def load_series_metadata():
                """모든 시리즈의 메타데이터 로드"""
                metadata = []
                
                for json_file in features_dir.glob("*.json"):
                    if json_file.name == "_summary.json":
                        continue
                        
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # JSON 구조: {series_id, plant, product_cat2, mid_category, data: [{year, month, claim_count, ...}]}
                        data_records = data.get('data', [])
                        
                        if data_records:
                            last_record = data_records[-1]
                            last_month = f"{last_record['year']}-{last_record['month']:02d}"
                        else:
                            last_month = None
                        
                        metadata.append({
                            'plant': data.get('plant', 'Unknown'),
                            'product_cat2': data.get('product_cat2', 'Unknown'),
                            'mid_category': data.get('mid_category', 'Unknown'),
                            'series_id': data.get('series_id', 'Unknown'),
                            'total_records': len(data_records),
                            'last_month': last_month,
                            'json_file': str(json_file)
                        })
                    except Exception as e:
                        continue
                
                return pd.DataFrame(metadata)
            
            metadata_df = load_series_metadata()
            
            if metadata_df.empty:
                st.warning("⚠️ 시리즈 메타데이터가 없습니다.")
                
                # Base 모델이 있는지 확인
                if models_dir.exists() and len(list(models_dir.glob("*.pkl"))) > 0:
                    model_count = len(list(models_dir.glob("*.pkl")))
                    st.success(f"✅ Base 모델 발견: {model_count:,}개")
                    
                    st.info("**메타데이터 생성 방법:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**1️⃣ 수동 생성 (터미널)**")
                        st.code("python generate_series_json.py", language="bash")
                        st.caption("모든 시리즈의 메타데이터를 한번에 생성")
                    
                    with col2:
                        st.markdown("**2️⃣ 자동 생성 (권장)**")
                        st.markdown("👉 `📤 데이터 업로드` 탭으로 이동")
                        st.caption("월별 데이터 업로드 시 자동으로 생성됨")
                else:
                    st.info("**먼저 Base 학습을 실행하세요:**")
                    st.code("python batch.py train --mode base --workers 4", language="bash")
                    st.caption("2021-2023 데이터로 Base 모델 학습")
                st.subheader("🎯 예측 시리즈 선택")
                
                # EWS 위험도 Top 5 표시
                st.markdown("### ⚠️ EWS 위험도 Top 5")
                st.caption("6개월 예측 기반 상대적 위험도 점수 (증가율, 변동성, 계절성, 가속도 종합)")
                
                # EWS 점수 로드 또는 계산
                ews_file = Path("artifacts/metrics/ews_scores.csv")
                
                if ews_file.exists():
                    df_ews = pd.read_csv(ews_file)
                    top5 = df_ews.head(5)
                    
                    # Top 5 테이블 표시
                    display_data = []
                    for _, row in top5.iterrows():
                        # 시리즈 정보 파싱
                        parts = row['series_id'].split('|')
                        plant = parts[0] if len(parts) > 0 else ''
                        product = parts[1] if len(parts) > 1 else ''
                        category = parts[2] if len(parts) > 2 else ''
                        
                        # MAPE 기반 신뢰도 (낮을수록 좋음)
                        mape = row.get('growth_score', 0)  # 임시로 growth_score 사용
                        confidence_pct = max(0, 100 - mape)
                        
                        display_data.append({
                            '랭킹': f"🔥 {int(row['rank'])}위",
                            '시리즈': f"{plant} | {product}",
                            '중분류': category,
                            'EWS점수': f"{row['total_score']:.1f}",
                            '신뢰도': f"{confidence_pct:.0f}%",
                            '예상시점': f"2024-{int(row.get('forecast_month', 1)):02d}",
                            '예상건수': f"{row['forecast_max']:.1f}건"
                        })
                    
                    df_display = pd.DataFrame(display_data)
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # 상세 정보 (확장 가능)
                    with st.expander("📊 위험도 점수 구성 보기"):
                        for _, row in top5.iterrows():
                            st.markdown(f"**[{int(row['rank'])}위] {row['series_id']}** - 종합 {row['total_score']:.1f}점")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("증가율", f"{row['growth_score']:.1f}", 
                                         f"{row['growth_rate_pct']:+.0f}%")
                            with col2:
                                st.metric("변동성", f"{row['volatility_score']:.1f}")
                            with col3:
                                st.metric("계절성", f"{row['seasonality_score']:.1f}")
                            with col4:
                                st.metric("가속도", f"{row['acceleration_score']:.1f}")
                            
                            st.caption(f"평균: {row['historical_mean']:.1f} → {row['forecast_mean']:.1f} 건/월 (최대: {row['forecast_max']:.1f})")
                            st.markdown("---")
                else:
                    st.warning("⚠️ EWS 점수가 계산되지 않았습니다.")
                    st.info("예측 생성 후 EWS 점수를 계산하세요:")
                    st.code("python src/ews_scoring.py --forecast artifacts/forecasts/2024/forecast_2024_01.parquet", language="bash")
                
                st.markdown("---")
                
                # 시리즈 필터링 UI
                st.markdown("### 🔮 향후 6개월 클레임 예측")
                
                col1, col2, col3 = st.columns(3)
                st.markdown("---")
                
                # 시리즈 선택 UI - 계층적 필터링
                st.markdown("### � 향후 6개월 클레임 예측")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # 1단계: 플랜트 선택
                with col1:
                    plants = sorted(metadata_df['plant'].unique().tolist())
                    selected_plant = st.selectbox("플랜트", plants, key="forecast_plant")
                
                # 플랜트 필터링 적용
                filtered_by_plant = metadata_df[metadata_df['plant'] == selected_plant]
                
                # 2단계: 제품범주2 선택
                with col2:
                    categories = sorted(filtered_by_plant['product_cat2'].unique().tolist())
                    selected_category = st.selectbox("제품범주2", categories, key="forecast_cat2")
                
                # 제품범주2 필터링 적용
                filtered_by_cat2 = filtered_by_plant[filtered_by_plant['product_cat2'] == selected_category]
                
                # 3단계: 중분류 선택
                with col3:
                    mid_categories = sorted(filtered_by_cat2['mid_category'].unique().tolist())
                    selected_mid = st.selectbox("중분류", mid_categories, key="forecast_mid")
                
                # 4단계: 신뢰구간 선택
                with col4:
                    ci_choice = st.selectbox("신뢰구간", ["95%", "99%"], index=0, key="ci")
                    ci = 0.99 if ci_choice == "99%" else 0.95
                
                # 최종 필터링
                final_filtered = filtered_by_cat2[filtered_by_cat2['mid_category'] == selected_mid]
                
                # 시리즈가 선택되었는지 확인
                if len(final_filtered) > 0:
                    series_info = final_filtered.iloc[0]
                    series_id = series_info['series_id']
                    
                    st.info(f"✅ 선택된 시리즈: **{series_id}**")
                    
                    st.markdown("---")
                    
                    # 예측 실행 버튼 - 6개월 고정
                    horizon_months = 6  # 6개월 고정
                    if st.button("🔮 예측 실행", type="primary", use_container_width=True):
                        with st.spinner(f"{series_id} 예측 중..."):
                            try:
                                # JSON 데이터 로드
                                import pickle
                                from datetime import timedelta
                                from dateutil.relativedelta import relativedelta
                                import plotly.graph_objects as go
                                
                                with open(series_info['json_file'], 'r', encoding='utf-8') as f:
                                    json_data = json.load(f)
                                
                                # JSON 구조: data = [{year, month, claim_count, ...}]
                                data_records = json_data.get('data', [])
                                
                                if not data_records:
                                    st.error(f"시리즈 {series_id}에 데이터가 없습니다.")
                                else:
                                    # DataFrame 생성
                                    df_hist = pd.DataFrame(data_records)
                                    df_hist['month_date'] = pd.to_datetime(
                                        df_hist['year'].astype(str) + '-' + df_hist['month'].astype(str).str.zfill(2) + '-01'
                                    )
                                    df_hist = df_hist.rename(columns={'claim_count': 'y'})
                                    df_hist = df_hist.sort_values('month_date')
                                    
                                    # 모델 파일 로드
                                    # series_id에서 | 구분자로 파일명 생성
                                    safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                                                   .replace('|', '_').replace('?', '_').replace('*', '_')
                                                   .replace('<', '_').replace('>', '_').replace('"', '_'))
                                    model_path = models_dir / f"{safe_filename}.pkl"
                                    
                                    st.info(f"🔍 찾는 모델 파일: `{model_path.name}`")
                                    
                                    if model_path.exists():
                                        with open(model_path, 'rb') as f:
                                            model_result = pickle.load(f)
                                        
                                        # 월별 모델: params를 SARIMAX에 직접 적용
                                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                                        
                                        # 학습 데이터 범위 결정: 2021년부터 최신 데이터까지
                                        # (2011-2020은 대부분 0이므로 제외)
                                        df_train = df_hist[df_hist['month_date'].dt.year >= 2021].copy()
                                        
                                        if len(df_train) < 12:
                                            st.error(f"훈련 데이터 부족: {len(df_train)}개월 (최소 12개월 필요)")
                                        else:
                                            y = df_train['y'].values
                                            
                                            # SARIMAX 모델 생성 및 파라미터 적용
                                            model = SARIMAX(
                                                y,
                                                order=model_result['model_spec']['order'],
                                                seasonal_order=model_result['model_spec']['seasonal_order'],
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                            )
                                            
                                            params = np.array(model_result['params'])
                                            fitted_model = model.smooth(params)
                                            
                                            # 예측 생성
                                            forecast_mean = fitted_model.forecast(steps=horizon_months)
                                            
                                            # 마지막 월 이후 날짜 생성
                                            last_month = df_train['month_date'].iloc[-1]
                                            future_months = [last_month + relativedelta(months=i+1) for i in range(horizon_months)]
                                            
                                            # 음수 처리 (클레임은 음수가 될 수 없음)
                                            yhat_values = np.maximum(forecast_mean, 0)
                                            
                                            df_forecast = pd.DataFrame({
                                                'month': future_months,
                                                'yhat': yhat_values
                                            })
                                            
                                            # 학습 기간 표시
                                            train_start_year = df_train['month_date'].dt.year.min()
                                            train_end_year = df_train['month_date'].dt.year.max()
                                            train_period = f"{train_start_year}-{train_end_year}" if train_start_year != train_end_year else str(train_start_year)
                                            
                                            # 메트릭 표시
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("학습 데이터", f"{len(df_train)}개월 ({train_period})")
                                            with col2:
                                                avg_claims = df_train['y'].mean()
                                                st.metric("평균 클레임", f"{avg_claims:.1f}건/월")
                                            with col3:
                                                last_claim = df_train['y'].iloc[-1]
                                                last_month_str = df_train['month_date'].iloc[-1].strftime('%Y-%m')
                                                st.metric(f"최근 클레임 ({last_month_str})", f"{last_claim:.0f}건")
                                            with col4:
                                                forecast_avg = df_forecast['yhat'].mean()
                                                change = ((forecast_avg - avg_claims) / avg_claims * 100) if avg_claims > 0 else 0
                                                st.metric("예측 평균", f"{forecast_avg:.1f}건", f"{change:+.1f}%")
                                            
                                            # 차트 생성
                                            st.subheader("📈 예측 차트")
                                            
                                            fig = go.Figure()
                                            
                                            # 훈련 데이터 (2021-2023)
                                            fig.add_trace(go.Scatter(
                                                x=df_train['month_date'],
                                                y=df_train['y'],
                                                mode='lines+markers',
                                                name='실제 데이터 (2021-2023)',
                                                line=dict(color='#1f77b4', width=2),
                                                marker=dict(size=4)
                                            ))
                                            
                                            # 예측값
                                            fig.add_trace(go.Scatter(
                                                x=df_forecast['month'],
                                                y=df_forecast['yhat'],
                                                mode='lines+markers',
                                                name='예측',
                                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                                marker=dict(size=6)
                                            ))
                                            
                                            fig.update_layout(
                                                title=f"{series_id} - {horizon_months}개월 예측",
                                                xaxis_title="월",
                                                yaxis_title="클레임 건수",
                                                hovermode='x unified',
                                                height=500,
                                                yaxis=dict(dtick=1),  # Y축 정수 단위
                                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # 예측 테이블
                                            st.subheader("📋 예측 상세")
                                            
                                            df_forecast_display = df_forecast.copy()
                                            df_forecast_display['month'] = df_forecast_display['month'].dt.strftime('%Y-%m')
                                            df_forecast_display['yhat'] = df_forecast_display['yhat'].round(1)
                                            df_forecast_display['yhat'] = df_forecast_display['yhat'].round(1)
                                            df_forecast_display.columns = ['월', '예측값']
                                            
                                            st.dataframe(df_forecast_display, use_container_width=True, hide_index=True)
                                            
                                            # 다운로드 버튼
                                            csv = df_forecast.to_csv(index=False, encoding='utf-8-sig')
                                            st.download_button(
                                                label="📥 예측 결과 다운로드 (CSV)",
                                                data=csv,
                                                file_name=f"forecast_{series_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                                                mime="text/csv",
                                                use_container_width=True
                                            )
                                    
                                    else:
                                        st.error(f"❌ 모델 파일이 없습니다: {safe_filename}.pkl")
                            
                            except Exception as e:
                                st.error(f"❌ 예측 실패: {str(e)}")
                                st.exception(e)

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
                    
                    # 각 레코드의 year/month 수집
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
                # 년도 추출 및 정렬
                years = sorted(set(year for year, month in year_month_data.keys()))
                months = list(range(1, 13))
                
                # 테이블 데이터 생성
                table_data = []
                for month in months:
                    row = {'월': f"{month}월"}
                    for year in years:
                        count = year_month_data.get((year, month), 0)
                        # 데이터 충분성 판단 (시리즈 수의 80% 이상이면 충분)
                        threshold = total_series * 0.8
                        if count >= threshold:
                            status = "✅"
                        elif count >= threshold * 0.5:
                            status = "⚠️"
                        else:
                            status = "❌"
                        row[f"{year}년"] = f"{status} {count}"
                    table_data.append(row)
                
                # DataFrame 생성 및 표시
                df_status = pd.DataFrame(table_data)
                st.dataframe(df_status, use_container_width=True, hide_index=True)
                
                # 범례
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
            help="발생일자 기준 1개월 데이터 (플랜트, 제품범주2, 중분류(보정), 발생일자, 제조일자, count 컬럼 필수)"
        )
    
    with col2:
        st.markdown("**필수 컬럼**")
        st.code("""
플랜트
제품범주2
중분류(보정)
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
            
            # 전체 데이터 로드
            uploaded_file.seek(0)  # 파일 포인터 초기화
            df_full = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            
            # 발생일자에서 년/월 자동 감지
            if '발생일자' in df_full.columns:
                df_full['발생일자'] = pd.to_datetime(df_full['발생일자'])
                detected_year = df_full['발생일자'].dt.year.mode()[0]  # 최빈값
                detected_month = df_full['발생일자'].dt.month.mode()[0]  # 최빈값
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
                st.metric("총 레코드", f"{len(df_full):,}건")
            with col2:
                st.metric("시리즈 수", f"{df_full.groupby(['플랜트', '제품범주2', '중분류(보정)']).ngroups:,}개")
            with col3:
                if '발생일자' in df_full.columns:
                    df_full['발생일자'] = pd.to_datetime(df_full['발생일자'])
                    date_range = f"{df_full['발생일자'].min().date()} ~ {df_full['발생일자'].max().date()}"
                    st.metric("발생일자 범위", date_range)
            with col4:
                if 'count' in df_full.columns:
                    st.metric("총 클레임 건수", f"{df_full['count'].sum():,}건")
            
            st.markdown("---")
            
            # 처리 버튼
            st.subheader("2️⃣ 파이프라인 실행")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                run_pipeline = st.button("🚀 파이프라인 실행", type="primary", width='stretch')
            
            with col2:
                show_command = st.checkbox("명령어 표시", value=False)
            
            if show_command:
                with col3:
                    st.code(f"python batch.py process --upload {temp_path} --month {month_key}", language="bash")
            
            if run_pipeline:
                st.markdown("---")
                st.subheader("⏳ 처리 중...")
                
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
                        status_text.error("❌ 파이프라인 실패")
                        st.error(f"파이프라인 실행 실패 (exit code: {result_code})")
                        st.info("💡 위의 로그를 확인하여 오류 원인을 파악하세요.")
                
                except Exception as e:
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
        
        # 파일 목록
        files = list(month_dir.glob("*"))
        
        if files:
            st.subheader("📁 생성된 파일")
            
            for file in sorted(files):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.caption(f"{file.stat().st_size / 1024:.1f} KB")
                with col3:
                    if file.suffix == '.csv':
                        if st.button(f"보기", key=f"view_{file.name}"):
                            df = pd.read_csv(file, encoding='utf-8-sig')
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
                # month 컬럼이 있으면 사용, 없으면 제외
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
            run_reconcile = st.button("🔧 Reconcile 실행", type="primary", width='stretch')
        
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
                                with open(file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                st.text(content)
    else:
        st.info(f"{month_key} 처리 결과가 없습니다. 먼저 Tab 2에서 데이터를 처리하세요.")

# Tab 5: 통계
with tab5:
    st.header("📊 전체 통계")
    
    incremental_dir = Path("artifacts/incremental")
    
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
