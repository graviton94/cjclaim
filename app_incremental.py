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
st.sidebar.header("⚙️ 설정")

# 연도/월 선택
current_year = datetime.now().year
selected_year = st.sidebar.selectbox("연도", range(2024, current_year + 2))
selected_month = st.sidebar.selectbox("월", range(1, 13))
month_key = f"{selected_year}-{selected_month:02d}"

st.sidebar.markdown("---")
st.sidebar.info(f"**대상 월:** {month_key}")

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
    models_dir = Path("artifacts/models/base_2021_2023")
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
                        
                        # JSON 구조: {series_id, plant, product_cat2, mid_category, data: [{year, week, claim_count, ...}]}
                        data_records = data.get('data', [])
                        
                        if data_records:
                            last_record = data_records[-1]
                            last_week = f"{last_record['year']}-W{last_record['week']:02d}"
                        else:
                            last_week = None
                        
                        metadata.append({
                            'plant': data.get('plant', 'Unknown'),
                            'product_cat2': data.get('product_cat2', 'Unknown'),
                            'mid_category': data.get('mid_category', 'Unknown'),
                            'series_id': data.get('series_id', 'Unknown'),
                            'total_records': len(data_records),
                            'last_week': last_week,
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
            else:
                st.subheader("🎯 예측 시리즈 선택")
                
                # Top 5 EWS 위험 시리즈 계산
                st.markdown("### � EWS 위험도 Top 5")
                st.caption("모델 예측 기반: 예측 확실도 × EWS 임계값 근접도")
                
                with st.spinner("Top 5 위험 시리즈 분석 중..."):
                    ews_candidates = []
                    models_dir = Path("artifacts/models")
                    
                    # 디버깅 정보
                    total_series = len(metadata_df)
                    processed = 0
                    skipped_no_data = 0
                    skipped_no_model = 0
                    skipped_low_ratio = 0
                    errors = 0
                    
                    # 설정
                    FORECAST_WEEKS = 4  # 4주 예측
                    EWS_THRESHOLD_MULTIPLIER = 1.5  # 과거 평균의 1.5배 이상이면 경고
                    
                    for idx, row in metadata_df.iterrows():
                        try:
                            # JSON 데이터 로드
                            with open(row['json_file'], 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            
                            data_records = json_data.get('data', [])
                            if len(data_records) < 52:  # 최소 1년 데이터 필요
                                skipped_no_data += 1
                                continue
                            
                            # 모델 파일 로드
                            series_id = row['series_id']
                            safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                                           .replace('|', '_').replace('?', '_').replace('*', '_')
                                           .replace('<', '_').replace('>', '_').replace('"', '_'))
                            model_path = models_dir / f"{safe_filename}.pkl"
                            
                            if not model_path.exists():
                                skipped_no_model += 1
                                continue
                            
                            # 모델 로드
                            import pickle
                            with open(model_path, 'rb') as f:
                                model_result = pickle.load(f)
                            
                            if isinstance(model_result, dict):
                                fitted_model = model_result.get('model')
                            else:
                                fitted_model = model_result
                            
                            if fitted_model is None:
                                skipped_no_model += 1
                                continue
                            
                            # 예측 생성 (4주)
                            forecast_obj = fitted_model.get_forecast(steps=FORECAST_WEEKS)
                            forecast_mean = forecast_obj.predicted_mean
                            forecast_ci_obj = forecast_obj.conf_int(alpha=0.05)  # 95% 신뢰구간
                            
                            # 예측값 및 신뢰구간
                            yhat_values = forecast_mean if isinstance(forecast_mean, np.ndarray) else forecast_mean.values
                            yhat_lower = forecast_ci_obj.iloc[:, 0].values if hasattr(forecast_ci_obj, 'iloc') else forecast_ci_obj[:, 0]
                            yhat_upper = forecast_ci_obj.iloc[:, 1].values if hasattr(forecast_ci_obj, 'iloc') else forecast_ci_obj[:, 1]
                            
                            # 음수 처리
                            yhat_values = np.maximum(yhat_values, 0)
                            yhat_lower = np.maximum(yhat_lower, 0)
                            yhat_upper = np.maximum(yhat_upper, 0)
                            
                            # 예측 평균
                            forecast_avg = yhat_values.mean()
                            
                            # 과거 평균 (최근 26주)
                            recent_data = data_records[-26:] if len(data_records) >= 26 else data_records
                            historical_avg = sum(r['claim_count'] for r in recent_data) / len(recent_data)
                            
                            # 1. 예측 확실도 (Prediction Confidence)
                            # 신뢰구간 폭의 역수 (좁을수록 확실)
                            ci_width = (yhat_upper - yhat_lower).mean()
                            if ci_width > 0 and forecast_avg > 0:
                                confidence_score = 1 / (1 + ci_width / (forecast_avg + 0.1))  # 0~1 사이
                            else:
                                confidence_score = 0
                            
                            # 2. EWS Score (Early Warning Score)
                            # 예측값이 과거 평균 대비 얼마나 높은지
                            if historical_avg > 0:
                                ews_ratio = forecast_avg / historical_avg
                                # EWS 임계값(1.5배) 근접도
                                ews_proximity = abs(ews_ratio - EWS_THRESHOLD_MULTIPLIER) / EWS_THRESHOLD_MULTIPLIER
                                ews_score = 1 / (1 + ews_proximity)  # 임계값에 가까울수록 1
                            else:
                                ews_ratio = 0
                                ews_score = 0
                            
                            # 3. 종합 위험도 점수
                            # 예측이 확실하고(confidence_score 높음) + EWS 임계값에 가까움(ews_score 높음)
                            risk_score = confidence_score * ews_score * (1 + ews_ratio * 0.1)  # 예측값도 반영
                            
                            # 예측값이 임계값보다 낮으면 제외
                            if ews_ratio < 1.0:
                                skipped_low_ratio += 1
                                continue
                            
                            processed += 1
                            
                            ews_candidates.append({
                                'series_id': series_id,
                                'plant': row['plant'],
                                'product_cat2': row['product_cat2'],
                                'mid_category': row['mid_category'],
                                'forecast_avg': forecast_avg,
                                'historical_avg': historical_avg,
                                'ews_ratio': ews_ratio,
                                'confidence_score': confidence_score,
                                'ews_score': ews_score,
                                'risk_score': risk_score,
                                'json_file': row['json_file']
                            })
                        except Exception as e:
                            errors += 1
                            continue
                    
                    # 디버깅 정보 표시
                    with st.expander("🔍 분석 상세 정보"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 시리즈", total_series)
                        with col2:
                            st.metric("성공", processed)
                        with col3:
                            st.metric("데이터 부족", skipped_no_data)
                        with col4:
                            st.metric("모델 없음", skipped_no_model)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("EWS < 1.0", skipped_low_ratio)
                        with col2:
                            st.metric("에러", errors)
                        with col3:
                            st.metric("후보", len(ews_candidates))
                    
                    if ews_candidates:
                        top_df = pd.DataFrame(ews_candidates).sort_values('risk_score', ascending=False).head(5)
                        
                        # 테이블 표시
                        display_top = top_df[['plant', 'product_cat2', 'mid_category', 'forecast_avg', 'historical_avg', 'ews_ratio', 'risk_score']].copy()
                        display_top.columns = ['플랜트', '제품범주2', '중분류', '4주 예측 평균', '과거 평균', 'EWS 비율', '위험도']
                        display_top['4주 예측 평균'] = display_top['4주 예측 평균'].round(2)
                        display_top['과거 평균'] = display_top['과거 평균'].round(2)
                        display_top['EWS 비율'] = display_top['EWS 비율'].round(2)
                        display_top['위험도'] = display_top['위험도'].round(3)
                        
                        st.dataframe(display_top, use_container_width=True, hide_index=True)
                        
                        st.caption("**위험도**: 예측 확실도 × EWS 임계값 근접도 (높을수록 위험)")
                        st.caption("**EWS 비율**: 예측값 / 과거 평균 (1.5배 이상이면 경고)")
                    else:
                        st.info("EWS 위험 시리즈가 없습니다.")
                
                st.markdown("---")
                
                # 시리즈 선택 UI - 계층적 필터링
                st.markdown("### 🔍 시리즈 검색 및 선택")
                
                col1, col2, col3 = st.columns(3)
                
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
                
                # 최종 필터링
                final_filtered = filtered_by_cat2[filtered_by_cat2['mid_category'] == selected_mid]
                
                # 시리즈가 선택되었는지 확인
                if len(final_filtered) > 0:
                    series_info = final_filtered.iloc[0]
                    series_id = series_info['series_id']
                    
                    st.info(f"✅ 선택된 시리즈: **{series_id}**")
                    
                    # 예측 설정
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        horizon_weeks = st.slider("예측 기간 (주)", 4, 26, 24, help="6개월 = 24주", key="horizon")
                    with col2:
                        ci_choice = st.selectbox("신뢰구간", ["95%", "99%"], index=0, key="ci")
                        ci = 0.99 if ci_choice == "99%" else 0.95
                        
                        # 예측 실행
                        if st.button("🔮 예측 실행", type="primary"):
                            with st.spinner(f"{series_id} 예측 중..."):
                                try:
                                    # JSON 데이터 로드
                                    import pickle
                                    from datetime import timedelta
                                    import plotly.graph_objects as go
                                    
                                    with open(series_info['json_file'], 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                    
                                    # JSON 구조: data = [{year, week, claim_count, ...}]
                                    data_records = json_data.get('data', [])
                                    
                                    if not data_records:
                                        st.error(f"시리즈 {series_id}에 데이터가 없습니다.")
                                    else:
                                        # DataFrame 생성
                                        df_hist = pd.DataFrame(data_records)
                                        df_hist['week_date'] = pd.to_datetime(
                                            df_hist['year'].astype(str) + '-W' + df_hist['week'].astype(str).str.zfill(2) + '-1',
                                            format='%Y-W%W-%w'
                                        )
                                        df_hist = df_hist.rename(columns={'claim_count': 'y'})
                                        df_hist = df_hist.sort_values('week_date')
                                        df_hist = df_hist.sort_values('week_date')
                                        
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
                                            
                                            # 모델에서 학습된 모델 객체 가져오기
                                            if isinstance(model_result, dict):
                                                fitted_model = model_result.get('model')
                                            else:
                                                fitted_model = model_result
                                            
                                            # 예측 생성
                                            forecast_obj = fitted_model.get_forecast(steps=horizon_weeks)
                                            forecast_mean = forecast_obj.predicted_mean
                                            forecast_ci_obj = forecast_obj.conf_int(alpha=1-ci)
                                            
                                            # 마지막 주차 이후 날짜 생성
                                            last_week = df_hist['week_date'].iloc[-1]
                                            future_weeks = [last_week + timedelta(weeks=i+1) for i in range(horizon_weeks)]
                                            
                                            # numpy array를 확인하고 적절히 변환
                                            yhat_values = forecast_mean if isinstance(forecast_mean, np.ndarray) else forecast_mean.values
                                            yhat_lower_values = forecast_ci_obj.iloc[:, 0].values if hasattr(forecast_ci_obj, 'iloc') else forecast_ci_obj[:, 0]
                                            yhat_upper_values = forecast_ci_obj.iloc[:, 1].values if hasattr(forecast_ci_obj, 'iloc') else forecast_ci_obj[:, 1]
                                            
                                            # 신뢰구간 음수 처리 (클레임은 음수가 될 수 없음)
                                            yhat_lower_values = np.maximum(yhat_lower_values, 0)
                                            yhat_values = np.maximum(yhat_values, 0)
                                            yhat_upper_values = np.maximum(yhat_upper_values, 0)
                                            
                                            df_forecast = pd.DataFrame({
                                                'week': future_weeks,
                                                'yhat': yhat_values,
                                                'yhat_lower': yhat_lower_values,
                                                'yhat_upper': yhat_upper_values
                                            })
                                            
                                            # 메트릭 표시
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("학습 데이터", f"{len(df_hist)}주")
                                            with col2:
                                                avg_claims = df_hist['y'].mean()
                                                st.metric("평균 클레임", f"{avg_claims:.1f}건/주")
                                            with col3:
                                                last_claim = df_hist['y'].iloc[-1]
                                                st.metric("최근 클레임", f"{last_claim:.0f}건")
                                            with col4:
                                                forecast_avg = df_forecast['yhat'].mean()
                                                change = ((forecast_avg - avg_claims) / avg_claims * 100) if avg_claims > 0 else 0
                                                st.metric("예측 평균", f"{forecast_avg:.1f}건", f"{change:+.1f}%")
                                            
                                            # 차트 생성
                                            st.subheader("📈 예측 차트")
                                            
                                            fig = go.Figure()
                                            
                                            # 과거 데이터
                                            fig.add_trace(go.Scatter(
                                                x=df_hist['week_date'],
                                                y=df_hist['y'],
                                                mode='lines+markers',
                                                name='실제 데이터',
                                                line=dict(color='#1f77b4', width=2),
                                                marker=dict(size=4)
                                            ))
                                            
                                            # 예측값
                                            fig.add_trace(go.Scatter(
                                                x=df_forecast['week'],
                                                y=df_forecast['yhat'],
                                                mode='lines+markers',
                                                name='예측',
                                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                                marker=dict(size=6)
                                            ))
                                            
                                            # 신뢰구간
                                            fig.add_trace(go.Scatter(
                                                x=df_forecast['week'].tolist() + df_forecast['week'].tolist()[::-1],
                                                y=df_forecast['yhat_upper'].tolist() + df_forecast['yhat_lower'].tolist()[::-1],
                                                fill='toself',
                                                fillcolor='rgba(255, 127, 14, 0.2)',
                                                line=dict(color='rgba(255,255,255,0)'),
                                                name=f'{ci_choice} 신뢰구간',
                                                showlegend=True
                                            ))
                                            
                                            fig.update_layout(
                                                title=f"{series_id} - {horizon_weeks}주 예측",
                                                xaxis_title="주차",
                                                yaxis_title="클레임 건수",
                                                hovermode='x unified',
                                                height=500,
                                                yaxis=dict(dtick=1),  # Y축 정수 단위
                                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                            )
                                            
                                            st.plotly_chart(fig, width='stretch')
                                            
                                            # 예측 테이블
                                            st.subheader("📋 예측 상세")
                                            
                                            df_forecast_display = df_forecast.copy()
                                            df_forecast_display['week'] = df_forecast_display['week'].dt.strftime('%Y-%m-%d')
                                            df_forecast_display['yhat'] = df_forecast_display['yhat'].round(1)
                                            df_forecast_display['yhat_lower'] = df_forecast_display['yhat_lower'].round(1)
                                            df_forecast_display['yhat_upper'] = df_forecast_display['yhat_upper'].round(1)
                                            df_forecast_display.columns = ['주차', '예측값', '하한', '상한']
                                            
                                            st.dataframe(df_forecast_display, width='stretch', hide_index=True)
                                            
                                            # 다운로드 버튼
                                            csv = df_forecast.to_csv(index=False, encoding='utf-8-sig')
                                            st.download_button(
                                                label="📥 예측 결과 다운로드 (CSV)",
                                                data=csv,
                                                file_name=f"forecast_{series_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                                                mime="text/csv"
                                            )
                                        
                                        else:
                                            st.error(f"❌ 모델 파일이 없습니다: {safe_filename}.pkl")
                                
                                except Exception as e:
                                    st.error(f"❌ 예측 실패: {str(e)}")
                                    st.exception(e)

# Tab 2: 데이터 업로드
with tab2:
    st.header("1️⃣ 월별 데이터 업로드")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            f"**{month_key} 월별 데이터 CSV 업로드**",
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
        temp_path = temp_dir / f"upload_{month_key.replace('-', '')}.csv"
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # 데이터 미리보기
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        try:
            df_preview = pd.read_csv(temp_path, encoding='utf-8-sig', nrows=10)
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
                        
                        # 결과 표시
                        month_dir = Path(f"artifacts/incremental/{selected_year}{selected_month:02d}")
                        summary_file = month_dir / f"summary_{selected_year}{selected_month:02d}.json"
                        
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
    
    # 월별 결과 디렉토리
    month_dir = Path(f"artifacts/incremental/{selected_year}{selected_month:02d}")
    
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
            predict_vs_actual = month_dir / f"predict_vs_actual_{selected_year}{selected_month:02d}.csv"
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
                top_errors = df_compare.nlargest(10, 'abs_error')[['series_id', 'week', 'claim_count', 'y_pred', 'error', 'abs_error']]
                st.dataframe(top_errors, width='stretch')
        else:
            st.info("파일이 없습니다.")
    else:
        st.info(f"{month_key} 처리 결과가 없습니다. 먼저 데이터를 업로드하고 파이프라인을 실행하세요.")

# Tab 4: Reconcile 보정
with tab4:
    st.header("🔧 Reconcile 보정")
    
    month_dir = Path(f"artifacts/incremental/{selected_year}{selected_month:02d}")
    reconcile_dir = Path(f"artifacts/reconcile/{selected_year}{selected_month:02d}")
    
    # 처리 결과 확인
    predict_vs_actual = month_dir / f"predict_vs_actual_{selected_year}{selected_month:02d}.csv"
    
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
        
        st.subheader("📊 현재 KPI")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mape_pass = mape < 0.20 if not np.isnan(mape) else False
            st.metric("MAPE", f"{mape:.2%}", delta=f"목표: <20%", delta_color="inverse" if not mape_pass else "normal")
        
        with col2:
            bias_pass = abs(bias) < 0.05 if not np.isnan(bias) else False
            st.metric("|Bias|", f"{abs(bias):.4f}", delta=f"목표: <0.05", delta_color="inverse" if not bias_pass else "normal")
        
        with col3:
            st.metric("MAE", f"{df_compare['abs_error'].mean():.2f}")
        
        with col4:
            kpi_pass = mape_pass and bias_pass
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
                    "--year", str(selected_year),
                    "--month", str(selected_month),
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
                    summary_file = reconcile_dir / f"reconcile_summary_{selected_year}{selected_month:02d}.json"
                    
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
