"""
EWS 조기경보 대시보드
6개월 예측 기반 고위험 시리즈 모니터링
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(page_title="EWS 조기경보 시스템", layout="wide", page_icon="⚠️")

# 스타일
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚠️ EWS 조기경보 시스템")
st.markdown("**6개월 예측 기반 고위험 클레임 시리즈 식별**")

# 데이터 로드
@st.cache_data
def load_ews_scores():
    """EWS 스코어 로드"""
    ews_path = Path("artifacts/metrics/ews_scores_2024_01.csv")
    if not ews_path.exists():
        return None
    return pd.read_csv(ews_path)

@st.cache_data
def load_forecast_data():
    """예측 데이터 로드"""
    forecast_path = Path("artifacts/forecasts/2024/forecast_2024_01.parquet")
    if not forecast_path.exists():
        return None
    return pd.read_parquet(forecast_path)

@st.cache_data
def load_training_results():
    """모델 학습 결과 로드"""
    results_path = Path("artifacts/models/base_monthly/training_results.csv")
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)

# 데이터 로드
df_ews = load_ews_scores()
df_forecast = load_forecast_data()
df_results = load_training_results()

if df_ews is None:
    st.error("❌ EWS 스코어 파일이 없습니다.")
    st.info("먼저 EWS 스코어링을 실행하세요:")
    st.code("python -m src.ews_scoring_v2 --forecast artifacts/forecasts/2024/forecast_2024_01.parquet --output artifacts/metrics/ews_scores_2024_01.csv", language="bash")
    st.stop()

if df_forecast is None:
    st.error("❌ 예측 파일이 없습니다.")
    st.stop()

# 사이드바 - 필터
st.sidebar.header("🔍 필터")

# EWS 레벨 필터
level_options = ["전체"] + sorted(df_ews['level'].unique().tolist())
selected_level = st.sidebar.multiselect(
    "EWS 레벨",
    level_options,
    default=["전체"]
)

# 신뢰도 필터
conf_threshold = st.sidebar.slider(
    "최소 신뢰도 (F2)",
    0.0, 1.0, 0.0, 0.1
)

# 증가율 필터
ratio_threshold = st.sidebar.slider(
    "최소 증가율 (F1)",
    0.0, 5.0, 0.0, 0.1
)

# 스코어 필터
score_threshold = st.sidebar.slider(
    "최소 EWS 스코어",
    0.0, 1.0, 0.0, 0.1
)

# 필터 적용
df_filtered = df_ews.copy()

if "전체" not in selected_level:
    df_filtered = df_filtered[df_filtered['level'].isin(selected_level)]

df_filtered = df_filtered[
    (df_filtered['f2_conf'] >= conf_threshold) &
    (df_filtered['f1_ratio'] >= ratio_threshold) &
    (df_filtered['ews_score'] >= score_threshold)
]

# 메인 대시보드
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
    st.metric("필터 결과", f"{len(df_filtered):,}개")

st.markdown("---")

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Top 위험 시리즈", 
    "📊 5-Factor 분석", 
    "📈 월별 예측 추이",
    "🔍 시리즈 검색",
    "💾 데이터 다운로드"
])

# Tab 1: Top 위험 시리즈
with tab1:
    st.header("🏆 Top 고위험 시리즈")
    
    # Top N 선택
    top_n = st.slider("표시할 시리즈 수", 10, 100, 20, 10)
    
    # 정렬 기준 선택
    sort_by = st.selectbox(
        "정렬 기준",
        ["EWS 스코어", "증가율 (F1)", "신뢰도 (F2)", "계절성 (F3)", "진폭 (F4)", "변곡 (F5)"]
    )
    
    sort_col_map = {
        "EWS 스코어": "ews_score",
        "증가율 (F1)": "f1_ratio",
        "신뢰도 (F2)": "f2_conf",
        "계절성 (F3)": "f3_season",
        "진폭 (F4)": "f4_ampl",
        "변곡 (F5)": "f5_inflect"
    }
    
    df_top = df_filtered.nlargest(top_n, sort_col_map[sort_by])
    
    # 테이블 표시
    st.dataframe(
        df_top[[
            'rank', 'series_id', 'ews_score', 'level',
            'f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect',
            'rationale'
        ]].style.background_gradient(subset=['ews_score'], cmap='YlOrRd')
        .format({
            'ews_score': '{:.3f}',
            'f1_ratio': '{:.2f}x',
            'f2_conf': '{:.2f}',
            'f3_season': '{:.2f}',
            'f4_ampl': '{:.2f}',
            'f5_inflect': '{:.2f}'
        }),
        use_container_width=True,
        height=600
    )
    
    # 상세 정보 선택
    if len(df_top) > 0:
        st.markdown("---")
        st.subheader("상세 정보")
        
        selected_series = st.selectbox(
            "시리즈 선택",
            df_top['series_id'].tolist(),
            key="top_series_select"
        )
        
        if selected_series:
            series_info = df_top[df_top['series_id'] == selected_series].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**기본 정보**")
                st.write(f"- **시리즈 ID**: {series_info['series_id']}")
                st.write(f"- **EWS 레벨**: {series_info['level']}")
                st.write(f"- **EWS 스코어**: {series_info['ews_score']:.3f}")
                st.write(f"- **순위**: {int(series_info['rank'])}")
                
            with col2:
                st.markdown("**5-Factor 점수**")
                st.write(f"- **F1 증가율**: {series_info['f1_ratio']:.2f}x")
                st.write(f"- **F2 신뢰도**: {series_info['f2_conf']:.2f}")
                st.write(f"- **F3 계절성**: {series_info['f3_season']:.2f}")
                st.write(f"- **F4 진폭**: {series_info['f4_ampl']:.2f}")
                st.write(f"- **F5 변곡**: {series_info['f5_inflect']:.2f}")
            
            st.markdown(f"**근거**: {series_info['rationale']}")

# Tab 2: 5-Factor 분석
with tab2:
    st.header("📊 5-Factor 분포 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # F1 증가율 분포
        fig1 = px.histogram(
            df_filtered,
            x='f1_ratio',
            nbins=50,
            title='F1: 증가율 분포',
            labels={'f1_ratio': '증가율 (배수)'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig1.add_vline(x=1.5, line_dash="dash", line_color="red", 
                       annotation_text="1.5x (50% 증가)")
        st.plotly_chart(fig1, use_container_width=True)
        
        # F3 계절성 분포
        fig3 = px.histogram(
            df_filtered,
            x='f3_season',
            nbins=50,
            title='F3: 계절성 강도 분포',
            labels={'f3_season': '계절성'},
            color_discrete_sequence=['#4ECDC4']
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # F5 변곡 분포
        fig5 = px.histogram(
            df_filtered,
            x='f5_inflect',
            nbins=50,
            title='F5: 변곡점 분포',
            labels={'f5_inflect': '변곡 위험'},
            color_discrete_sequence=['#95E1D3']
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # F2 신뢰도 분포
        fig2 = px.histogram(
            df_filtered,
            x='f2_conf',
            nbins=50,
            title='F2: 신뢰도 분포',
            labels={'f2_conf': '신뢰도'},
            color_discrete_sequence=['#FFD93D']
        )
        fig2.add_vline(x=0.5, line_dash="dash", line_color="orange",
                       annotation_text="0.5 (중간)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # F4 진폭 분포
        fig4 = px.histogram(
            df_filtered,
            x='f4_ampl',
            nbins=50,
            title='F4: 진폭 분포',
            labels={'f4_ampl': '정규화 진폭'},
            color_discrete_sequence=['#6BCB77']
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # EWS 레벨 분포
        level_counts = df_filtered['level'].value_counts()
        fig_level = px.pie(
            values=level_counts.values,
            names=level_counts.index,
            title='EWS 레벨 분포',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_level, use_container_width=True)

# Tab 3: 월별 예측 추이
with tab3:
    st.header("📈 월별 예측 추이")
    
    # 시리즈 선택
    series_list = df_filtered['series_id'].tolist()
    
    if series_list:
        selected_series_forecast = st.selectbox(
            "시리즈 선택",
            series_list,
            key="forecast_series_select"
        )
        
        if selected_series_forecast:
            # 예측 데이터 필터링
            series_forecast = df_forecast[df_forecast['series_id'] == selected_series_forecast]
            
            if len(series_forecast) > 0:
                # 월 레이블 생성
                series_forecast = series_forecast.copy()
                series_forecast['month_label'] = series_forecast['year'].astype(str) + '-' + series_forecast['month'].astype(str).str.zfill(2)
                series_forecast = series_forecast.sort_values(['year', 'month'])
                
                # 예측 그래프
                fig = go.Figure()
                
                # 예측값
                fig.add_trace(go.Scatter(
                    x=series_forecast['month_label'],
                    y=series_forecast['y_pred'],
                    mode='lines+markers',
                    name='예측값',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                # 신뢰구간
                fig.add_trace(go.Scatter(
                    x=series_forecast['month_label'],
                    y=series_forecast['y_pred_upper'],
                    mode='lines',
                    name='Upper 95% CI',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=series_forecast['month_label'],
                    y=series_forecast['y_pred_lower'],
                    mode='lines',
                    name='Lower 95% CI',
                    line=dict(width=0),
                    fillcolor='rgba(0,100,255,0.2)',
                    fill='tonexty',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f'6개월 예측: {selected_series_forecast}',
                    xaxis_title='월',
                    yaxis_title='예측 클레임 수',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_pred = series_forecast['y_pred'].mean()
                    st.metric("평균 예측", f"{avg_pred:.2f}건")
                
                with col2:
                    max_pred = series_forecast['y_pred'].max()
                    max_month = series_forecast.loc[series_forecast['y_pred'].idxmax(), 'month_label']
                    st.metric("최대 예측", f"{max_pred:.2f}건", delta=f"{max_month}")
                
                with col3:
                    total_pred = series_forecast['y_pred'].sum()
                    st.metric("6개월 합계", f"{total_pred:.0f}건")
                
                # 상세 테이블
                st.markdown("**상세 예측 데이터**")
                st.dataframe(
                    series_forecast[['month_label', 'y_pred', 'y_pred_lower', 'y_pred_upper']].style.format({
                        'y_pred': '{:.2f}',
                        'y_pred_lower': '{:.2f}',
                        'y_pred_upper': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("해당 시리즈의 예측 데이터가 없습니다.")
    else:
        st.info("필터 조건에 맞는 시리즈가 없습니다.")

# Tab 4: 시리즈 검색
with tab4:
    st.header("🔍 시리즈 검색")
    
    search_query = st.text_input("시리즈 ID 검색 (부분 일치)")
    
    if search_query:
        search_results = df_ews[df_ews['series_id'].str.contains(search_query, case=False, na=False)]
        
        st.write(f"**검색 결과: {len(search_results)}개**")
        
        if len(search_results) > 0:
            st.dataframe(
                search_results[[
                    'rank', 'series_id', 'ews_score', 'level',
                    'f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect'
                ]].style.background_gradient(subset=['ews_score'], cmap='RdYlGn_r')
                .format({
                    'ews_score': '{:.3f}',
                    'f1_ratio': '{:.2f}x',
                    'f2_conf': '{:.2f}',
                    'f3_season': '{:.2f}',
                    'f4_ampl': '{:.2f}',
                    'f5_inflect': '{:.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("검색 결과가 없습니다.")

# Tab 5: 데이터 다운로드
with tab5:
    st.header("💾 데이터 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 필터링된 EWS 데이터")
        st.write(f"총 {len(df_filtered)}개 시리즈")
        
        csv_ews = df_filtered.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv_ews,
            file_name=f"ews_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("### 전체 예측 데이터")
        st.write(f"총 {len(df_forecast)}개 레코드")
        
        csv_forecast = df_forecast.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv_forecast,
            file_name=f"forecast_2024_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# 사이드바 하단 - 통계 요약
st.sidebar.markdown("---")
st.sidebar.header("📈 전체 통계")
st.sidebar.write(f"**총 시리즈**: {len(df_ews):,}개")
st.sidebar.write(f"**HIGH**: {(df_ews['level']=='HIGH').sum():,}개")
st.sidebar.write(f"**MEDIUM**: {(df_ews['level']=='MEDIUM').sum():,}개")
st.sidebar.write(f"**LOW**: {(df_ews['level']=='LOW').sum():,}개")
st.sidebar.write(f"**LOW_CONF**: {(df_ews['level']=='LOW_CONF').sum():,}개")

st.sidebar.markdown("---")
st.sidebar.caption("💡 F3, F4 개선 버전 (Fallback 로직 적용)")
st.sidebar.caption(f"데이터: 2021-2023 (36개월)")
