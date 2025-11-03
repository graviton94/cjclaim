# path guard (project root → sys.path)
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -*- coding: utf-8 -*-
import importlib
import streamlit as st
import pandas as pd

# Quick runtime dependency check to provide a clear message in Streamlit
_required_pkgs = [
    "streamlit",
    "pandas",
    "numpy",
    "ruptures",
    "scipy",
    "statsmodels",
    "plotly",
]
_missing = []
for _p in _required_pkgs:
    try:
        importlib.import_module(_p)
    except Exception:
        _missing.append(_p)

if _missing:
    st.set_page_config(page_title="CJ Claim – MVP", layout="wide")
    st.title("품질 클레임 주간예측 (의존성 문제)")
    st.error(
        "다음 필수 패키지가 설치되어 있지 않거나 import에 실패했습니다: "
        + ", ".join(_missing)
    )
    st.write("프로젝트 루트에서 `python -m pip install -r requirements.txt` 를 실행하거나 README의 설치 지침을 따라주세요.")
    st.stop()

# local project imports (after deps verified)
from src.io_utils import load_min_csv
from src.preprocess import weekly_agg_from_counts, GROUP_COLS
from src.cycle_features import compute_psi, amplitude, find_peaks_basic
from src.changepoint import detect_changepoints
from src.forecasting import fit_forecast
from src.scoring import early_warning_rule

st.set_page_config(page_title="CJ Claim – MVP", layout="wide")
st.title("품질 클레임 주간예측 (제조일자 기반)")

file = st.file_uploader("CSV 업로드 (제품범주2, 플랜트, 제조일자, 중분류(보정), count)", type=["csv"])
if not file:
    st.stop()

df = load_min_csv(file)
agg = weekly_agg_from_counts(df)

# 필터링을 위한 함수
def get_filtered_options(df, plant=None, category=None):
    """선택된 조건에 따라 필터링된 옵션을 반환"""
    filtered = df.copy()
    if plant and plant != "(ALL)":
        filtered = filtered[filtered["플랜트"] == plant]
    if category and category != "(ALL)":
        filtered = filtered[filtered["제품범주2"] == category]
    return filtered

# 예측 성능이 좋은 조합 계산
@st.cache_data
def calculate_combination_scores(data):
    """각 조합별 예측 성능 계산"""
    from src.scoring import calculate_prediction_score
    
    scores = []
    for name, group in data.groupby(GROUP_COLS):
        if len(group) >= 26:  # 최소 6개월 데이터 필요
            group_sorted = group.sort_values('week')
            y = group_sorted['y'].values
            
            # 마지막 6개월을 테스트 셋으로 사용
            train_size = len(y) - 26
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            # 예측 수행
            fc = fit_forecast(pd.Series(y_train), horizon=26)
            
            if len(y_test) > 0:
                # 성능 지표 계산
                score = calculate_prediction_score(
                    y_test,
                    fc['yhat'],
                    fc['yhat_lower'],
                    fc['yhat_upper']
                )
                
                scores.append({
                    '플랜트': name[0],
                    '제품범주2': name[1],
                    '중분류': name[2],
                    'reliability_score': score['reliability_score'],
                    'mape': score['mape'],
                    'r2': score['r2'],
                    'interval_width': score['interval_width']
                })
    
    return pd.DataFrame(scores)

# 신뢰도 점수 계산
combination_scores = calculate_combination_scores(agg)

# 필터 컨트롤
st.write("### 데이터 필터")
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

# 신뢰도 점수 필터
min_reliability = c4.slider(
    "최소 신뢰도 점수",
    min_value=0,
    max_value=100,
    value=50,
    help="예측 신뢰도 점수가 이 값 이상인 조합만 표시됩니다. "
         "점수는 MAPE, R-squared, 신뢰구간 너비를 종합적으로 고려하여 계산됩니다."
)

# 신뢰도 기준을 만족하는 조합만 필터링
reliable_combinations = combination_scores[
    combination_scores['reliability_score'] >= min_reliability
]

# 플랜트 선택
available_plants = ["(ALL)"] + sorted(reliable_combinations["플랜트"].unique().tolist())
f1 = c1.selectbox("플랜트", available_plants, 0)

# 제품범주2 선택 (플랜트 기반 필터링)
filtered_scores = reliable_combinations
if f1 != "(ALL)":
    filtered_scores = filtered_scores[filtered_scores["플랜트"] == f1]
available_cats = ["(ALL)"] + sorted(filtered_scores["제품범주2"].unique().tolist())
f2 = c2.selectbox("제품범주2", available_cats, 0)

# 중분류 선택 (플랜트, 제품범주2 기반 필터링)
if f2 != "(ALL)":
    filtered_scores = filtered_scores[filtered_scores["제품범주2"] == f2]
available_mids = ["(ALL)"] + sorted(filtered_scores["중분류"].unique().tolist())
f3 = c3.selectbox("중분류", available_mids, 0)

# 선택된 조합의 신뢰도 정보 표시
if f1 != "(ALL)" or f2 != "(ALL)" or f3 != "(ALL)":
    current_scores = filtered_scores
    if f3 != "(ALL)":
        current_scores = current_scores[current_scores["중분류"] == f3]
    
    if not current_scores.empty:
        st.info(f"""
        선택된 조합의 예측 신뢰도:
        - 종합 신뢰도 점수: {current_scores['reliability_score'].mean():.1f}/100
        - 평균 예측 오차(MAPE): {current_scores['mape'].mean():.1f}%
        - 결정계수(R²): {current_scores['r2'].mean():.3f}
        - 평균 신뢰구간 너비: {current_scores['interval_width'].mean():.1f}%
        """)

# 선택된 조건으로 데이터 필터링
sel = agg.copy()
if f1 != "(ALL)": sel = sel[sel["플랜트"] == f1]
if f2 != "(ALL)": sel = sel[sel["제품범주2"] == f2]
if f3 != "(ALL)": sel = sel[sel["중분류"] == f3]

if sel.empty:
    st.warning("선택된 조건의 데이터가 없습니다.")
    st.stop()

# 전체 시리즈 합계 계산
series_data = sel.groupby('week')['y'].sum().reset_index()
for col in ['플랜트', '제품범주2', '중분류']:
    if f1 != "(ALL)" and col == '플랜트':
        series_data[col] = f1
    elif f2 != "(ALL)" and col == '제품범주2':
        series_data[col] = f2
    elif f3 != "(ALL)" and col == '중분류':
        series_data[col] = f3
    else:
        series_data[col] = '전체'

# 데이터를 연도별로 분리하고 재구성
def prepare_yearly_data(data, future_data=None):
    """데이터를 연도별로 분리하고, 월-주차 포맷으로 변환"""
    yearly_data = {}
    
    # 실제 데이터 처리
    for _, row in data.iterrows():
        year = row['week'].year
        if year not in yearly_data:
            yearly_data[year] = {'week_labels': [], 'values': []}
        
        # 월-주차 레이블 생성
        week_label = row['week'].strftime("%m월%W주차")
        yearly_data[year]['week_labels'].append(week_label)
        yearly_data[year]['values'].append(row['y'])
    
    # 예측 데이터 처리
    if future_data is not None:
        current_year = max(yearly_data.keys())
        yearly_data[current_year + 1] = {
            'week_labels': [],
            'values': future_data['yhat'].tolist()
        }
        
        future_dates = pd.date_range(
            start=data['week'].iloc[-1] + pd.Timedelta(weeks=1),
            periods=len(future_data['yhat']),
            freq='W'
        )
        
        for date in future_dates:
            week_label = date.strftime("%m월%W주차")
            yearly_data[current_year + 1]['week_labels'].append(week_label)
    
    return yearly_data

# 기본 데이터 준비
ts = series_data["y"].reset_index(drop=True)
psi = compute_psi(ts)
amp = amplitude(ts)
peaks = find_peaks_basic(ts)
cps = detect_changepoints(ts)
fc = fit_forecast(ts, horizon=26)
ews = early_warning_rule(ts)

# 연도별 데이터 준비
yearly_data = prepare_yearly_data(series_data, fc)

st.subheader("클레임 발생 패턴 분석")
import plotly.graph_objects as go

# 현재 선택된 필터 정보 표시
filter_info = f"{'전체' if f1 == '(ALL)' else f1} > {'전체' if f2 == '(ALL)' else f2} > {'전체' if f3 == '(ALL)' else f3}"
st.caption(f"선택된 필터: {filter_info}")

# 1. 시계열 데이터와 예측 그래프
st.write("### 1. 클레임 발생 추세 및 예측")

# 전체 시계열 데이터 준비
series_data_sorted = series_data.copy().sort_values('week')
future_dates = pd.date_range(
    start=series_data['week'].iloc[-1] + pd.Timedelta(weeks=1),
    periods=len(fc['yhat']),
    freq='W'
)

# 호버 텍스트 준비
def create_hover_text(row):
    week_num = int(row['week'].strftime('%W'))
    return (f"기준일: {row['week'].strftime('%y년 %m월 %W주차')}<br>"
            f"클레임 건수: {row['y']}<br>"
            f"플랜트: {row['플랜트']}<br>"
            f"제품범주2: {row['제품범주2']}<br>"
            f"중분류: {row['중분류']}")

series_data_sorted['hover_text'] = series_data_sorted.apply(create_hover_text, axis=1)

recent_fig = go.Figure()

# 실제 데이터 plotting
recent_fig.add_trace(go.Scatter(
    x=series_data_sorted['week'],
    y=series_data_sorted['y'],
    mode='lines+markers',
    name='실제 데이터',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=8),
    hovertext=series_data_sorted['hover_text'],
    hoverinfo='text'
))

# 예측 데이터의 호버 텍스트 준비
future_hover = [f"기준일: {date.strftime('%y년 %m월 %W주차')}<br>예측 클레임 건수: {val:.1f}"
                for date, val in zip(future_dates, fc['yhat'])]

# 예측 데이터 plotting
recent_fig.add_trace(go.Scatter(
    x=future_dates,
    y=fc['yhat'],
    mode='lines',
    name='예측',
    line=dict(color='#ff7f0e', dash='dash', width=2),
    hovertext=future_hover,
    hoverinfo='text'
))

# 신뢰 구간
recent_fig.add_trace(go.Scatter(
    x=future_dates,
    y=fc['yhat_upper'],
    fill=None,
    mode='lines',
    line=dict(color='rgba(255, 127, 14, 0.3)', width=0),
    showlegend=False,
    hoverinfo='skip'
))

recent_fig.add_trace(go.Scatter(
    x=future_dates,
    y=fc['yhat_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(color='rgba(255, 127, 14, 0.3)', width=0),
    name='95% 신뢰구간',
    hoverinfo='skip'
))

# x축 눈금 설정 (월별로)
date_range = pd.date_range(
    start=series_data_sorted['week'].min(),
    end=future_dates[-1],
    freq='MS'  # 각 월의 시작일
)

recent_fig.update_layout(
    xaxis_title="연도-월",
    yaxis_title="클레임 발생 건수",
    hovermode='closest',
    xaxis=dict(
        tickmode='array',
        ticktext=[d.strftime('%y.%m') for d in date_range],
        tickvals=date_range,
        tickangle=-45,
        type='date'
    ),
    legend_title="구분",
    height=400,
    margin=dict(b=100)  # x축 레이블이 잘리지 않도록 여백 추가
)

st.plotly_chart(recent_fig, width='stretch')

# 2. 연도별 패턴 비교 (히트맵)
st.write("### 2. 연도별 월간 패턴 비교")

# 월별 평균 데이터 준비
monthly_data = []
month_names = []
year_labels = []
current_year = max(series_data['week'].dt.year)

for year in sorted(yearly_data.keys()):
    if year <= current_year:  # 실제 데이터만 포함
        data = yearly_data[year]
        monthly_avg = []
        
        # 월별 데이터 집계
        current_month = None
        month_sum = 0
        month_count = 0
        
        for i, label in enumerate(data['week_labels']):
            month = int(label.split('월')[0])
            
            if current_month is None:
                current_month = month
                
            if current_month != month:
                monthly_avg.append(month_sum / month_count if month_count > 0 else 0)
                current_month = month
                month_sum = data['values'][i]
                month_count = 1
            else:
                month_sum += data['values'][i]
                month_count += 1
                
        # 마지막 월 처리
        if month_count > 0:
            monthly_avg.append(month_sum / month_count)
            
        while len(monthly_avg) < 12:  # 12개월로 맞추기
            monthly_avg.append(0)
            
        monthly_data.append(monthly_avg)
        year_labels.append(str(year))
        
        if not month_names:
            month_names = [f"{i}월" for i in range(1, 13)]

# 히트맵 생성
heatmap_fig = go.Figure(data=go.Heatmap(
    z=monthly_data,
    x=month_names,
    y=year_labels,
    colorscale='RdYlBu_r',
    text=[[f"{val:.1f}" for val in row] for row in monthly_data],
    texttemplate="%{text}",
    textfont={"size": 12},
    hoverongaps=False,
    colorbar=dict(title='평균<br>클레임<br>건수')
))

heatmap_fig.update_layout(
    title="월별 평균 클레임 발생 패턴",
    xaxis_title="월",
    yaxis_title="연도",
    height=300
)

st.plotly_chart(heatmap_fig, width='stretch')

# 3. 주요 지표 표시
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "PSI (Peak Significance)",
        f"{psi:.2f}" if pd.notna(psi) else "N/A",
        help="피크의 중요도를 나타내는 지수. 높을수록 뚜렷한 주기성 존재"
    )
with col2:
    st.metric(
        "Amplitude (진폭)",
        f"{amp:.1f}",
        help="클레임 발생의 최대-최소 차이. 변동성 크기를 나타냄"
    )
with col3:
    st.metric(
        "EWS Alert",
        "ON" if ews["alert"] else "OFF",
        delta="주의" if ews["alert"] else "정상",
        delta_color="inverse",
        help="클레임 패턴의 이상 징후 감지 여부"
    )
with col4:
    st.metric(
        "EWS z-score",
        f"{ews['score']:.2f}",
        help="|z| > 2: 주의 수준, |z| > 3: 경고 수준"
    )

# 지표 설명
st.write("### 주요 지표 설명")
metrics_cols = st.columns(4)

with metrics_cols[0]:
    st.metric("PSI (Peak Significance Index)", f"{psi:.2f}" if pd.notna(psi) else "NaN")
    st.write("""
    - 피크의 중요도를 나타내는 지수
    - 피크의 높이(prominence)와 폭(width)의 비율
    - 높을수록 뚜렷한 주기성 존재
    """)

with metrics_cols[1]:
    st.metric("Amplitude (진폭)", f"{amp:.1f}")
    st.write("""
    - 시계열의 최대값과 최소값의 차이
    - 클레임 발생의 변동 폭을 표시
    - 높을수록 클레임 발생의 변동성이 큼
    """)

with metrics_cols[2]:
    st.metric("EWS Alert (조기 경보)", "ON" if ews["alert"] else "OFF")
    st.write("""
    - 클레임 패턴의 이상 징후 감지
    - ON: 주의 필요
    - OFF: 정상 범위
    """)

with metrics_cols[3]:
    st.metric("EWS z-score", f"{ews['score']:.2f}")
    st.write("""
    - 표준화된 이상 점수
    - |z| > 2: 주의 수준
    - |z| > 3: 경고 수준
    """)
