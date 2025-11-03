# path guard (project root → sys.path)
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -*- coding: utf-8 -*-
import importlib
import streamlit as st
import pandas as pd
import numpy as np

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
from src.forecasting import fit_forecast, safe_forecast
from src.scoring import early_warning_rule

st.set_page_config(page_title="CJ Claim – MVP", layout="wide")
st.title("품질 클레임 주간예측 (제조일자 기반)")

file = st.file_uploader("CSV 업로드 (제품범주2, 플랜트, 제조일자, 중분류(보정), count)", type=["csv"])
if not file:
    st.stop()

df = load_min_csv(file)
# 2024년 12월까지 빈 주차를 0으로 채우기
cutoff_date = pd.Timestamp('2024-12-31')
agg = weekly_agg_from_counts(df, pad_to_date=cutoff_date)

# 필터링을 위한 함수
def get_filtered_options(df, plant=None, category=None):
    """선택된 조건에 따라 필터링된 옵션을 반환"""
    filtered = df.copy()

# 예측 성능이 좋은 조합 계산
@st.cache_data
def calculate_combination_scores(data: pd.DataFrame, ci: float = 0.99) -> pd.DataFrame:
    """각 조합별 예측 성능 계산

    반환 컬럼: 플랜트, 제품범주2, 중분류, reliability_score, mae_score, rmse_score,
    width_score, interval_score, volatility_score, r2
    """
    from src.scoring import calculate_prediction_score

    scores = []
    for name, group in data.groupby(GROUP_COLS):
        group_sorted = group.sort_values('week')
        if len(group_sorted) < 26:
            continue
            
        # 2024년 12월까지의 데이터를 학습에 사용
        cutoff = pd.Timestamp('2024-12-31')
        train_mask = group_sorted['week'] <= cutoff
        test_mask = group_sorted['week'] > cutoff
        
        if not train_mask.any():
            continue
            
        y_train = group_sorted[train_mask]['y'].values
        y_test = group_sorted[test_mask]['y'].values if test_mask.any() else np.array([])
        
        try:
            fc = safe_forecast(pd.Series(y_train), horizon=26, seasonal_order=(0,1,1,52), ci=ci)
        except Exception:
            # if forecasting fails, skip this group
            continue

        # get arrays and truncate/pad to test length
        yhat = np.asarray(fc.get('yhat', []))[:len(y_test)]
        yhat_lower = np.asarray(fc.get('yhat_lower', []))[:len(y_test)]
        yhat_upper = np.asarray(fc.get('yhat_upper', []))[:len(y_test)]

        if len(y_test) == 0 or yhat.size == 0:
            continue

        # calculate score (guard internally)
        try:
            score = calculate_prediction_score(np.asarray(y_test), yhat, yhat_lower, yhat_upper)
        except Exception:
            continue

        scores.append({
            '플랜트': name[0],
            '제품범주2': name[1],
            '중분류': name[2],
            'reliability_score': float(score.get('reliability_score', 0.0)),
            'mae_score': float(score.get('mae_score', 0.0)),
            'rmse_score': float(score.get('rmse_score', 0.0)),
            'width_score': float(score.get('width_score', 0.0)),
            'interval_score': float(score.get('interval_score', 0.0)),
            'volatility_score': float(score.get('volatility_score', 0.0)),
            'r2': float(score.get('r2', 0.0))
        })

    if len(scores) == 0:
        return pd.DataFrame(columns=['플랜트','제품범주2','중분류','reliability_score','mae_score','rmse_score','width_score','interval_score','volatility_score','r2'])

    return pd.DataFrame(scores)

# 신뢰수준 선택 (신뢰구간)
ci_choice = st.selectbox("신뢰구간 선택", ["95%", "99%"], index=1, help="예측 신뢰구간 수준을 선택하세요. 품질관리에서 더 보수적인 99%를 권장합니다.")
ci = 0.99 if ci_choice == "99%" else 0.95

# 신뢰도 점수 계산
combination_scores = calculate_combination_scores(agg, ci)

# 1) 최소 신뢰도 점수 선택 (맨 위)
st.write("### 신뢰도 기반 조합 필터")
min_reliability = st.slider(
    "최소 신뢰도 점수",
    min_value=0,
    max_value=100,
    value=50,
    help="이 값 이상인 조합만 아래 목록에 표시됩니다. 기본값 50"
)

# 2) 조건을 만족하는 조합들을 점수 순으로 테이블로 보여주기
reliable_combinations = combination_scores[combination_scores['reliability_score'] >= min_reliability]
reliable_combinations_sorted = reliable_combinations.sort_values('reliability_score', ascending=False)

st.write(f"### 조건을 만족하는 조합 (총 {len(reliable_combinations_sorted)}개)")
if reliable_combinations_sorted.empty:
    st.info("선택한 신뢰도 조건을 만족하는 조합이 없습니다. 신뢰도 기준을 낮춰보세요.")
    st.stop()

# show ranked table with top columns
rank_table = reliable_combinations_sorted.reset_index(drop=True)
rank_table.index = rank_table.index + 1
st.dataframe(rank_table[['플랜트','제품범주2','중분류','reliability_score']].rename_axis('rank'))

# 3) 사용자가 테이블에서 선택할 수 있도록 selectbox (테이블 클릭 선택은 기본 streamlit에서 제한적임)
combo_options = [f"{r['플랜트']} > {r['제품범주2']} > {r['중분류']} (score {r['reliability_score']:.1f})" for _, r in reliable_combinations_sorted.iterrows()]
selected_combo = st.selectbox("테이블에서 조합 선택 (또는 아래 상세 필터 활용)", combo_options)

# parse selection into f1,f2,f3 defaults
if selected_combo:
    sel_parts = selected_combo.split(' > ')
    sel_plant = sel_parts[0]
    sel_category = sel_parts[1]
    sel_sub = sel_parts[2].split(' (')[0]
else:
    sel_plant = sel_category = sel_sub = None

# 4) 상세 필터 (선택사항) — 사용자가 선택하거나 override 가능
c1, c2, c3 = st.columns(3)
available_plants = ["(ALL)"] + sorted(combination_scores['플랜트'].unique().tolist())
f1 = c1.selectbox("플랜트", available_plants, index=(available_plants.index(sel_plant) if sel_plant in available_plants else 0))

# 제품범주2 선택 (플랜트 기반 필터링)
filtered_scores = combination_scores if f1 == "(ALL)" else combination_scores[combination_scores['플랜트'] == f1]
available_cats = ["(ALL)"] + sorted(filtered_scores['제품범주2'].unique().tolist())
f2 = c2.selectbox("제품범주2", available_cats, index=(available_cats.index(sel_category) if sel_category in available_cats else 0))

# 중분류
filtered_scores = filtered_scores if f2 == "(ALL)" else filtered_scores[filtered_scores['제품범주2'] == f2]
available_mids = ["(ALL)"] + sorted(filtered_scores['중분류'].unique().tolist())
f3 = c3.selectbox("중분류", available_mids, index=(available_mids.index(sel_sub) if sel_sub in available_mids else 0))

# 선택된 조합의 신뢰도 정보 표시
current_scores = reliable_combinations_sorted
if f1 != "(ALL)":
    current_scores = current_scores[current_scores['플랜트'] == f1]
if f2 != "(ALL)":
    current_scores = current_scores[current_scores['제품범주2'] == f2]
if f3 != "(ALL)":
    current_scores = current_scores[current_scores['중분류'] == f3]

if not current_scores.empty:
    st.info(f"종합 신뢰도: {current_scores['reliability_score'].mean():.1f}/100 — 조합 수: {len(current_scores)}")

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
series_data_sorted = series_data.sort_values('week')
cutoff = pd.Timestamp('2024-12-31')
train_mask = series_data_sorted['week'] <= cutoff

ts_train = series_data_sorted[train_mask]["y"].values
psi = compute_psi(ts_train)
amp = amplitude(ts_train)
peaks = find_peaks_basic(ts_train)
cps = detect_changepoints(ts_train)
fc = safe_forecast(pd.Series(ts_train), horizon=26, ci=ci)
ews = early_warning_rule(pd.Series(ts_train))

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
    start=pd.Timestamp('2025-01-01'),  # 2025년 1월부터 예측 시작
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
    name=f'{int(ci*100)}% 신뢰구간',
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

for year in sorted(yearly_data.keys()):
    if year <= 2024:  # 2024년까지의 실제 데이터만 포함
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
