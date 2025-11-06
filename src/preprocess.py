# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import product

GROUP_COLS = ["플랜트", "제품범주2", "중분류"]

def create_complete_yearweek_grid(df: pd.DataFrame, pad_to: tuple) -> pd.DataFrame:
    """
    전체 연도-주차 그리드 생성
    
    Args:
        df: 데이터프레임 (year, week 컬럼 필요)
        pad_to: (year, week) 튜플로 마지막 연도-주차 지정
    
    Returns:
        year, week 컬럼을 가진 데이터프레임
    """
    min_year = df['year'].min()
    max_year, max_week = pad_to
    
    # 모든 연도-주차 조합 생성
    year_weeks = []
    for year in range(min_year, max_year + 1):
        # 마지막 연도는 max_week까지만
        end_week = max_week if year == max_year else 53
        for week in range(1, end_week + 1):
            year_weeks.append((year, week))
    
    return pd.DataFrame(year_weeks, columns=['year', 'week'])

def generate_series_keys(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """유니크한 시리즈 조합 생성"""
    # 그룹 컬럼들의 유니크한 조합 추출
    series_keys = df[group_cols].drop_duplicates()
    series_keys['series_id'] = series_keys.astype(str).agg('|'.join, axis=1)
    return series_keys

def weekly_agg_from_counts(
    df: pd.DataFrame,
    date_col: str = "제조일자",
    value_col: str = "count",
    group_cols: List[str] = GROUP_COLS,
    pad_to_date: tuple = None
) -> pd.DataFrame:
    """
    원본 데이터를 주간 단위로 집계하고 패딩을 적용
    
    Args:
        df: 원본 데이터프레임
        date_col: 날짜 컬럼명
        value_col: 집계할 값 컬럼명
        group_cols: 그룹화 컬럼 목록
        pad_to_date: (year, week) 튜플로 패딩 종료 주차 지정
    
    Returns:
        집계 및 패딩된 데이터프레임 (series_id, group_cols, year, week, claim_count)
    """
    # 1. 날짜를 ISO 연도/주차로 변환
    dates = pd.to_datetime(df[date_col])
    df = df.assign(
        year=dates.dt.isocalendar().year,
        week=dates.dt.isocalendar().week
    )
    
    # 2. 주간 단위로 집계
    agg = (df.groupby([*group_cols, 'year', 'week'])[value_col]
             .sum()
             .reset_index()
             .rename(columns={value_col: 'claim_count'}))
    
    # 3. 시리즈 키 생성
    series_keys = generate_series_keys(agg, group_cols)
    
    # 패딩이 필요한 경우
    if pad_to_date is not None:
        # 4. 모든 연도-주차 조합 생성
        yearweek_grid = create_complete_yearweek_grid(agg, pad_to_date)
        
        # 5. 모든 시리즈와 연도-주차의 조합 생성
        full_grid = (pd.merge(
            yearweek_grid, 
            series_keys,
            how='cross'  # cartesian product
        ))
        
        # 6. 실제 데이터와 병합
        result = (full_grid.merge(
            agg,
            on=[*group_cols, 'year', 'week'],
            how='left'
        ))
        
        # 7. 결측값을 0으로 채우기
        result['claim_count'] = result['claim_count'].fillna(0.0)
        
    else:
        # 패딩이 필요없는 경우 series_id만 추가
        result = agg.merge(series_keys[['series_id', *group_cols]], on=group_cols)
    
    # 8. 최종 컬럼 순서 조정 및 정렬
    final_cols = ['series_id'] + group_cols + ['year', 'week', 'claim_count']
    return result[final_cols].sort_values(['series_id', 'year', 'week'])
