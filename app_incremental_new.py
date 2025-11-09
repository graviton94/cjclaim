"""
Streamlit 통합 품질 클레임 관리 시스템
EWS 조기경보 | 월별 데이터 업로드 | Lag 필터링 | 예측 비교 | 재학습
"""
import streamlit as st
import pandas as pd
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

# 앱 전체 데이터 새로고침 함수
def refresh_app_data():
    st.cache_data.clear()
    st.experimental_rerun()

# 나머지 코드는 이전과 동일...

# 재학습 실행 시
if rerun:
    # 상태 표시 컨테이너
    status = st.empty()
    progress = st.empty()
    log = st.empty()
    
    try:
        # 재학습 대상 시리즈 저장
        retrain_dir = Path(f"artifacts/incremental/{eval_year}{eval_month:02d}")
        retrain_dir.mkdir(parents=True, exist_ok=True)
        
        retrain_file = retrain_dir / f"retrain_series_{eval_year}{eval_month:02d}.txt"
        high_error_series['series_id'].to_csv(
            retrain_file,
            index=False,
            header=False
        )
        
        # 재학습 명령 실행
        cmd = [
            sys.executable,
            "batch.py",
            "retrain",
            "--month", f"{eval_year}-{eval_month:02d}",
            "--series-list", str(retrain_file),
            "--workers", "4"
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
        completed_series = 0
        total_series = len(high_error_series)
        start_time = time.time()
        
        # 실시간 출력 처리
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            
            if output:
                log_output.append(output.strip())
                
                # 진행률 업데이트
                if "Completed training for series" in output:
                    completed_series += 1
                    progress = min(completed_series / total_series, 1.0)
                    progress_bar.progress(progress)
                    
                    with status.container():
                        st.info(f"""
                        🔄 재학습 진행 중...
                        - 완료: {completed_series}/{total_series} 시리즈
                        - 진행률: {progress*100:.1f}%
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
                "total_series": len(df_compare),
                "retrain_series": len(high_error_series),
                "metrics_before": {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape),
                    "wmape": float(wmape),
                    "bias": float(bias)
                }
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
            
            # 다음 단계 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.button("1️⃣ 새로운 예측 생성", use_container_width=True):
                    with st.spinner("새로운 예측 생성 중..."):
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
                                st.success("✅ 새로운 예측 및 EWS 점수 생성 완료!")
                            else:
                                st.error("❌ EWS 점수 계산 실패")
                        else:
                            st.error("❌ 새로운 예측 생성 실패")
            
            with col2:
                if st.button("2️⃣ 데이터 새로고침", use_container_width=True):
                    refresh_app_data()
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