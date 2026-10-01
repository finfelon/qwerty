import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("1908년 이후 데이터를 바탕으로 회귀 분석을 수행하여 연도별 예상 기온을 예측합니다.")


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # 컬럼 공백 제거 및 날짜 형변환
    df.columns = df.columns.str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 결측치 제거 및 2025년 이전 데이터만 추출
    df = df.dropna(subset=["평균기온"])
    df = df[df["연도"] <= 2025]

    # 관측일이 300일 이상인 해만 필터링
    day_counts = df.groupby("연도")["평균기온"].count()
    valid_years = day_counts[day_counts >= 300].index
    df_valid = df[df["연도"].isin(valid_years)]

    # 연도별 평균기온 계산
    annual_df = (
        df_valid.groupby("연도")["평균기온"].mean().reset_index()
    )

    # 1908년부터 경과한 연수(독립변수 X) 계산
    annual_df["X_1908"] = annual_df["연도"] - 1908

    return annual_df


# 데이터 불러오기
try:
    annual_df = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 선형 회귀 분석 및 상관계수 계산
X = annual_df["X_1908"].values
Y = annual_df["평균기온"].values

# 회귀 계수 (기울기, 절편)
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 (피어슨 상관계수)
correlation = np.corrcoef(annual_df["연도"], Y)[0, 1]

start_year = int(annual_df["연도"].min())
end_year = int(annual_df["연도"].max())
num_years = len(annual_df)

# 3. 화면 상단 정보 및 요약 지표 출력
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{correlation:.4f}")

st.markdown("---")

# 4. 연도 선택 슬라이더 및 예측 결과 크게 표시
st.subheader("🔮 연도별 예상 기온 예측")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 예측 공식: Y = slope * (Selected_Year - 1908) + intercept
pred_x = selected_year - 1908
predicted_temp = slope * pred_x + intercept

st.markdown(
    f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
        <h3 style="margin:0; color: #333;">{selected_year}년 서울 예상 평균기온</h3>
        <h1 style="margin:10px 0 0 0; color: #ff4b4b; font-size: 3rem;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=1,
)

# 5. Plotly 시각화 (산점도 및 회귀 직선)
st.subheader("📊 연도별 기온 변화 및 회귀선 분석")

# 전체 범위(1900~2100)의 회귀선 좌표 생성
plot_years = np.arange(1900, 2101)
plot_x = plot_years - 1908
plot_pred_y = slope * plot_x + intercept

fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=annual_df["연도"],
        y=annual_df["평균기온"],
        mode="markers",
        name="연평균 관측 기온",
        marker=dict(color="#1f77b4", size=8, opacity=0.8),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>",
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        hovertemplate="%{x}년 회귀 예측값: %{y:.2f}°C<extra></extra>",
    )
)

# 선택된 연도 강조 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택한 해 ({selected_year}년)",
        marker=dict(color="#ff7f0e", size=14, symbol="star"),
        hovertemplate=f"선택 연도: {selected_year}년<br>예상 기온: {predicted_temp:.2f}°C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도 (Year)",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d", range=[1895, 2105]),
    hovermode="closest",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    height=550,
)

st.plotly_chart(fig, use_container_width=True)
