import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기 및 온난화 속도 비교")
st.write(
    "서울 기온 데이터를 바탕으로 100년당 기온 상승 폭을 분석하고 미래 기온을 예측합니다."
)


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # 컬럼 공백 제거 및 날짜 형변환
    df.columns = df.columns.str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 결측치 제거 및 2025년 이하 데이터만 추출
    df = df.dropna(subset=["평균기온"])
    df = df[df["연도"] <= 2025]

    # 관측일이 300일 이상인 해만 필터링
    day_counts = df.groupby("연도")["평균기온"].count()
    valid_years = day_counts[day_counts >= 300].index
    df_valid = df[df["연도"].isin(valid_years)]

    # 연도별 평균기온 계산 및 정렬
    annual_df = (
        df_valid.groupby("연도")["평균기온"].mean().reset_index()
    )
    annual_df = annual_df.sort_values("연도").reset_index(drop=True)

    # 1908년 기준 경과 연수(독립변수 X)
    annual_df["X_1908"] = annual_df["연도"] - 1908

    return annual_df


try:
    annual_df = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 기울기 계산 (전체 기간 vs 최근 20년)
# 2-1) 전체 기간 회귀분석
X_all = annual_df["X_1908"].values
Y_all = annual_df["평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)
rate_100_all = slope_all * 100  # 100년당 상승 온도의 변화량
corr_all = np.corrcoef(annual_df["연도"], Y_all)[0, 1]

# 2-2) 최근 20년 회귀분석
df_recent20 = annual_df.tail(20)
X_recent = df_recent20["X_1908"].values
Y_recent = df_recent20["평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)
rate_100_recent = slope_recent * 100  # 최근 20년 기준 100년당 상승 폭

start_year = int(annual_df["연도"].min())
end_year = int(annual_df["연도"].max())
num_years = len(annual_df)

recent_start = int(df_recent20["연도"].min())
recent_end = int(df_recent20["연도"].max())

# 3. 기본 데이터 정보
st.info(
    f"📌 **회귀선 데이터 기준**: 총 **{num_years}개 해**의 데이터 활용 "
    f"({start_year}년 ~ {end_year}년) | 상관계수(r): **{corr_all:.4f}**"
)

# 4. 100년당 기온 상승 폭 비교 카드 (화면에 크게 나란히 표시)
st.subheader("🔥 온난화 속도 비교 (100년당 기온 상승 폭)")

col_slope1, col_slope2 = st.columns(2)

with col_slope1:
    st.markdown(
        f"""
        <div style="background-color: #e8f4f8; padding: 22px; border-radius: 12px; border-left: 6px solid #1f77b4; text-align: center;">
            <h4 style="margin:0; color: #2c3e50; font-size: 1.1rem;">🌐 전체 기간 기준 (1908~2025)</h4>
            <h1 style="margin:12px 0 6px 0; color: #1f77b4; font-size: 2.8rem;">+{rate_100_all:.2f} °C <span style="font-size: 1.2rem;">/ 100년</span></h1>
            <p style="margin:0; color: #666; font-size: 0.9rem;">분석 대상: {start_year}년 ~ {end_year}년 ({num_years}개 해)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_slope2:
    st.markdown(
        f"""
        <div style="background-color: #fde8e8; padding: 22px; border-radius: 12px; border-left: 6px solid #d62728; text-align: center;">
            <h4 style="margin:0; color: #2c3e50; font-size: 1.1rem;">⚡ 최근 20년 기준 ({recent_start}~{recent_end})</h4>
            <h1 style="margin:12px 0 6px 0; color: #d62728; font-size: 2.8rem;">+{rate_100_recent:.2f} °C <span style="font-size: 1.2rem;">/ 100년</span></h1>
            <p style="margin:0; color: #666; font-size: 0.9rem;">분석 대상: {recent_start}년 ~ {recent_end}년 (20개 해)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# 5. 연도 선택 슬라이더 및 예측 결과
st.subheader("🔮 연도별 예상 기온 예측")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 전체 기간 회귀선 기준 예측값
pred_x = selected_year - 1908
predicted_temp_all = slope_all * pred_x + intercept_all
predicted_temp_recent = slope_recent * pred_x + intercept_recent

st.markdown(
    f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
        <h3 style="margin:0; color: #495057;">{selected_year}년 서울 예상 평균기온</h3>
        <h1 style="margin:10px 0; color: #ff4b4b; font-size: 3.2rem;">{predicted_temp_all:.2f} °C</h1>
        <p style="margin:0; color: #6c757d; font-size: 0.95rem;">(최근 20년 추세 지속 시 추정값: <strong>{predicted_temp_recent:.2f} °C</strong>)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 6. Plotly 시각화
st.subheader("📊 연도별 기온 변화 및 회귀 직선 비교")

# 그래프용 X축 범위 (1900~2100년)
plot_years = np.arange(1900, 2101)
plot_x = plot_years - 1908

# 회귀선 Y값 계산
y_pred_all = slope_all * plot_x + intercept_all
y_pred_recent = slope_recent * plot_x + intercept_recent

fig = go.Figure()

# 1) 전체 연평균 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=annual_df["연도"],
        y=annual_df["평균기온"],
        mode="markers",
        name="연평균 관측 기온",
        marker=dict(color="#1f77b4", size=8, opacity=0.7),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>",
    )
)

# 2) 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=y_pred_all,
        mode="lines",
        name=f"전체 기간 회귀선 (+{rate_100_all:.2f}°C/100년)",
        line=dict(color="#1f77b4", width=2.5, dash="dash"),
        hovertemplate="%{x}년 (전체 기준): %{y:.2f}°C<extra></extra>",
    )
)

# 3) 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=y_pred_recent,
        mode="lines",
        name=f"최근 20년 회귀선 (+{rate_100_recent:.2f}°C/100년)",
        line=dict(color="#d62728", width=2.5, dash="dot"),
        hovertemplate="%{x}년 (최근 20년 기준): %{y:.2f}°C<extra></extra>",
    )
)

# 4) 선택된 연도 강조 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp_all],
        mode="markers",
        name=f"선택 연도 ({selected_year}년)",
        marker=dict(color="#ff7f0e", size=14, symbol="star"),
        hovertemplate=f"선택 연도: {selected_year}년<br>예상 기온: {predicted_temp_all:.2f}°C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도 (Year)",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d", range=[1895, 2105]),
    hovermode="closest",
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"
    ),
    height=550,
)

st.plotly_chart(fig, use_container_width=True)
