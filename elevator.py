import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import datetime
import time

# ==================================================
# SECTION 11 — UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Smart Elevator AI", page_icon="🏢", layout="wide")

# Custom Dark Mode & Professional UI CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stAlert { border-radius: 8px; }
    h1, h2, h3 { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# ==================================================
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    except Exception:
        st.error("Dataset not found. Please upload 'Elevator predictive-maintenance-dataset.csv'.")
        st.stop()
        
    # 5. Handle missing values (Drop NA in vibration)
    df = df.dropna(subset=['vibration']).copy()
    
    # 6. Remove duplicates
    df = df.drop_duplicates()
    
    # FEATURE ENGINEERING: Synthesizing requested columns from raw data
    # 2. Convert Timestamp (Synthesized starting from Jan 1, 2024, 5 mins per ID)
    df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
    
    # 4. Rename and map columns
    df['Temperature'] = 20 + (df['revolutions'] / 10) + np.random.normal(0, 1.5, len(df))
    df['Load'] = (df['revolutions'] / 8).astype(int).clip(0, 20) # Persons
    
    # Synthesize Failure events (Extreme vibration & temp)
    df['Failure'] = np.where((df['vibration'] > 65) & (df['Temperature'] > 30), 1, 0)
    
    # 3. Sort by timestamp
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

df_raw = load_and_preprocess()

# ==================================================
# SECTION 2 — SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("⚙️ Control Panel")
    
    # 1. Date Range Filter
    min_date, max_date = df_raw['Timestamp'].min().date(), df_raw['Timestamp'].max().date()
    date_range = st.date_input("Date Range Filter", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # 2. Elevator ID
    elevator_id = st.selectbox("Elevator ID", ["Elevator A (Main)", "Elevator B (Service)"])
    
    # 3-5. Threshold Sliders
    temp_thresh = st.slider("Temperature Threshold (°C)", 20.0, 50.0, 32.0)
    vib_thresh = st.slider("Vibration Threshold (Hz)", 10.0, 100.0, 55.0)
    load_thresh = st.slider("Load Threshold (Persons)", 1, 25, 15)
    
    # 6. Rolling Window
    roll_window = st.selectbox("Rolling Window Trends", [3, 7, 14], format_func=lambda x: f"{x} Days")
    
    # Toggles
    st.markdown("### AI Engines")
    toggle_anomaly = st.toggle("🚨 Enable Anomaly Detection", value=True)
    toggle_predict = st.toggle("🧠 Enable ML Failure Prediction", value=True)
    
    if st.button("🔄 Reset Filters"):
        st.rerun()

# Apply Date Filter
if len(date_range) == 2:
    mask = (df_raw['Timestamp'].dt.date >= date_range[0]) & (df_raw['Timestamp'].dt.date <= date_range[1])
    df = df_raw[mask].copy()
else:
    df = df_raw.copy()

# Downsample for UI Performance (Plotly crashes on 100k rows)
df_sample = df.iloc[::20].copy() if len(df) > 10000 else df.copy()

# ==================================================
# SECTION 6 & 9 — RISK SCORING & MAINTENANCE REC.
# ==================================================
latest = df.iloc[-1]
if latest['vibration'] > vib_thresh and latest['Temperature'] > temp_thresh:
    risk_status, risk_color, risk_score = "Critical", "#ef4444", 95
    rec_msg = "🚨 Immediate Maintenance Required within 48 hours."
elif latest['vibration'] > vib_thresh or latest['Temperature'] > temp_thresh:
    risk_status, risk_color, risk_score = "Warning", "#f59e0b", 65
    rec_msg = "⚠️ Schedule Preventive Maintenance."
else:
    risk_status, risk_color, risk_score = "Stable", "#10b981", 15
    rec_msg = "✅ System Operating Normally."

# ==================================================
# SECTION 3 — KPI DASHBOARD
# ==================================================
st.title(f"🏢 Predictive Maintenance OS: {elevator_id}")
st.markdown(f"**Status:** <span style='color:{risk_color}'>{rec_msg}</span>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Total Failures", df['Failure'].sum(), delta_color="inverse")
c3.metric("Avg Temperature", f"{df['Temperature'].mean():.1f} °C")
c4.metric("Avg Vibration", f"{df['vibration'].mean():.1f} Hz")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Avg Load", f"{df['Load'].mean():.1f} Persons")
c6.metric("Volatility Index", f"{df['vibration'].std():.2f} (StdDev)")
c7.metric("Current Risk Score", f"{risk_score}%")
c8.metric("System Health", risk_status)

# ==================================================
# SECTION 4 — VISUALIZATION ENGINE
# ==================================================
st.markdown("---")
st.header("📊 Telemetry & Visualization Engine")

t1, t2, t3 = st.tabs(["Time Series Analysis", "Correlation & Stability", "Advanced Analytics"])

with t1:
    # 1, 2, 3, 4, 5, 6: Multi-line comparison with Rolling Avg & Failures
    df_sample[f'Vib_Rolling_{roll_window}D'] = df_sample['vibration'].rolling(roll_window*24).mean()
    
    fig_ts = px.line(df_sample, x='Timestamp', y=['vibration', 'Temperature', f'Vib_Rolling_{roll_window}D'], 
                     title="Telemetry Over Time (Zoom Enabled)", template="plotly_dark")
    
    # Highlight Failures
    failures = df_sample[df_sample['Failure'] == 1]
    fig_ts.add_trace(go.Scatter(x=failures['Timestamp'], y=failures['vibration'], 
                                mode='markers', marker=dict(color='red', size=10), name='Failure Event'))
    st.plotly_chart(fig_ts, use_container_width=True)
    
[Image of elevator predictive maintenance system architecture]

with t2:
    col_a, col_b = st.columns(2)
    with col_a:
        # 8. Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df[['vibration', 'Temperature', 'Load', 'humidity', 'revolutions']].corr()
        fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_b:
        # 10. Stability vs Unstable zone
        st.subheader("Stability vs Load Profile")
        fig_scatter = px.scatter(df_sample, x='Load', y='vibration', color='Failure', 
                                 template="plotly_dark", color_continuous_scale='Reds')
        fig_scatter.add_hline(y=vib_thresh, line_dash="dash", line_color="orange", annotation_text="Danger Zone")
        st.plotly_chart(fig_scatter, use_container_width=True)

with t3:
    # 9. Boxplot & 7. Volume of failures
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Outlier Detection (Boxplot)")
        fig_box = px.box(df_sample, y=['vibration', 'Temperature'], template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)
    with col_d:
        st.subheader("Failure Volume per Month")
        df['Month'] = df['Timestamp'].dt.to_period('M').astype(str)
        fail_vol = df.groupby('Month')['Failure'].sum().reset_index()
        fig_bar = px.bar(fail_vol, x='Month', y='Failure', template="plotly_dark", color_discrete_sequence=['#ef4444'])
        st.plotly_chart(fig_bar, use_container_width=True)

# ==================================================
# SECTION 5 — ANOMALY DETECTION (Z-SCORE)
# ==================================================
if toggle_anomaly:
    st.markdown("---")
    st.header("🚨 Z-Score Anomaly Detection")
    
    # Calculate Z-Score
    df_sample['Z_Score'] = np.abs((df_sample['vibration'] - df_sample['vibration'].mean()) / df_sample['vibration'].std())
    anomalies = df_sample[df_sample['Z_Score'] > 2]
    
    st.warning(f"Detected {len(anomalies)} anomalies exceeding 2 Standard Deviations.")
    
    fig_anom = px.scatter(df_sample, x='Timestamp', y='vibration', color=df_sample['Z_Score'] > 2,
                          color_discrete_map={True: 'red', False: '#1f77b4'}, template="plotly_dark",
                          title="Vibration Anomalies (Red = >2 Std Dev)")
    st.plotly_chart(fig_anom, use_container_width=True)

# ==================================================
# SECTION 7 & 8 — ML PREDICTION & FORECASTING
# ==================================================
if toggle_predict:
    st.markdown("---")
    st.header
