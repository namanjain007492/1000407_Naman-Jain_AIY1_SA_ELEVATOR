import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import time

# Safe import for SciPy Calculus Integrals
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    cumulative_trapezoid = None

# ==================================================
# UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Elevator AI Operations", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #38bdf8; }
    .context-box { background-color: #1e293b; border-left: 5px solid #38bdf8; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
    .insight-box { background-color: #0f291e; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-bottom: 15px;}
    .report-box { background-color: #1e1e24; border-left: 5px solid #a855f7; padding: 20px; border-radius: 8px; margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# DATA PIPELINE & MATH ENGINE
# ==================================================
@st.cache_data
def load_and_clean_data():
    try:
        raw_df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    except Exception:
        st.error("⚠️ Dataset not found! Please ensure 'Elevator predictive-maintenance-dataset.csv' is in the exact same folder.")
        st.stop()
        
    clean_df = raw_df.dropna().drop_duplicates().copy()
    
    t = np.arange(len(clean_df))
    clean_df['Ideal_Resonance'] = 20 + 15 * np.sin(2 * np.pi * 0.05 * t) 
    
    if cumulative_trapezoid is not None:
        clean_df['Cumulative_Stress'] = cumulative_trapezoid(clean_df['vibration'], initial=0) * 0.1
    else:
        clean_df['Cumulative_Stress'] = np.cumsum(clean_df['vibration'].values) * 0.1 
        
    return raw_df, clean_df

raw_df, df = load_and_clean_data()
plot_df = df.iloc[::10].copy() if len(df) > 10000 else df.copy()

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    
    nav = st.radio("System Modules", [
        "📖 Project Overview",
        "🧹 Data Processing",
        "📊 Telemetry Visualizations",
        "🧊 Physics & Speed Simulator",
        "💡 Insights & GenAI",
        "📑 Engineering Report",
        "🚨 Anomaly Detection"
    ])
    
    st.markdown("---")
    st.header("🔑 AI Integration")
    
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            st.success("✅ API Key securely loaded from secrets!")
        else:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
    except Exception:
        api_key = st.text_input("Enter Gemini API Key:", type="password")

# ==================================================
# MODULE 7: ANOMALY DETECTION
# ==================================================
elif nav == "🚨 Anomaly Detection":
    st.title("🚨 Real-Time Anomaly Detection Engine")
    
    window = st.slider("Rolling Window Size:", 10, 200, 50)

    df['Rolling_Mean'] = df['vibration'].rolling(window=window).mean()
    df['Rolling_STD'] = df['vibration'].rolling(window=window).std()

    warning_threshold = df['Rolling_Mean'] + 2 * df['Rolling_STD']
    critical_threshold = df['Rolling_Mean'] + 3 * df['Rolling_STD']

    df['Risk_Level'] = np.where(
        df['vibration'] > critical_threshold, "Critical",
        np.where(df['vibration'] > warning_threshold, "Warning", "Normal")
    )

    warnings = (df['Risk_Level'] == "Warning").sum()
    criticals = (df['Risk_Level'] == "Critical").sum()

    health_score = 100 - ((criticals*2 + warnings)/len(df))*100
    health_score = max(0, health_score)

    c1, c2, c3 = st.columns(3)
    c1.metric("⚠ Warning Events", int(warnings))
    c2.metric("🔴 Critical Events", int(criticals))
    c3.metric("🩺 System Health", f"{health_score:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ID'], y=df['vibration'],
                             mode='lines', name='Vibration',
                             line=dict(color='#38bdf8')))
    
    fig.add_trace(go.Scatter(
        x=df[df['Risk_Level']=="Warning"]['ID'],
        y=df[df['Risk_Level']=="Warning"]['vibration'],
        mode='markers',
        marker=dict(color='orange', size=6),
        name='Warning'
    ))
    
    fig.add_trace(go.Scatter(
        x=df[df['Risk_Level']=="Critical"]['ID'],
        y=df[df['Risk_Level']=="Critical"]['vibration'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Critical'
    ))

    fig.update_layout(template="plotly_dark",
                      title="Live Risk Monitoring",
                      xaxis_title="Time (ID)",
                      yaxis_title="Vibration (Hz)")
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⏳ Remaining Useful Life Estimation")

    growth = df['vibration'].diff().mean()
    if growth > 0:
        limit = df['vibration'].mean() + 4 * df['vibration'].std()
        remaining = (limit - df['vibration'].iloc[-1]) / growth
        remaining = max(0, remaining)
        st.metric("Estimated Cycles Until Failure", f"{int(remaining)} cycles")
    else:
        st.metric("Estimated Cycles Until Failure", "Stable")

    st.success("Predictive Monitoring Active")
