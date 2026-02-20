import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from scipy import stats
import time

# ==========================================
# 1. ENTERPRISE UI & CSS
# ==========================================
st.set_page_config(page_title="Elevator OS | 3D Animated", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #f8fafc; }
    div[data-testid="stMetric"] {
        background-color: #0f172a; border: 1px solid #1e293b;
        border-radius: 10px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.8);
    }
    .tutorial-box { background-color: #164e63; padding: 20px; border-radius: 10px; border-left: 5px solid #06b6d4; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE & MATHEMATICAL MODELLING
# ==========================================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv").dropna(subset=['vibration'])
    
    # Mathematical Modelling: Volatility & Stability
    df['health_score'] = (100 - (df['vibration'] * 0.7 + (df['humidity'] - 70) * 1.5)).clip(0, 100)
    df['vibration_rolling'] = df['vibration'].rolling(window=50).mean().fillna(df['vibration'])
    df['volatility'] = df['vibration'].pct_change().fillna(0).abs() # Price-swing style math for sensor noise
    
    # Z-Score Anomaly Detection
    df['z_score'] = np.abs(stats.zscore(df['vibration']))
    df['is_anomaly'] = df['z_score'] > 3
    
    return df

try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"Dataset missing! Upload the CSV. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.title("🏢 Smart Elevator OS")
    st.markdown("---")
    
    nav = st.radio("Select Module", [
        "📖 How to Use This App (Demo)",
        "🎛️ Live Command Center", 
        "🎬 3D Animated Elevator", 
        "📈 Volatility & Math Lab", 
        "🤖 Gemini AI Assistant"
    ])
    
    st.markdown("---")
    st.subheader("Global Parameters")
    vib_alert = st.slider("Critical Vibration (Hz)", 10, 100, 40)
    sim_speed = st.slider("Simulation Speed", 0.1, 2.0, 1.0)

# ==========================================
# MODULE 1: APP DEMO & TUTORIAL (NEW)
# ==========================================
if nav == "📖 How to Use This App (Demo)":
    st.title("Welcome to the Predictive Maintenance OS")
    
    st.markdown("""
    <div class="tutorial-box">
        <h3>🚀 Quick Start Guide</h3>
        <p>This dashboard uses advanced mathematical functions and machine learning to detect patterns of stability and volatility in elevator sensor data. Here is how to use it:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1. Command Center")
    st.write("View the real-time health of the elevator. The line charts use rolling averages to smooth out sudden jumps (random noise) from the actual long-term mechanical drift.")
    
    st.subheader("2. 3D Animated Elevator")
    st.write("Watch the digital twin move! This module animates the `X3` (vertical) sensor data over time, showing you exactly how the elevator car moves up and down the shaft while changing color based on vibration intensity.")
    
    st.subheader("3. Volatility & Math Lab")
    st.write("Explore the statistical relationship between environmental factors (like humidity) and mechanical wear. You can download the anomaly reports here.")
    
    st.subheader("4. AI Assistant")
    st.write("Ask the built-in Google Gemini AI questions about the data. (Ensure your API key is in the Streamlit Secrets!).")
    
    if st.button("Start Exploring Data Now"):
        st.success("Navigate using the sidebar on the left!")

# ==========================================
# MODULE 2: COMMAND CENTER & STREAMING
# ==========================================
elif nav == "🎛️ Live Command Center":
    st.title("🎛️ Live Operational Dashboard")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System Health", f"{df['health_score'].mean():.1f}%")
    c2.metric("Volatility Index", f"{df['volatility'].mean()*100:.2f}%")
    c3.metric("Anomalies", f"{df['is_anomaly'].sum()}")
    c4.metric("Avg Vibration", f"{df['vibration'].mean():.1f} Hz")
    
    st.subheader("Live Sensor Pulse (With Volatility High/Low)")
    
    # Real-time simulation toggle
    if st.checkbox("🟢 Enable Real-Time Data Streaming Simulation"):
        placeholder = st.empty()
        for i in range(100, 1000, 10):
            with placeholder.container():
                stream_df = df.iloc[i-100:i]
                fig = px.line(stream_df, x='ID', y=['vibration', 'vibration_rolling'], template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1 / sim_speed)
    else:
        fig_pulse = px.line(df.iloc[::25], x='ID', y=['vibration', 'vibration_rolling'], template="plotly_dark")
        fig_pulse.add_hline(y=vib_alert, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(fig_pulse, use_container_width=True)

# ==========================================
# MODULE 3: 3D ANIMATED ELEVATOR (NEW)
# ==========================================
elif nav == "🎬 3D Animated Elevator":
    st.title("🎬 Animated 3D Digital Twin")
    st.markdown("This simulates the elevator traveling up and down. The sphere represents the elevator car. **Press the 'Play' button on the timeline below to watch it move.**")
    
    
    
    # Create an animation sequence
    # We take a continuous slice of 150 points to show a smooth up/down trip
    anim_df = df.iloc[1000:1150].copy()
    anim_df['TimeStep'] = range(len(anim_df))  # Create frame sequence
    
    fig_anim = px.scatter_3d(
        anim_df, 
        x='x1', y='x2', z='x3',
        animation_frame='TimeStep', # THIS CREATES THE ANIMATION
        color='vibration',
        size='vibration',
        size_max=30,
        range_x=[df['x1'].min(), df['x1'].max()],
        range_y=[df['x2'].min(), df['x2'].max()],
        range_z=[df['x3'].min(), df['x3'].max()],
        color_continuous_scale='Turbo',
        template="plotly_dark",
        height=750
    )
    
    # Add the "Shaft" wireframe for visual context
    fig_anim.add_trace(go.Scatter3d(
        x=[120, 120, 120, 120], y=[-30, -30, -30, -30], 
        z=[df['x3'].min(), df['x3'].max()],
        mode='lines', line=dict(color='cyan', width=2), name="Guide Rail"
    ))
    
    fig_anim.update_layout(scene=dict(
        xaxis_title='Lateral Sway (X1)', 
        yaxis_title='Depth (X2)', 
        zaxis_title='Vertical Position (X3) ↕️'
    ))
    
    st.plotly_chart(fig_anim, use_container_width=True)
    st.info("💡 **Observation:** As the animation plays, watch how the color changes (indicating vibration intensity) as the vertical position (Z-axis) changes.")

# ==========================================
# MODULE 4: VOLATILITY & MATH LAB
# ==========================================
elif nav == "📈 Volatility & Math Lab":
    st.title("📈 Statistical Volatility Lab")
    
    tab1, tab2 = st.tabs(["Volatility Heatmap", "Data Export Engine"])
    
    with tab1:
        st.write("Matrix showing patterns of stability and volatility across all variables.")
        corr = df[['revolutions', 'humidity', 'vibration', 'volatility', 'x1', 'x3']].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with tab2:
        st.subheader("Export Anomaly Data")
        st.write("Filter and download data rows where mathematical thresholds were breached.")
        anomaly_df = df[df['vibration'] > vib_alert]
        st.dataframe(anomaly_df.head(50), use_container_width=True)
        
        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Critical Logs (CSV)", data=csv, file_name='volatility_spikes.csv', mime='text/csv')

# ==========================================
# MODULE 5: AI ASSISTANT
# ==========================================
elif nav == "🤖 Gemini AI Assistant":
    st.title("🤖 Gemini Enterprise Assistant")
    
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🔑 API Key Missing! Please add `GOOGLE_API_KEY` to Streamlit Secrets.")
    else:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am connected to the dataset. Ask me to calculate ROI, analyze volatility, or suggest repairs."}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., What causes high volatility in the Z-axis?"):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Data context: Avg Vib {df['vibration'].mean():.1f}Hz, Volatility {df['volatility'].mean()*100:.2f}%. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
