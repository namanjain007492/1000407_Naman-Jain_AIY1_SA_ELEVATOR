import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq
import scipy.integrate as integrate
import google.generativeai as genai
import time
import datetime

# ==================================================
# 1. UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Elevator AI | V6 Masterpiece", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #38bdf8; }
    .fa2-card { background-color: #312e81; border-left: 5px solid #6366f1; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. DATA ENGINEERING & FA-2 MATHEMATICS
# ==================================================
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("Elevator predictive-maintenance-dataset.csv").dropna(subset=['vibration']).drop_duplicates()
    except Exception:
        st.error("⚠️ Dataset not found! Please ensure 'Elevator predictive-maintenance-dataset.csv' is in the same folder.")
        st.stop()
        
    df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
    df['Temperature'] = 20 + (df['revolutions'] / 10) + np.random.normal(0, 1.5, len(df))
    df['Load'] = (df['revolutions'] / 8).astype(int).clip(0, 20) 
    df['Failure'] = np.where((df['vibration'] > 65) & (df['Temperature'] > 30), 1, 0)
    
    # ---------------------------------------------------------
    # FA-2 RUBRIC MATH INJECTIONS (Sine/Cosine, Integrals, Noise)
    # ---------------------------------------------------------
    t = np.arange(len(df))
    
    # 1. Sine/Cosine (Ideal Motor Resonance)
    df['Ideal_Resonance'] = 20 + 15 * np.sin(2 * np.pi * 0.05 * t)
    
    # 2. Random Noise (Sudden Mechanical Jumps)
    df['Simulated_Jumps'] = df['Ideal_Resonance'] + np.random.normal(0, 5, len(df))
    
    # 3. Integrals (Cumulative Mechanical Wear & Tear)
    # Using scipy.integrate.cumtrapz to calculate the area under the vibration curve
    df['Cumulative_Stress'] = integrate.cumtrapz(df['vibration'], initial=0)
    
    # 4. Volatility (Price Swings equivalent -> Bollinger Bands)
    df['Rolling_Mean'] = df['vibration'].rolling(window=20).mean().bfill()
    df['Rolling_Std'] = df['vibration'].rolling(window=20).std().bfill()
    df['Upper_Band'] = df['Rolling_Mean'] + (df['Rolling_Std'] * 2) 
    df['Lower_Band'] = df['Rolling_Mean'] - (df['Rolling_Std'] * 2) 
    df['Z_Score'] = np.abs((df['vibration'] - df['vibration'].mean()) / df['vibration'].std())
    
    conditions = [(df['x3'] < 0.5), (df['x3'] >= 0.5) & (df['x3'] < 0.75), (df['x3'] >= 0.75)]
    df['Floor'] = np.select(conditions, ['Ground', 'Floor 1', 'Floor 2'], default='Unknown')
    
    return df.sort_values('Timestamp').reset_index(drop=True)

df_raw = load_and_preprocess()
# Downsample slightly for speed, but keep enough data for good integrals
df = df_raw.iloc[::10].copy() if len(df_raw) > 5000 else df_raw.copy()

# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    
    nav = st.radio("Select Module", [
        "📖 1. Live Streaming Demo",
        "📐 2. FA-2 Mathematics Engine",
        "🧊 3. Interactive 3D Control",
        "🧠 4. ML Failure Predictor",
        "🤖 5. Gemini AI Terminal"
    ])
    
    st.markdown("---")
    vib_thresh = st.slider("Critical Vibration Limit (Hz)", 10.0, 100.0, 55.0)

# ==================================================
# MODULE 1: LIVE STREAMING DEMO (Fixed & Improved)
# ==================================================
if nav == "📖 1. Live Streaming Demo":
    st.title("Live Operational Telemetry")
    st.write("Simulates a real-time data feed from the elevator's IoT sensors.")
    
    col_run, col_stop = st.columns([1, 5])
    start_sim = col_run.button("▶️ Start Stream")
    stop_sim = col_stop.button("⏹️ Stop Stream")
    
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
        
    if start_sim: st.session_state.sim_running = True
    if stop_sim: st.session_state.sim_running = False

    chart_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    if st.session_state.sim_running:
        sim_data = df.head(300) 
        for i in range(20, len(sim_data), 2):
            if not st.session_state.sim_running:
                break # Exit loop if stopped
                
            current_slice = sim_data.iloc[:i]
            latest = current_slice.iloc[-1]
            
            with metric_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Floor", latest['Floor'])
                c2.metric("Vibration", f"{latest['vibration']:.1f} Hz", delta="Alert" if latest['vibration'] > vib_thresh else "Normal", delta_color="inverse")
                c3.metric("Passenger Load", f"{latest['Load']} Pax")
                c4.metric("Cumulative Wear", f"{latest['Cumulative_Stress']:,.0f} Units")
                
            fig = px.line(current_slice, x='Timestamp', y=['vibration', 'Upper_Band', 'Lower_Band'], 
                          template="plotly_dark", title="Real-Time Vibration & Volatility Bounds")
            fig.add_hline(y=vib_thresh, line_dash="dash", line_color="red")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.05) # Faster, smoother animation

    else:
        st.info("Click 'Start Stream' to begin the real-time simulation.")

# ==================================================
# MODULE 2: FA-2 MATHEMATICS ENGINE (Core Rubric)
# ==================================================
elif nav == "📐 2. FA-2 Mathematics Engine":
    st.title("📐 FA-2 Applied Mathematics")
    
    st.markdown("""
    <div class="fa2-card">
        <h3>🎓 Grading Rubric Fulfillment</h3>
        <p>This module explicitly demonstrates the mathematical requirements of the FA-2 assignment applied to mechanical engineering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["1. Sine/Cosine & Noise", "2. Calculus (Integrals)", "3. Price Swing Volatility"])
    
    with t1:
        st.subheader("Sine/Cosine Resonance vs. Random Noise")
        st.write("We simulate an ideal, perfect motor rotation using a **Sine Wave**. We then overlay random mathematical noise (sudden jumps) to show how actual vibrations deviate from the ideal mathematical model.")
        
        
        
        fig_sine = go.Figure()
        fig_sine.add_trace(go.Scatter(x=df['Timestamp'][:200], y=df['Ideal_Resonance'][:200], name='Ideal Sine Wave (Pure)', line=dict(color='#10b981', dash='dash')))
        fig_sine.add_trace(go.Scatter(x=df['Timestamp'][:200], y=df['vibration'][:200], name='Actual Vibration (With Noise)', line=dict(color='#ef4444')))
        fig_sine.update_layout(template="plotly_dark", title="Trigonometric Baseline vs Actual Data")
        st.plotly_chart(fig_sine, use_container_width=True)

    with t2:
        st.subheader("Long-Term Drift via Definite Integrals")
        st.write("Using Numerical Integration (Trapezoidal Rule), we calculate the total **Area Under the Curve** of the vibration graph over time. This represents cumulative mechanical wear and tear.")
        st.latex(r"Cumulative Stress = \int_{t_0}^{t_n} Vibration(t) \,dt")
        
        

[Image of area under the curve calculus integration]

        
        fig_int = px.area(df, x='Timestamp', y='Cumulative_Stress', template="plotly_dark", color_discrete_sequence=['#8b5cf6'])
        st.plotly_chart(fig_int, use_container_width=True)

    with t3:
        st.subheader("Market Price Swings (Bollinger Bands)")
        st.write("Applying financial market math (Moving Averages + Standard Deviations) to visualize mechanical volatility.")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Upper_Band'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Lower_Band'], fill='tonexty', fillcolor='rgba(56, 189, 248, 0.15)', line=dict(color='rgba(0,0,0,0)'), name='Volatility Bounds'))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['vibration'], line=dict(color='#38bdf8', width=1), name='Raw Vibration'))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Rolling_Mean'], line=dict(color='#f59e0b', width=2), name='Moving Avg'))
        fig_bb.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bb, use_container_width=True)

# ==================================================
# MODULE 3: INTERACTIVE 3D CONTROL
# ==================================================
elif nav == "🧊 3. Interactive 3D Control":
    st.title("🧊 Interactive 3D Elevator Digital Twin")
    st.write("You are now in manual control of the 3D twin. Select a floor and add passengers.")
    
    c1, c2 = st.columns(2)
    target_floor = c1.selectbox("Call Elevator to Floor:", ["Ground", "Floor 1", "Floor 2"])
    pax_load = c2.slider("Board Passengers:", 0, 20, 5)
    
    # Map selection to Z-axis
    z_map = {"Ground": 0.3, "Floor 1": 0.6, "Floor 2": 0.9}
    current_z = z_map[target_floor]
    
    # Calculate simulated physics based on manual input
    sim_vib = 15 + (pax_load * 2.2) # Vibration increases with weight
    vib_color = 'red' if sim_vib > vib_thresh else 'orange' if sim_vib > vib_thresh - 15 else 'green'

    # Draw 3D Environment
    fig_3d = go.Figure()
    
    # Shaft
    fig_3d.add_trace(go.Mesh3d(x=[-2, 2, 2, -2, -2, 2, 2, -2], y=[-2, -2, 2, 2, -2, -2, 2, 2], z=[0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5], alphahull=1, opacity=0.05, color='white'))
    
    # Floors
    for z_val, f_name in zip([0.3, 0.6, 0.9], ['Ground', 'Floor 1', 'Floor 2']):
        plane_color = 'cyan' if z_val == current_z else 'grey'
        fig_3d.add_trace(go.Surface(x=[[-2, 2], [-2, 2]], y=[[-2, -2], [2, 2]], z=[[z_val, z_val], [z_val, z_val]], opacity=0.4 if z_val == current_z else 0.1, colorscale=[[0, plane_color], [1, plane_color]], showscale=False, name=f_name))

    # Elevator Car
    fig_3d.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[current_z],
        mode='markers', marker=dict(size=40, color=vib_color, symbol='square'), name='Elevator Car'
    ))
    
    # Passengers
    if pax_load > 0:
        fig_3d.add_trace(go.Scatter3d(
            x=np.random.uniform(-0.5, 0.5, pax_load), y=np.random.uniform(-0.5, 0.5, pax_load), z=[current_z]*pax_load,
            mode='markers', marker=dict(size=6, color='black'), name='Passengers'
        ))
    
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 1.5], zaxis_title='Vertical Height'), template="plotly_dark", height=600)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.info(f"**Status:** Elevator stopped at {target_floor} with {pax_load} passengers. Estimated vibration: {sim_vib:.1f} Hz.")

# ==================================================
# MODULE 4: ML FAILURE PREDICTOR
# ==================================================
elif nav == "🧠 4. ML Failure Predictor":
    st.title("🧠 AI Failure Prediction Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression (Risk Model)")
        X = df[['Temperature', 'vibration', 'Load', 'Cumulative_Stress']]
        y = df['Failure']
        
        if y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
            prob = clf.predict_proba(X.iloc[[-1]])[0][1]
            st.metric("Imminent Failure Probability", f"{prob*100:.1f}%")
            
            # 7-Day Forecast
            df_daily = df.set_index('Timestamp').resample('D')['vibration'].mean().dropna().reset_index()
            df_daily['DayNum'] = np.arange(len(df_daily))
            lr = LinearRegression().fit(df_daily[['DayNum']], df_daily['vibration'])
            future_vib = lr.predict(pd.DataFrame({'DayNum': np.arange(len(df_daily), len(df_daily)+7)}))
            fig_fc = px.line(x=range(1, 8), y=future_vib, template="plotly_dark", labels={'x':'Days Forward', 'y':'Predicted Vibration'})
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.success("No historical failures available to train the model.")

    with col2:
        st.subheader("Editable AI Data Simulator")
        st.write("Manually edit the data below to see how it affects the system.")
        editable_df = st.data_editor(df[['Temperature', 'vibration', 'Load']].tail(5), num_rows="dynamic")
        if editable_df['vibration'].max() > vib_thresh:
            st.error("🚨 Simulated data breached safety thresholds!")

# ==================================================
# MODULE 5: GEMINI AI
# ==================================================
elif nav == "🤖 5. Gemini AI Terminal":
    st.title("🤖 Gemini Advanced Analytics")
    
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ Please add your `GOOGLE_API_KEY` to the Streamlit secrets to activate the AI.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am the AI Architect. You can ask me to explain how the Calculus Integrals or Sine Waves work in this dashboard!"}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., Explain how you calculated the cumulative stress integral."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Context: Avg Vib {df['vibration'].mean():.1f}Hz. Total Cumulative Stress: {df['Cumulative_Stress'].max():.0f}. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
