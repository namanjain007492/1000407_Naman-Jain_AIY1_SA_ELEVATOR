import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.fft import fft, fftfreq
import google.generativeai as genai
import time

# ==================================================
# 1. UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Smart Elevator AI | V8", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #38bdf8; }
    .metric-card { background-color: #0f291e; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. DATA ENGINEERING & ML PREP
# ==================================================
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("Elevator predictive-maintenance-dataset.csv").dropna(subset=['vibration']).drop_duplicates()
    except Exception:
        st.error("⚠️ Dataset not found! Please ensure 'Elevator predictive-maintenance-dataset.csv' is in the exact same folder.")
        st.stop()
        
    df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
    df['Temperature'] = 20 + (df['revolutions'] / 10) + np.random.normal(0, 1.5, len(df))
    df['Load'] = (df['revolutions'] / 8).astype(int).clip(0, 20) 
    
    # IMPROVED ML DATA LOGIC: Calculate a weighted risk factor to determine realistic failures
    df['Risk_Factor'] = (df['vibration'] * 0.5) + (df['Temperature'] * 0.3) + (df['Load'] * 2.0)
    risk_threshold = df['Risk_Factor'].quantile(0.96) # Top 4% highest risk moments are mechanical failures
    df['Failure'] = (df['Risk_Factor'] > risk_threshold).astype(int)
    
    # Mathematics for AI (FA-2 Integrals & Resonance)
    t = np.arange(len(df))
    df['Ideal_Resonance'] = 20 + 15 * np.sin(2 * np.pi * 0.05 * t)
    df['Cumulative_Stress'] = np.cumsum(df['vibration']) * 0.1 
    df['Energy_Consumed_kWh'] = np.cumsum(df['revolutions'] * df['vibration'] * 0.001)
    
    # Volatility bounds
    df['Rolling_Mean'] = df['vibration'].rolling(window=20).mean().bfill()
    df['Rolling_Std'] = df['vibration'].rolling(window=20).std().bfill()
    df['Upper_Band'] = df['Rolling_Mean'] + (df['Rolling_Std'] * 2) 
    df['Lower_Band'] = df['Rolling_Mean'] - (df['Rolling_Std'] * 2) 
    df['Z_Score'] = np.abs((df['vibration'] - df['vibration'].mean()) / df['vibration'].std())
    
    conditions = [(df['x3'] < 0.5), (df['x3'] >= 0.5) & (df['x3'] < 0.75), (df['x3'] >= 0.75)]
    df['Floor'] = np.select(conditions, ['Ground', 'Floor 1', 'Floor 2'], default='Unknown')
    
    return df.sort_values('Timestamp').reset_index(drop=True)

df_raw = load_and_preprocess()
df = df_raw.iloc[::10].copy() if len(df_raw) > 5000 else df_raw.copy()

# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    
    nav = st.radio("Select Module", [
        "📖 1. Live Operational Stream",
        "📐 2. Mathematical Modeling",
        "🧊 3. 3D Speed & Physics Engine",
        "🧠 4. Advanced ML Training",
        "🤖 5. Gemini AI Terminal"
    ])
    
    st.markdown("---")
    vib_thresh = st.slider("Critical Vibration Limit (Hz)", 10.0, 100.0, 55.0)

# ==================================================
# MODULE 1: LIVE OPERATIONAL STREAM
# ==================================================
if nav == "📖 1. Live Operational Stream":
    st.title("Live Operational Telemetry")
    
    col_run, col_stop = st.columns([1, 5])
    start_sim = col_run.button("▶️ Start Stream")
    stop_sim = col_stop.button("⏹️ Stop Stream")
    
    if "sim_running" not in st.session_state: st.session_state.sim_running = False
    if start_sim: st.session_state.sim_running = True
    if stop_sim: st.session_state.sim_running = False

    chart_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    if st.session_state.sim_running:
        sim_data = df.head(300) 
        for i in range(20, len(sim_data), 2):
            if not st.session_state.sim_running: break 
            current_slice = sim_data.iloc[:i]
            latest = current_slice.iloc[-1]
            
            with metric_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Floor", latest['Floor'])
                c2.metric("Vibration", f"{latest['vibration']:.1f} Hz", delta="Alert" if latest['vibration'] > vib_thresh else "Normal", delta_color="inverse")
                c3.metric("Passenger Load", f"{latest['Load']} Pax")
                c4.metric("Z-Score Volatility", f"{latest['Z_Score']:.2f} σ")
                
            fig = px.line(current_slice, x='Timestamp', y=['vibration', 'Upper_Band', 'Lower_Band'], template="plotly_dark", title="Real-Time Vibration & Volatility Bounds")
            fig.add_hline(y=vib_thresh, line_dash="dash", line_color="red")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.05) 
    else:
        st.info("Click 'Start Stream' to begin the real-time simulation.")

# ==================================================
# MODULE 2: MATHEMATICAL MODELING
# ==================================================
elif nav == "📐 2. Mathematical Modeling":
    st.title("📐 Mathematical Diagnostics")
    
    t1, t2 = st.tabs(["Sine Resonance Integrals", "Bollinger Swings"])
    
    with t1:
        st.write("Using Numerical Integration to calculate Cumulative Mechanical Stress (Area Under the Curve).")
        fig_int = px.area(df, x='Timestamp', y='Cumulative_Stress', template="plotly_dark", color_discrete_sequence=['#8b5cf6'])
        st.plotly_chart(fig_int, use_container_width=True)

    with t2:
        st.write("Applying financial market math to visualize mechanical volatility.")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Upper_Band'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Lower_Band'], fill='tonexty', fillcolor='rgba(56, 189, 248, 0.15)', line=dict(color='rgba(0,0,0,0)'), name='Volatility Bounds'))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['vibration'], line=dict(color='#38bdf8', width=1), name='Raw Vibration'))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Rolling_Mean'], line=dict(color='#f59e0b', width=2), name='Moving Avg'))
        fig_bb.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bb, use_container_width=True)

# ==================================================
# MODULE 3: 3D SPEED & PHYSICS ENGINE
# ==================================================
elif nav == "🧊 3. 3D Speed & Physics Engine":
    st.title("🧊 Interactive Physics & Speed Engine")
    st.write("See exactly how passenger weight impacts the mechanical speed of the elevator.")
    
    c1, c2 = st.columns(2)
    start_floor = c1.selectbox("Start Floor:", ["Ground", "Floor 1", "Floor 2"], index=0)
    target_floor = c2.selectbox("Destination Floor:", ["Ground", "Floor 1", "Floor 2"], index=2)
    pax_load = st.slider("Board Passengers (Weight):", 0, 20, 8)
    
    # Physics Calculations
    z_map = {"Ground": 0.0, "Floor 1": 4.0, "Floor 2": 8.0} # Representing meters in height
    z_start = z_map[start_floor]
    z_end = z_map[target_floor]
    distance = abs(z_end - z_start)
    
    base_speed = 2.5 # Ideal m/s empty
    speed_penalty = pax_load * 0.08 # Lose 0.08 m/s per passenger due to motor strain
    actual_speed = max(0.5, base_speed - speed_penalty)
    travel_time = distance / actual_speed if distance > 0 else 0
    
    sim_vib = 15 + (pax_load * 2.2) 
    vib_color = 'red' if sim_vib > vib_thresh else 'orange' if sim_vib > vib_thresh - 15 else '#10b981'

    st.markdown("""<div class="metric-card"><h4>⏱️ Live Physics Telemetry</h4></div>""", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance to Travel", f"{distance} meters")
    m2.metric("Actual Speed", f"{actual_speed:.2f} m/s", delta=f"-{speed_penalty:.2f} m/s (Load Penalty)", delta_color="inverse")
    m3.metric("Est. Travel Time", f"{travel_time:.1f} seconds")
    m4.metric("Mechanical Strain", f"{sim_vib:.1f} Hz")

    fig_3d = go.Figure()
    # Shaft & Floors
    fig_3d.add_trace(go.Mesh3d(x=[-2, 2, 2, -2, -2, 2, 2, -2], y=[-2, -2, 2, 2, -2, -2, 2, 2], z=[0, 0, 0, 0, 8, 8, 8, 8], alphahull=1, opacity=0.05, color='white'))
    for z_val, f_name in zip([0.0, 4.0, 8.0], ['Ground', 'Floor 1', 'Floor 2']):
        plane_color = 'cyan' if z_val == z_end else 'grey'
        fig_3d.add_trace(go.Surface(x=[[-2, 2], [-2, 2]], y=[[-2, -2], [2, 2]], z=[[z_val, z_val], [z_val, z_val]], opacity=0.3 if z_val == z_end else 0.1, colorscale=[[0, plane_color], [1, plane_color]], showscale=False, name=f_name))

    # Elevator Car
    fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[z_end], mode='markers', marker=dict(size=40, color=vib_color, symbol='square'), name='Elevator Car'))
    
    if pax_load > 0:
        fig_3d.add_trace(go.Scatter3d(x=np.random.uniform(-0.5, 0.5, pax_load), y=np.random.uniform(-0.5, 0.5, pax_load), z=[z_end]*pax_load, mode='markers', marker=dict(size=6, color='black'), name='Passengers'))
    
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 9], zaxis_title='Height (Meters)'), template="plotly_dark", height=500, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================================================
# MODULE 4: ADVANCED ML TRAINING
# ==================================================
elif nav == "🧠 4. Advanced ML Training":
    st.title("🧠 AI Model Training & Feature Importance")
    
    st.write("The dataset has been balanced. The algorithm now learns from the mechanical relationship between Temperature, Load, and Cumulative Stress.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest Classifier")
        X = df[['Temperature', 'vibration', 'Load', 'Cumulative_Stress']]
        y = df['Failure']
        
        if y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Using Random Forest to get Feature Importances
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train, y_train)
            
            acc = accuracy_score(y_test, clf.predict(X_test))
            prob = clf.predict_proba(X.iloc[[-1]])[0][1]
            st.metric("Model Testing Accuracy", f"{acc*100:.1f}%")
            st.metric("Current Imminent Failure Probability", f"{prob*100:.1f}%")
            
            st.write("### What causes failures?")
            # Feature Importance Chart
            importances = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', template="plotly_dark", color='Importance', color_continuous_scale='Reds')
            st.plotly_chart(fig_imp, use_container_width=True)
            
        else:
            st.error("Model failure: No breakdown events found in the dataset to train on.")

    with col2:
        st.subheader("7-Day Predictive Trend")
        df_daily = df.set_index('Timestamp').resample('D')['vibration'].mean().dropna().reset_index()
        df_daily['DayNum'] = np.arange(len(df_daily))
        if len(df_daily) > 2:
            lr = LinearRegression().fit(df_daily[['DayNum']], df_daily['vibration'])
            future_vib = lr.predict(pd.DataFrame({'DayNum': np.arange(len(df_daily), len(df_daily)+7)}))
            fig_fc = px.line(x=range(1, 8), y=future_vib, template="plotly_dark", labels={'x':'Days Forward', 'y':'Predicted Vibration (Hz)'})
            st.plotly_chart(fig_fc, use_container_width=True)

# ==================================================
# MODULE 5: GEMINI AI
# ==================================================
elif nav == "🤖 5. Gemini AI Terminal":
    st.title("🤖 Gemini Maintenance AI")
    
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ Please add your `GOOGLE_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am the AI Architect. Ask me about the Random Forest model or the physics speed penalties!"}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., Which feature causes the most elevator failures?"):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Context: Avg Vib {df['vibration'].mean():.1f}Hz. Peak Load {df['Load'].max()} Pax. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
