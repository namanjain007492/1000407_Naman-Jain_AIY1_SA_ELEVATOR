import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import datetime
import time

# ==================================================
# 1. UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Elevator AI | Capstone Edition", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #38bdf8; }
    .demo-card { background-color: #0f291e; border-left: 5px solid #10b981; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. DATA ENGINEERING & MATHEMATICS
# ==================================================
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("Elevator predictive-maintenance-dataset.csv").dropna(subset=['vibration']).drop_duplicates()
    except Exception:
        st.error("Dataset not found. Please upload 'Elevator predictive-maintenance-dataset.csv'.")
        st.stop()
        
    # Synthesize production columns
    df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
    df['Temperature'] = 20 + (df['revolutions'] / 10) + np.random.normal(0, 1.5, len(df))
    df['Load'] = (df['revolutions'] / 8).astype(int).clip(0, 20) 
    df['Failure'] = np.where((df['vibration'] > 65) & (df['Temperature'] > 30), 1, 0)
    
    # Mathematical Volatility (FA-2 Requirement)
    df['Rolling_Mean'] = df['vibration'].rolling(window=20).mean().fillna(method='bfill')
    df['Rolling_Std'] = df['vibration'].rolling(window=20).std().fillna(method='bfill')
    df['Upper_Band'] = df['Rolling_Mean'] + (df['Rolling_Std'] * 2) # Bollinger Upper
    df['Lower_Band'] = df['Rolling_Mean'] - (df['Rolling_Std'] * 2) # Bollinger Lower
    
    # Floor Mapping
    conditions = [(df['x3'] < 0.5), (df['x3'] >= 0.5) & (df['x3'] < 0.75), (df['x3'] >= 0.75)]
    df['Floor'] = np.select(conditions, ['Ground (Food Court)', 'Floor 1 (Apparel)', 'Floor 2 (Cinema)'], default='Unknown')
    
    return df.sort_values('Timestamp').reset_index(drop=True)

df_raw = load_and_preprocess()
df = df_raw.iloc[::20].copy() if len(df_raw) > 10000 else df_raw.copy()

# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    
    nav = st.radio("Select Module", [
        "📖 1. Live Demo & Walkthrough",
        "📈 2. Mathematical Volatility (FA-2)",
        "🧠 3. ML Predictor & Radar",
        "🧊 4. 3D Time-Lapse Twin",
        "🤖 5. Gemini AI Chief"
    ])
    
    st.markdown("---")
    vib_thresh = st.slider("Vibration Alert (Hz)", 10.0, 100.0, 55.0)

# ==================================================
# MODULE 1: LIVE DEMO & WALKTHROUGH
# ==================================================
if nav == "📖 1. Live Demo & Walkthrough":
    st.title("Welcome to the Predictive Maintenance Demo")
    
    st.markdown("""
    <div class="demo-card">
        <h3>🚀 Interactive Demonstration Mode</h3>
        <p>This module simulates real-time data streaming from a shopping mall elevator. It processes the mathematics of mechanical volatility instantly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### How the Math Works:")
    st.write("We detect anomalies using the Z-Score formula to measure volatility against the mean:")
    st.latex(r"Z = \frac{x - \mu}{\sigma}")
    
    if st.button("▶️ Start Live Mall Simulation"):
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        metric_placeholder = st.empty()
        
        sim_data = df.head(200) # Take first 200 rows for demo
        for i in range(20, len(sim_data), 5):
            progress_bar.progress(int((i/len(sim_data))*100))
            current_slice = sim_data.iloc[:i]
            latest = current_slice.iloc[-1]
            
            with metric_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Floor", latest['Floor'])
                c2.metric("Live Vibration", f"{latest['vibration']:.1f} Hz")
                c3.metric("System Load", f"{latest['Load']} Pax")
                
            fig = px.line(current_slice, x='Timestamp', y='vibration', title="Live Sensor Feed", template="plotly_dark")
            fig.add_hline(y=vib_thresh, line_dash="dash", line_color="red")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.2)
            
        st.success("✅ Simulation Complete! Navigate to other modules to explore the data deeply.")

# ==================================================
# MODULE 2: MATHEMATICAL VOLATILITY (FA-2 Focus)
# ==================================================
elif nav == "📈 2. Mathematical Volatility (FA-2)":
    st.title("📈 Mathematical Volatility Analysis")
    st.write("Applying financial market math (Bollinger Bands) to mechanical sensor data to visualize 'Price Swing' style volatility.")
    
    [Image of Bollinger Bands technical analysis]

    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Upper_Band'], line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Lower_Band'], fill='tonexty', fillcolor='rgba(56, 189, 248, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Volatility Bounds (2 Std Dev)'))
    fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['vibration'], line=dict(color='#38bdf8', width=1), name='Raw Vibration'))
    fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Rolling_Mean'], line=dict(color='#f59e0b', width=2), name='Moving Average'))
    
    fig_bb.update_layout(template="plotly_dark", title="Vibration Volatility (Bollinger Bands Engine)", hovermode="x")
    st.plotly_chart(fig_bb, use_container_width=True)

    st.subheader("Statistical Anomaly Map")
    df['Z_Score'] = np.abs((df['vibration'] - df['vibration'].mean()) / df['vibration'].std())
    fig_scatter = px.scatter(df, x='Load', y='vibration', color=df['Z_Score'] > 2, color_discrete_map={True: '#ef4444', False: '#10b981'}, template="plotly_dark", title="Z-Score Outlier Detection (Red = >2 Std Dev)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==================================================
# MODULE 3: ML PREDICTOR & RADAR
# ==================================================
elif nav == "🧠 3. ML Predictor & Radar":
    st.title("🧠 ML Prediction & Sensor Harmony")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Logistic Regression Predictor")
        X = df[['Temperature', 'vibration', 'Load']]
        y = df['Failure']
        
        if y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
            prob = clf.predict_proba(X.iloc[[-1]])[0][1]
            st.metric("Imminent Failure Probability", f"{prob*100:.1f}%")
        else:
            st.success("Not enough historical failures to train ML model.")

    with c2:
        st.subheader("Raw Sensor Radar (x1 - x5)")
        st.write("Multivariate analysis of internal mechanical states.")
        latest_sensors = df[['x1', 'x2', 'x3', 'x4', 'x5']].iloc[-1]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=latest_sensors.values,
            theta=latest_sensors.index,
            fill='toself', marker_color='#10b981'
        ))
        fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0, latest_sensors.max()*1.2])))
        st.plotly_chart(fig_radar, use_container_width=True)

# ==================================================
# MODULE 4: 3D TIME-LAPSE TWIN
# ==================================================
elif nav == "🧊 4. 3D Time-Lapse Twin":
    st.title("🧊 Time-Lapse 3D Elevator Simulator")
    st.write("Scrub the timeline to watch the elevator move through the mall floors.")
    
    time_idx = st.slider("Scrub Timeline", 0, len(df)-1, 0)
    current = df.iloc[time_idx]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Floor", current['Floor'])
    col2.metric("Payload / Load", f"{current['Load']} Pax")
    col3.metric("Vibration Status", f"{current['vibration']:.1f} Hz", delta="Warning" if current['vibration'] > vib_thresh else "Normal", delta_color="inverse")

    fig_3d = go.Figure()
    
    # Draw Shaft Bounds
    fig_3d.add_trace(go.Mesh3d(x=[-2, 2, 2, -2, -2, 2, 2, -2], y=[-2, -2, 2, 2, -2, -2, 2, 2], z=[0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5], alphahull=1, opacity=0.05, color='white'))
    
    # Draw Floor Markers
    for z_val, f_name in zip([0.3, 0.6, 0.9], ['Ground', 'Floor 1', 'Floor 2']):
        fig_3d.add_trace(go.Surface(x=[[-2, 2], [-2, 2]], y=[[-2, -2], [2, 2]], z=[[z_val, z_val], [z_val, z_val]], opacity=0.2, colorscale='Greens', showscale=False, name=f_name))

    # Draw Elevator Car
    vib_color = 'red' if current['vibration'] > vib_thresh else 'orange' if current['vibration'] > vib_thresh - 15 else 'green'
    fig_3d.add_trace(go.Scatter3d(
        x=[current['x1']], y=[current['x2']], z=[current['x3']],
        mode='markers', marker=dict(size=35, color=vib_color, symbol='square'), name='Elevator Car'
    ))
    
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 1.5], zaxis_title='Vertical Height (x3)'), template="plotly_dark", height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================================================
# MODULE 5: GEMINI AI
# ==================================================
elif nav == "🤖 5. Gemini AI Chief":
    st.title("🤖 Gemini AI Control Center")
    
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 Add `GOOGLE_API_KEY` to Streamlit Cloud Secrets.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am the AI Maintenance Chief. Run the demo or ask me about our mathematical volatility metrics."}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("Ask about Z-Scores, Bollinger Bands, or repair plans..."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Data context: Avg Vib {df['vibration'].mean():.1f}Hz. Peak Load {df['Load'].max()} Pax. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
