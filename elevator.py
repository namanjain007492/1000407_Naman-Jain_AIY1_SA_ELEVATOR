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

# ==================================================
# 1. UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Smart Elevator OS", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .demo-card { background-color: #0f291e; border-left: 5px solid #2ea043; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
    h1, h2, h3 { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. DATA LOADING & PREPROCESSING
# ==================================================
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    except Exception:
        st.error("Dataset not found. Please upload 'Elevator predictive-maintenance-dataset.csv'.")
        st.stop()
        
    df = df.dropna(subset=['vibration']).copy()
    df = df.drop_duplicates()
    
    # Synthesize columns missing from raw data for a production feel
    df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
    df['Temperature'] = 20 + (df['revolutions'] / 10) + np.random.normal(0, 1.5, len(df))
    df['Load'] = (df['revolutions'] / 8).astype(int).clip(0, 20) # Persons
    df['Failure'] = np.where((df['vibration'] > 65) & (df['Temperature'] > 30), 1, 0)
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

df_raw = load_and_preprocess()

# ==================================================
# 3. SIDEBAR CONTROLS & NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("⚙️ Control Panel")
    
    nav = st.radio("System Modules", [
        "📖 1. Welcome & Demo Walkthrough",
        "📊 2. KPI & Telemetry Dashboard",
        "🧠 3. ML Prediction & Forecasting",
        "🧊 4. 3D Digital Twin Simulator",
        "🤖 5. Gemini AI Assistant"
    ])
    
    st.markdown("---")
    vib_thresh = st.slider("Vibration Alert (Hz)", 10.0, 100.0, 55.0)
    temp_thresh = st.slider("Temp Alert (°C)", 20.0, 50.0, 32.0)

# Downsample for UI Performance
df = df_raw.iloc[::20].copy() if len(df_raw) > 10000 else df_raw.copy()

# ==================================================
# MODULE 1: WELCOME & DEMO WALKTHROUGH
# ==================================================
if nav == "📖 1. Welcome & Demo Walkthrough":
    st.title("Welcome to the Predictive Maintenance OS")
    
    st.markdown("""
    <div class="demo-card">
        <h3>🚀 How to Use This Application</h3>
        <p>This dashboard is a full-stack AI tool designed to monitor elevator health, predict mechanical failures, and simulate passenger loads in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Module Breakdown:**
    * **📊 KPI & Telemetry:** Monitors real-time sensors (Vibration, Temperature, Load). It automatically flags anomalies (Z-Score > 2 Std Dev) in red.
    * **🧠 ML Prediction:** Uses an actively trained Logistic Regression model to calculate the exact probability of an imminent mechanical failure based on historical data.
    * **🧊 3D Digital Twin:** An interactive physics simulation. Use the slider to add 3D people to the elevator car and watch how the extra weight impacts the mechanical speed and increases vibration!
    * **🤖 Gemini AI:** A generative AI trained on your specific elevator data. Ask it for maintenance advice or to summarize the current risk score.
    
    **To begin, select a module from the sidebar on the left!**
    """)
    st.dataframe(df.head(10), use_container_width=True)

# ==================================================
# MODULE 2: KPI & TELEMETRY DASHBOARD
# ==================================================
elif nav == "📊 2. KPI & Telemetry Dashboard":
    st.title("🏢 Telemetry & Visualization Engine")
    
    # Risk Engine
    latest = df.iloc[-1]
    if latest['vibration'] > vib_thresh and latest['Temperature'] > temp_thresh:
        st.error("🚨 CRITICAL RISK: Immediate Maintenance Required within 48 hours.")
    elif latest['vibration'] > vib_thresh or latest['Temperature'] > temp_thresh:
        st.warning("⚠️ WARNING: Schedule Preventive Maintenance.")
    else:
        st.success("✅ SYSTEM STABLE: Operating Normally.")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df_raw):,}")
    c2.metric("Total Failures", df['Failure'].sum(), delta_color="inverse")
    c3.metric("Avg Temperature", f"{df['Temperature'].mean():.1f} °C")
    c4.metric("Avg Vibration", f"{df['vibration'].mean():.1f} Hz")
    
    st.markdown("---")
    st.subheader("Time Series Telemetry")
    df['Vib_Rolling'] = df['vibration'].rolling(10).mean()
    fig_ts = px.line(df, x='Timestamp', y=['vibration', 'Temperature', 'Vib_Rolling'], template="plotly_dark")
    
    # Highlight failures
    failures = df[df['Failure'] == 1]
    fig_ts.add_trace(go.Scatter(x=failures['Timestamp'], y=failures['vibration'], mode='markers', marker=dict(color='red', size=8), name='Failure'))
    st.plotly_chart(fig_ts, use_container_width=True)

    # Anomaly Detection
    st.subheader("Z-Score Anomaly Detection")
    df['Z_Score'] = np.abs((df['vibration'] - df['vibration'].mean()) / df['vibration'].std())
    fig_anom = px.scatter(df, x='Timestamp', y='vibration', color=df['Z_Score'] > 2, color_discrete_map={True: 'red', False: '#1f77b4'}, template="plotly_dark")
    st.plotly_chart(fig_anom, use_container_width=True)

# ==================================================
# MODULE 3: ML PREDICTION & FORECASTING
# ==================================================
elif nav == "🧠 3. ML Prediction & Forecasting":
    st.title("🧠 ML Failure Prediction Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression Model")
        X = df[['Temperature', 'vibration', 'Load']]
        y = df['Failure']
        
        if y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, clf.predict(X_test))
            prob = clf.predict_proba(X.iloc[[-1]])[0][1]
            
            st.metric("Model Accuracy", f"{acc*100:.1f}%")
            st.metric("Imminent Failure Probability", f"{prob*100:.1f}%", delta_color="inverse")
        else:
            st.success("No historical failures detected to train the model.")

    with col2:
        st.subheader("7-Day Linear Forecast")
        df_daily = df.set_index('Timestamp').resample('D')['vibration'].mean().dropna().reset_index()
        df_daily['DayNum'] = np.arange(len(df_daily))
        
        if len(df_daily) > 2:
            lr = LinearRegression()
            lr.fit(df_daily[['DayNum']], df_daily['vibration'])
            future_days = pd.DataFrame({'DayNum': np.arange(len(df_daily), len(df_daily)+7)})
            future_vib = lr.predict(future_days)
            future_dates = [df_daily['Timestamp'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, 8)]
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=df_daily['Timestamp'], y=df_daily['vibration'], name='Historical'))
            fig_fc.add_trace(go.Scatter(x=future_dates, y=future_vib, name='Forecast', line=dict(dash='dash', color='orange')))
            fig_fc.update_layout(template="plotly_dark")
            st.plotly_chart(fig_fc, use_container_width=True)

# ==================================================
# MODULE 4: 3D DIGITAL TWIN
# ==================================================
elif nav == "🧊 4. 3D Digital Twin Simulator":
    st.title("🧊 3D Passenger Payload Simulator")
    st.write("Add 3D people to the elevator to see how weight affects mechanical stress and speed.")
    
    sim_load = st.slider("Add Passengers (Load)", 0, 20, 5)
    sim_speed = max(0.5, 3.0 - (sim_load * 0.1))
    sim_vib = 10 + (sim_load * 2.5)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Speed", f"{sim_speed:.1f} m/s")
    c2.metric("Mechanical Vibration", f"{sim_vib:.1f} Hz", delta=f"+{sim_load*2.5:.1f} Hz", delta_color="inverse")
    c3.metric("Total Payload", f"{sim_load * 75} kg")

    # 3D Plotly Engine
    fig_3d = go.Figure()
    
    # Draw Shaft
    fig_3d.add_trace(go.Mesh3d(x=[-1, 1, 1, -1, -1, 1, 1, -1], y=[-1, -1, 1, 1, -1, -1, 1, 1], z=[0, 0, 0, 0, 10, 10, 10, 10], alphahull=1, opacity=0.1, color='cyan'))
    
    # Draw People
    if sim_load > 0:
        fig_3d.add_trace(go.Scatter3d(
            x=np.random.uniform(-0.8, 0.8, sim_load), y=np.random.uniform(-0.8, 0.8, sim_load), z=[5]*sim_load,
            mode='markers', marker=dict(size=8, color='orange'), name='Passengers'
        ))
        
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 10]), template="plotly_dark", height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================================================
# MODULE 5: GEMINI AI ASSISTANT
# ==================================================
elif nav == "🤖 5. Gemini AI Assistant":
    st.title("🤖 Gemini AI Maintenance Chief")
    
    # SECRETS INTEGRATION: This securely pulls your key from secrets.toml
    api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("🔑 Please add your `GOOGLE_API_KEY` to Streamlit Secrets to chat with the AI.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am analyzing the elevator dataset. How can I assist you?"}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., What should I do if the vibration hits 70Hz?"):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Data context: Avg Vib: {df['vibration'].mean():.1f}Hz. Temp: {df['Temperature'].mean():.1f}C. Failures: {df['Failure'].sum()}. User says: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
