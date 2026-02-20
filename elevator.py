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
st.set_page_config(page_title="Elevator AI Pro", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #38bdf8; }
    .highlight-card { background-color: #0f291e; border-left: 5px solid #10b981; padding: 15px; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. DATA ENGINEERING & PREPROCESSING
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
    
    # NEW: Volatility Math (Rolling Standard Deviation)
    df['Volatility'] = df['vibration'].rolling(window=12).std().fillna(0)
    df['State'] = np.where(df['Volatility'] > df['Volatility'].quantile(0.85), 'Volatile', 'Stable')
    
    return df.sort_values('Timestamp').reset_index(drop=True)

df_raw = load_and_preprocess()

# Downsample for UI Performance but keep enough for good math
df = df_raw.iloc[::20].copy() if len(df_raw) > 10000 else df_raw.copy()

# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    
    nav = st.radio("Select Module", [
        "📊 1. Volatility & Telemetry",
        "🧠 2. Predictive ML & ROI",
        "🧊 3. Time-Lapse 3D Twin",
        "🤖 4. Gemini AI Chief"
    ])
    
    st.markdown("---")
    st.subheader("Global Settings")
    vib_thresh = st.slider("Vibration Alert (Hz)", 10.0, 100.0, 55.0)
    temp_thresh = st.slider("Temp Alert (°C)", 20.0, 50.0, 32.0)

# ==================================================
# MODULE 1: VOLATILITY & TELEMETRY DASHBOARD
# ==================================================
if nav == "📊 1. Volatility & Telemetry":
    st.title("📊 System Telemetry & Volatility Analysis")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Vibration", f"{df['vibration'].mean():.1f} Hz")
    c2.metric("Peak Volatility", f"{df['Volatility'].max():.2f} StdDev", delta="High Risk", delta_color="inverse")
    c3.metric("System Load", f"{df['Load'].mean():.1f} Pax")
    c4.metric("Failures Prevented", df['Failure'].sum())

    t1, t2 = st.tabs(["Stable vs Volatile Periods", "High/Low Comparisons"])
    
    with t1:
        st.subheader("Vibration Swings (Stability vs. Volatility)")
        st.write("Mathematical tracking of random noise and mechanical drift.")
        fig_vol = px.scatter(df, x='Timestamp', y='vibration', color='State', 
                             color_discrete_map={'Stable': '#10b981', 'Volatile': '#ef4444'},
                             template="plotly_dark", opacity=0.7)
        fig_vol.add_hline(y=vib_thresh, line_dash="dash", line_color="orange", annotation_text="Danger Zone")
        st.plotly_chart(fig_vol, use_container_width=True)

    with t2:
        st.subheader("Daily High vs Low Extremes")
        df_daily = df.set_index('Timestamp').resample('D')['vibration'].agg(['max', 'min']).dropna().reset_index()
        fig_hl = go.Figure()
        fig_hl.add_trace(go.Bar(x=df_daily['Timestamp'], y=df_daily['max'], name='Daily High (Max Stress)', marker_color='#ef4444'))
        fig_hl.add_trace(go.Bar(x=df_daily['Timestamp'], y=df_daily['min'], name='Daily Low (Baseline)', marker_color='#3b82f6'))
        fig_hl.update_layout(template="plotly_dark", barmode='group')
        st.plotly_chart(fig_hl, use_container_width=True)

# ==================================================
# MODULE 2: PREDICTIVE ML & ROI
# ==================================================
elif nav == "🧠 2. Predictive ML & ROI":
    st.title("🧠 Machine Learning & Financial ROI")
    
    col_ml, col_roi = st.columns(2)
    
    with col_ml:
        st.subheader("Logistic Regression (Failure Predictor)")
        X = df[['Temperature', 'vibration', 'Load', 'Volatility']]
        y = df['Failure']
        
        if y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_train, y_train)
            
            prob = clf.predict_proba(X.iloc[[-1]])[0][1]
            st.metric("Imminent Failure Probability", f"{prob*100:.1f}%")
            
            # Forecast
            st.write("**7-Day Forecast (Linear Math)**")
            df_daily = df.set_index('Timestamp').resample('D')['vibration'].mean().dropna().reset_index()
            df_daily['DayNum'] = np.arange(len(df_daily))
            lr = LinearRegression().fit(df_daily[['DayNum']], df_daily['vibration'])
            future_vib = lr.predict(pd.DataFrame({'DayNum': np.arange(len(df_daily), len(df_daily)+7)}))
            
            fig_fc = px.line(x=range(1, 8), y=future_vib, template="plotly_dark", markers=True, title="Expected Vibration (Next 7 Days)")
            fig_fc.update_traces(line_color='#f59e0b')
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.success("No historical failures detected to train the ML model.")

    with col_roi:
        st.subheader("💰 ROI & Cost Savings Calculator")
        st.markdown("""
        <div class="highlight-card">
        Calculates estimated savings by predicting failures before a catastrophic motor burnout occurs.
        </div>
        """, unsafe_allow_html=True)
        
        cost_catastrophic = st.number_input("Cost of Catastrophic Failure ($)", value=15000)
        cost_preventative = st.number_input("Cost of Preventative Fix ($)", value=800)
        
        failures_caught = df['Failure'].sum()
        money_saved = (cost_catastrophic - cost_preventative) * failures_caught
        
        st.metric("Total Money Saved by AI", f"${money_saved:,.2f}", delta="Positive ROI")
        st.write(f"*Based on successfully intercepting {failures_caught} critical events.*")

# ==================================================
# MODULE 3: TIME-LAPSE 3D TWIN
# ==================================================
elif nav == "🧊 3. Time-Lapse 3D Twin":
    st.title("🧊 Time-Lapse 3D Elevator Simulator")
    st.write("Use the timeline slider to scrub through the historical data and watch the elevator car move physically based on the `x3` (Vertical) sensor.")
    
    

    # Select a time slice using a slider
    max_idx = len(df) - 1
    time_idx = st.slider("Scrub Timeline", 0, max_idx, max_idx // 2)
    
    current_state = df.iloc[time_idx]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Timestamp", current_state['Timestamp'].strftime('%Y-%m-%d %H:%M'))
    c2.metric("Payload (Revolutions)", f"{current_state['revolutions']:.1f}")
    c3.metric("Vibration (Color)", f"{current_state['vibration']:.1f} Hz")

    # Dynamic 3D Engine
    fig_3d = go.Figure()
    
    # Static Shaft
    fig_3d.add_trace(go.Mesh3d(x=[-2, 2, 2, -2, -2, 2, 2, -2], y=[-2, -2, 2, 2, -2, -2, 2, 2], z=[0, 0, 0, 0, 2, 2, 2, 2], alphahull=1, opacity=0.05, color='cyan'))
    
    # Dynamic Elevator Car (Height based on x3, Color based on Vibration)
    z_height = current_state['x3'] # Maps to vertical position
    vib_color = 'red' if current_state['vibration'] > vib_thresh else 'orange' if current_state['vibration'] > vib_thresh - 15 else 'green'
    
    fig_3d.add_trace(go.Scatter3d(
        x=[current_state['x1']], y=[current_state['x2']], z=[z_height],
        mode='markers', marker=dict(size=30, color=vib_color, symbol='square'), name='Elevator Car'
    ))
    
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 2], xaxis_title='Sway X', yaxis_title='Depth Y', zaxis_title='Height Z'), template="plotly_dark", height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================================================
# MODULE 4: GEMINI AI
# ==================================================
elif nav == "🤖 4. Gemini AI Chief":
    st.title("🤖 Gemini Maintenance Chief")
    
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I'm monitoring the elevator's volatility and load metrics. How can I help?"}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., Analyze the cost savings vs. failure rate."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Data context: Avg Vib: {df['vibration'].mean():.1f}Hz. Volatility state: {df['State'].iloc[-1]}. AI prevented {df['Failure'].sum()} failures. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
