import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq
import google.generativeai as genai
import time
import datetime

# ==================================================
# 1. UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Elevator AI | V5 Capstone", page_icon="🏢", layout="wide")

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
    df['Rolling_Mean'] = df['vibration'].rolling(window=20).mean().bfill()
    df['Rolling_Std'] = df['vibration'].rolling(window=20).std().bfill()
    df['Upper_Band'] = df['Rolling_Mean'] + (df['Rolling_Std'] * 2) # Bollinger Upper
    df['Lower_Band'] = df['Rolling_Mean'] - (df['Rolling_Std'] * 2) # Bollinger Lower
    df['Z_Score'] = np.abs((df['vibration'] - df['vibration'].mean()) / df['vibration'].std())
    
    # Floor Mapping
    conditions = [(df['x3'] < 0.5), (df['x3'] >= 0.5) & (df['x3'] < 0.75), (df['x3'] >= 0.75)]
    df['Floor'] = np.select(conditions, ['Ground', 'Floor 1', 'Floor 2'], default='Unknown')
    
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
        "📖 1. Live Demo & Streaming",
        "📈 2. Math & Volatility (FA-2)",
        "🧊 3. 3D Digital Twin",
        "🧠 4. ML & Financial ROI",
        "🤖 5. Gemini AI Chat"
    ])
    
    st.markdown("---")
    vib_thresh = st.slider("Vibration Alert (Hz)", 10.0, 100.0, 55.0)

# ==================================================
# MODULE 1: LIVE DEMO & STREAMING
# ==================================================
if nav == "📖 1. Live Demo & Streaming":
    st.title("Welcome to the Predictive Maintenance OS")
    
    st.markdown("""
    <div class="demo-card">
        <h3>🚀 Interactive Demonstration Mode</h3>
        <p>Press the button below to simulate real-time sensor data streaming from the elevator. Watch the AI detect anomalies instantly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ Start Live Telemetry Stream"):
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        metric_placeholder = st.empty()
        
        sim_data = df.head(150) # Demo stream length
        for i in range(10, len(sim_data), 2):
            progress_bar.progress(int((i/len(sim_data))*100))
            current_slice = sim_data.iloc[:i]
            latest = current_slice.iloc[-1]
            
            with metric_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Floor", latest['Floor'])
                c2.metric("Vibration", f"{latest['vibration']:.1f} Hz", delta="High" if latest['vibration'] > vib_thresh else "Normal", delta_color="inverse")
                c3.metric("Passenger Load", f"{latest['Load']} Pax")
                c4.metric("Z-Score Volatility", f"{latest['Z_Score']:.2f} σ")
                
            fig = px.line(current_slice, x='Timestamp', y='vibration', template="plotly_dark", title="Live Sensor Feed (Auto-updating)")
            fig.add_hline(y=vib_thresh, line_dash="dash", line_color="red")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
            
        st.success("✅ Demo Complete! Explore the advanced math in the next modules.")

# ==================================================
# MODULE 2: MATHEMATICAL VOLATILITY (FA-2)
# ==================================================
elif nav == "📈 2. Math & Volatility (FA-2)":
    st.title("📈 Advanced Mathematical Analysis")
    st.write("Applying sine/cosine transformations and financial market math to mechanical sensors.")
    
    tab1, tab2, tab3 = st.tabs(["Bollinger Bands", "Fast Fourier Transform (FFT)", "Z-Score Matrix"])
    
    with tab1:
        st.subheader("Volatility Bounds (Bollinger Bands)")
        st.write("Using 2 Standard Deviations from the Moving Average to map acceptable mechanical 'price' swings.")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Upper_Band'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Lower_Band'], fill='tonexty', fillcolor='rgba(56, 189, 248, 0.1)', line=dict(color='rgba(0,0,0,0)'), name='Volatility Bounds'))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['vibration'], line=dict(color='#38bdf8', width=1), name='Raw Vibration'))
        fig_bb.add_trace(go.Scatter(x=df['Timestamp'], y=df['Rolling_Mean'], line=dict(color='#f59e0b', width=2), name='Moving Average'))
        fig_bb.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_bb, use_container_width=True)

    with tab2:
        st.subheader("Frequency Analysis (Fast Fourier Transform)")
        st.write("Deconstructing complex vibration noise into pure sine and cosine frequencies.")
        
        # FFT Math Algorithm
        N = len(df)
        T = 1.0 / 800.0 # Sample spacing
        y = df['vibration'].values
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        
        fig_fft = px.line(x=xf[1:], y=2.0/N * np.abs(yf[1:N//2]), template="plotly_dark", labels={'x': 'Frequency (Hz)', 'y': 'Amplitude'})
        fig_fft.update_traces(line_color='#a855f7')
        st.plotly_chart(fig_fft, use_container_width=True)

    with tab3:
        st.subheader("Z-Score Outlier Density")
        fig_scatter = px.scatter(df, x='Load', y='vibration', color=df['Z_Score'] > 2, color_discrete_map={True: '#ef4444', False: '#10b981'}, template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==================================================
# MODULE 3: 3D DIGITAL TWIN
# ==================================================
elif nav == "🧊 3. 3D Digital Twin":
    st.title("🧊 Time-Lapse 3D Elevator Twin")
    st.write("Slide the timeline to animate the elevator car's movement through the shaft.")
    
    time_idx = st.slider("Scrub Timeline", 0, len(df)-1, 0)
    current = df.iloc[time_idx]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Position (x3)", f"{current['x3']:.3f}", current['Floor'])
    c2.metric("Payload / Load", f"{current['Load']} Pax")
    c3.metric("Vibration State", f"{current['vibration']:.1f} Hz", delta="Critical" if current['vibration'] > vib_thresh else "Stable", delta_color="inverse")

    fig_3d = go.Figure()
    
    # Shaft
    fig_3d.add_trace(go.Mesh3d(x=[-2, 2, 2, -2, -2, 2, 2, -2], y=[-2, -2, 2, 2, -2, -2, 2, 2], z=[0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5], alphahull=1, opacity=0.05, color='white'))
    
    # Floors
    for z_val, f_name in zip([0.3, 0.6, 0.9], ['Ground', 'Floor 1', 'Floor 2']):
        fig_3d.add_trace(go.Surface(x=[[-2, 2], [-2, 2]], y=[[-2, -2], [2, 2]], z=[[z_val, z_val], [z_val, z_val]], opacity=0.2, colorscale='Greens', showscale=False, name=f_name))

    # Elevator Car
    vib_color = 'red' if current['vibration'] > vib_thresh else 'orange' if current['vibration'] > vib_thresh - 15 else 'green'
    fig_3d.add_trace(go.Scatter3d(
        x=[current['x1']], y=[current['x2']], z=[current['x3']],
        mode='markers', marker=dict(size=35, color=vib_color, symbol='square'), name='Elevator Car'
    ))
    
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 1.5], zaxis_title='Vertical Height'), template="plotly_dark", height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================================================
# MODULE 4: ML & FINANCIAL ROI
# ==================================================
elif nav == "🧠 4. ML & Financial ROI":
    st.title("💰 Machine Learning & Financial ROI")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression Predictor")
        X = df[['Temperature', 'vibration', 'Load']]
        y = df['Failure']
        
        if y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
            prob = clf.predict_proba(X.iloc[[-1]])[0][1]
            st.metric("Imminent Failure Probability", f"{prob*100:.1f}%")
            
            # Forecast
            st.write("7-Day Forecast (Linear Extrapolation)")
            df_daily = df.set_index('Timestamp').resample('D')['vibration'].mean().dropna().reset_index()
            df_daily['DayNum'] = np.arange(len(df_daily))
            lr = LinearRegression().fit(df_daily[['DayNum']], df_daily['vibration'])
            future_vib = lr.predict(pd.DataFrame({'DayNum': np.arange(len(df_daily), len(df_daily)+7)}))
            fig_fc = px.line(x=range(1, 8), y=future_vib, template="plotly_dark", labels={'x':'Days in Future', 'y':'Expected Vibration'})
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.success("No historical failures to train ML model.")

    with col2:
        st.subheader("Financial Savings Estimator")
        st.write("Calculate money saved by predicting failures.")
        cost_breakdown = st.number_input("Cost of Major Breakdown ($)", value=25000)
        cost_preventative = st.number_input("Cost of Preventative Maintenance ($)", value=1200)
        
        anomalies_caught = len(df[df['Z_Score'] > 2])
        money_saved = (cost_breakdown - cost_preventative) * anomalies_caught
        
        st.metric("Total Money Saved", f"${money_saved:,.2f}", delta="Positive ROI")
        st.info(f"The AI intercepted **{anomalies_caught}** critical anomalies before they caused major structural damage.")

# ==================================================
# MODULE 5: GEMINI AI
# ==================================================
elif nav == "🤖 5. Gemini AI Chat":
    st.title("🤖 Gemini AI Control Center")
    
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 Add `GOOGLE_API_KEY` to Streamlit Cloud Secrets.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am the AI Maintenance Chief. Ask me about FFT frequencies or Bollinger Bands!"}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., Explain what the Bollinger Bands show."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Data context: Avg Vib {df['vibration'].mean():.1f}Hz. Peak Load {df['Load'].max()} Pax. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
