import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import numpy as np

# ==========================================
# 1. PREMIUM UI & BRANDING
# ==========================================
st.set_page_config(page_title="Sentinel AI | Elevator OS", page_icon="🏗️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border-radius: 12px; padding: 20px;
        border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #238636; color: white; }
    .status-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (PRO CACHING)
# ==========================================
@st.cache_data
def get_processed_data():
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    df = df.dropna(subset=['vibration'])
    # Predictive Math: Calculate a rolling 'Health Score'
    df['health_score'] = 100 - (df['vibration'] * 0.5 + (df['humidity'] - 72) * 2)
    df['health_score'] = df['health_score'].clip(lower=0, upper=100)
    return df

try:
    df = get_processed_data()
except Exception as e:
    st.error("🚨 Dataset Missing. Please upload 'Elevator predictive-maintenance-dataset.csv' to your GitHub root.")
    st.stop()

# ==========================================
# 3. SIDEBAR COMMAND CENTER
# ==========================================
with st.sidebar:
    st.title("🏗️ SENTINEL v2.0")
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=100)
    st.markdown("---")
    page = st.radio("OPERATIONAL MODULES", ["Command Center", "3D Digital Twin", "Analytics Lab", "Gemini AI Advisor"])
    st.markdown("---")
    st.subheader("System Configuration")
    alert_threshold = st.slider("Vibration Alert Limit", 0, 100, 45)
    st.success("Database Connected")

# ==========================================
# 4. MODULE: COMMAND CENTER (DASHBOARD)
# ==========================================
if page == "Command Center":
    st.title("🎮 Fleet Command Center")
    
    # Real-time Metrics
    c1, c2, c3, c4 = st.columns(4)
    avg_vib = df['vibration'].mean()
    health = df['health_score'].iloc[-1]
    anomalies = len(df[df['vibration'] > alert_threshold])
    
    c1.metric("Current Health", f"{health:.1f}%", delta=f"{health-90:.1f}%")
    c2.metric("Mean Vibration", f"{avg_vib:.2f} Hz")
    c3.metric("Anomaly Events", anomalies, delta="Active" if anomalies > 0 else "None", delta_color="inverse")
    c4.metric("Status", "Operational" if health > 70 else "URGENT REPAIR")

    # Live Sensor Pulse
    st.subheader("📈 Vibration Pulse Stream")
    fig_pulse = px.area(df.iloc[::25], x='ID', y='vibration', template="plotly_dark", color_discrete_sequence=['#58a6ff'])
    fig_pulse.add_hline(y=alert_threshold, line_dash="dash", line_color="#f85149", annotation_text="CRITICAL THRESHOLD")
    st.plotly_chart(fig_pulse, use_container_width=True)

# ==========================================
# 5. MODULE: 3D DIGITAL TWIN (PREMIUM)
# ==========================================
elif page == "3D Digital Twin":
    st.title("🧊 3D Movement Digital Twin")
    st.markdown("Mapping high-vibration hotspots across the vertical shaft coordinates ($X1, X2, X3$).")
    
    # 3D Mapping
    sample_df = df.sample(min(5000, len(df)))
    fig_3d = px.scatter_3d(
        sample_df, x='x1', y='x2', z='x3',
        color='vibration', size='revolutions',
        color_continuous_scale='Turbo', opacity=0.8,
        template="plotly_dark", height=700
    )
    fig_3d.update_layout(scene=dict(xaxis_title='Lateral X', yaxis_title='Depth Y', zaxis_title='Vertical Z'))
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# 6. MODULE: ANALYTICS LAB
# ==========================================
elif page == "Analytics Lab":
    st.title("🧪 Predictive Analytics Lab")
    
    t1, t2 = st.tabs(["Stress Correlation", "Sensor Distribution"])
    
    with t1:
        st.subheader("The Humidity Effect")
        st.write("Applying OLS Regression to understand environmental impact on vibration.")
        fig_ols = px.scatter(df.sample(2000), x='humidity', y='vibration', trendline="ols", 
                           template="plotly_dark", color='revolutions', trendline_color_override="#f85149")
        st.plotly_chart(fig_ols, use_container_width=True)

    with t2:
        st.subheader("Inter-Sensor Variance")
        df_melt = df[['x1','x2','x3','x4','x5']].melt()
        fig_box = px.box(df_melt, x='variable', y='value', color='variable', template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# 7. MODULE: GEMINI AI ADVISOR
# ==========================================
elif page == "Gemini AI Advisor":
    st.title("🤖 Maintenance Intelligence")
    
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🔑 API Key Missing. Add `GOOGLE_API_KEY` to your Streamlit Secrets.")
    else:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat interface
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about repair procedures or sensor anomalies..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                context = f"Elevator Data Summary: Avg Vibration {df['vibration'].mean():.2f}, Max Humidity {df['humidity'].max():.2f}. Anomaly Count: {len(df[df['vibration']>alert_threshold])}."
                response = model.generate_content(f"{context} User Question: {prompt}")
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
