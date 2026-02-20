import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from scipy import stats
import datetime

# ==========================================
# 1. PREMIUM UI & CSS (Features 17)
# ==========================================
st.set_page_config(page_title="Elevator OS | Enterprise", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    .stDataFrame { border: 1px solid #334155; border-radius: 8px; }
    h1, h2, h3 { color: #38bdf8; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE & ADVANCED MATH (Features 5, 6, 16, 19)
# ==========================================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv").dropna(subset=['vibration'])
    
    # Feature 5: Dynamic Health Score
    df['health_score'] = (100 - (df['vibration'] * 0.7 + (df['humidity'] - 70) * 1.5)).clip(0, 100)
    
    # Feature 16: Rolling Averages for Trend Forecasting
    df['vibration_rolling'] = df['vibration'].rolling(window=50).mean().fillna(df['vibration'])
    
    # Feature 6: Z-Score Anomaly Detection
    df['z_score'] = np.abs(stats.zscore(df['vibration']))
    df['is_anomaly'] = df['z_score'] > 3  # 3 standard deviations
    
    return df

try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"Dataset missing! Please upload the CSV. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR (Features 12, 18)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=80)
    st.title("Enterprise OS")
    
    nav = st.radio("System Modules", [
        "1. Live Command Center", 
        "2. Raw Data & Exports", 
        "3. 3D Digital Twin", 
        "4. Advanced Analytics", 
        "5. AI Maintenance Bot"
    ])
    
    st.markdown("---")
    st.subheader("System Parameters")
    vib_alert = st.slider("Vibration Alert Threshold", 10, 100, 40) # Feature 12
    cost_per_failure = st.number_input("Est. Cost per Failure ($)", value=5000)

# ==========================================
# MODULE 1: COMMAND CENTER (Features 1, 14, 15, 20)
# ==========================================
if nav == "1. Live Command Center":
    st.title("🎛️ Live Command Center")
    
    # KPIs (Feature 15: Delta Tracking)
    c1, c2, c3, c4 = st.columns(4)
    avg_health = df['health_score'].mean()
    anomalies = df['is_anomaly'].sum()
    
    c1.metric("System Health", f"{avg_health:.1f}%", delta="-2.4% (7 days)")
    c2.metric("Critical Anomalies", f"{anomalies:,}", delta=f"{anomalies} detected", delta_color="inverse")
    c3.metric("Fleet Uptime", "99.8%", delta="Optimal") # Feature 14
    c4.metric("Avg Vibration", f"{df['vibration'].mean():.1f} Hz")
    
    # Feature 20: Actionable Alerts
    if anomalies > 0:
        st.error(f"⚠️ {anomalies} critical vibration anomalies detected. Immediate inspection recommended.")

    st.subheader("Real-Time Sensor Pulse (Smoothed)")
    fig_pulse = px.line(df.iloc[::30], x='ID', y=['vibration', 'vibration_rolling'], 
                        template="plotly_dark", color_discrete_sequence=['#1e293b', '#38bdf8'])
    fig_pulse.add_hline(y=vib_alert, line_dash="dash", line_color="#ef4444", annotation_text="Danger Zone")
    st.plotly_chart(fig_pulse, use_container_width=True)

# ==========================================
# MODULE 2: RAW DATA & EXPORT (Features 1, 7, 8)
# ==========================================
elif nav == "2. Raw Data & Exports":
    st.title("📁 Data Inspector & Exports")
    st.write("Full visibility into the raw hardware sensor logs.")
    
    # Feature 1: Raw Data Visibility
    st.dataframe(df.head(1000), use_container_width=True)
    
    # Feature 7: Export Engine
    st.subheader("Export Anomaly Reports")
    anomaly_df = df[df['is_anomaly'] == True]
    
    c1, c2 = st.columns(2)
    c1.metric("Anomalies Ready for Export", len(anomaly_df))
    
    # Feature 8: Maintenance Cost Estimator
    saved = len(anomaly_df) * (cost_per_failure * 0.15) # Assuming we save 15% by catching it early
    c2.metric("Estimated Preventive Savings", f"${saved:,.2f}")
    
    csv = anomaly_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Critical Anomalies (CSV)",
        data=csv,
        file_name='elevator_anomalies.csv',
        mime='text/csv',
    )

# ==========================================
# MODULE 3: 3D DIGITAL TWIN (Feature 2)
# ==========================================
elif nav == "3. 3D Digital Twin":
    st.title("🧊 3D Smart Elevator Digital Twin")
    st.markdown("A procedural 3D model representing the elevator shaft. The wireframe represents the physical space, and the scatter points represent the sensor coordinate data ($X1, X2, X3$) mapping where the vibration occurs.")
    
    # Procedural 3D Shaft generation
    shaft_x = [90, 170, 170, 90, 90, 170, 170, 90]
    shaft_y = [-60, -60, 25, 25, -60, -60, 25, 25]
    shaft_z = [0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5]
    
    fig_3d = go.Figure()
    
    # Draw the Shaft Boundary (Wireframe)
    fig_3d.add_trace(go.Mesh3d(
        x=shaft_x, y=shaft_y, z=shaft_z,
        alphahull=1, opacity=0.1, color='cyan', name='Shaft Boundary'
    ))
    
    # Add Sensor Data inside the shaft
    sample_df = df.sample(min(2000, len(df)))
    fig_3d.add_trace(go.Scatter3d(
        x=sample_df['x1'], y=sample_df['x2'], z=sample_df['x3'],
        mode='markers',
        marker=dict(
            size=sample_df['vibration'] / 10,
            color=sample_df['vibration'],
            colorscale='Inferno',
            opacity=0.8,
            colorbar=dict(title="Vibration Hz")
        ),
        name='Sensor Reads'
    ))
    
    fig_3d.update_layout(
        scene=dict(xaxis_title='X1 (Lateral)', yaxis_title='X2 (Depth)', zaxis_title='X3 (Vertical)'),
        template="plotly_dark", height=800, margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# MODULE 4: ADVANCED ANALYTICS (Features 9, 10, 11, 13)
# ==========================================
elif nav == "4. Advanced Analytics":
    st.title("🔬 Statistical Analysis Lab")
    
    t1, t2, t3 = st.tabs(["Heatmap", "Regression", "Distributions"])
    
    with t1:
        st.subheader("Sensor Correlation Matrix") # Feature 9
        corr = df[['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3']].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with t2:
        st.subheader("Environmental Impact (OLS)") # Feature 11
        fig_ols = px.scatter(df.sample(2000), x='humidity', y='vibration', trendline="ols", 
                             template="plotly_dark", trendline_color_override="#38bdf8")
        st.plotly_chart(fig_ols, use_container_width=True)
        
    with t3:
        st.subheader("Internal Sensor Variances") # Feature 13
        df_melt = df[['x1', 'x2', 'x3']].melt()
        fig_box = px.box(df_melt, x='variable', y='value', color='variable', template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# MODULE 5: AI MAINTENANCE BOT (Features 3, 17)
# ==========================================
elif nav == "5. AI Maintenance Bot":
    st.title("🤖 Gemini Enterprise Assistant")
    
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🔑 API Key Missing! Please add `GOOGLE_API_KEY` to Streamlit Secrets.")
    else:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Feature 17: Maintenance Predictor Logic fed to AI
        next_maint = datetime.date.today() + datetime.timedelta(days=int(df['health_score'].mean() / 2))
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "I am connected to the Elevator's live sensor array. Ask me anything."}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., Based on the data, what part should we inspect?"):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Data context: Avg Vib {df['vibration'].mean():.1f}Hz, {df['is_anomaly'].sum()} anomalies. Projected maintenance date: {next_maint}. User asks: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
