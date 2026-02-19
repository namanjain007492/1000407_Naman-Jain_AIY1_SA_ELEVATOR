import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openai

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Elevator Sentinel Pro",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS for a dark, premium industrial feel
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_safe: True)

# ==========================================
# DATA LOADING (Stage 2)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    df_clean = df.dropna(subset=['vibration']).copy()
    return df_clean

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🏗️ Sentinel Pro")
st.sidebar.markdown("---")
menu = st.sidebar.selectbox("Dashboard Modules", 
    ["Live Fleet Status", "3D Movement Analysis", "Statistical Trends", "AI Maintenance Assistant"])

vibration_threshold = st.sidebar.slider("Anomaly Threshold (Vibration)", 0, 100, 45)

# ==========================================
# MODULE 1: LIVE FLEET STATUS
# ==========================================
if menu == "Live Fleet Status":
    st.title("⚡ Operational Health Overview")
    
    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    avg_vib = df['vibration'].mean()
    anomalies = df[df['vibration'] > vibration_threshold].shape[0]
    
    m1.metric("Avg. Vibration", f"{avg_vib:.2f} Hz")
    m2.metric("Total Revolutions", f"{df['revolutions'].sum():,.0f}")
    m3.metric("Anomalies Detected", anomalies, delta=f"{anomalies}", delta_color="inverse")
    m4.metric("Fleet Status", "Warning" if anomalies > 50 else "Optimal")

    st.subheader("Vibration Timeline (Sensor Stream)")
    fig_line = px.line(df.iloc[::50], x='ID', y='vibration', color_discrete_sequence=['#00d4ff'])
    fig_line.add_hline(y=vibration_threshold, line_dash="dash", line_color="red", annotation_text="Critical Limit")
    st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# MODULE 2: 3D MOVEMENT ANALYSIS (The "Working Elevator" View)
# ==========================================
elif menu == "3D Movement Analysis":
    st.title("🧊 3D Spatial Sensor Mapping")
    st.markdown("This plot visualizes the **geometric path** of the elevator car using sensors X1, X2, and X3. Points are colored by vibration intensity.")
    
    # 3D Scatter Plot
    # We sample the data to keep the 3D interaction smooth
    df_sample = df.sample(min(5000, len(df)))
    
    fig_3d = px.scatter_3d(
        df_sample, x='x1', y='x2', z='x3',
        color='vibration',
        size='revolutions',
        opacity=0.7,
        color_continuous_scale='Inferno',
        title="Elevator Movement Path in 3D Space",
        labels={'x1': 'Lateral X', 'x2': 'Depth Y', 'x3': 'Vertical Z'}
    )
    
    fig_3d.update_layout(scene=dict(bgcolor='#0e1117'), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.info("💡 **Insight:** Look for clusters of bright yellow dots. These represent specific physical coordinates in the elevator shaft where vibration is peaking, indicating a potential rail misalignment.")

# ==========================================
# MODULE 3: STATISTICAL TRENDS
# ==========================================
elif menu == "Statistical Trends":
    st.title("📈 Correlation & Stress Testing")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Humidity vs Vibration (Regression)")
        # This is where STATSMODELS is needed!
        fig_scat = px.scatter(df.sample(2000), x='humidity', y='vibration', 
                             trendline="ols", trendline_color_override="red",
                             color_discrete_sequence=['#30363d'])
        st.plotly_chart(fig_scat, use_container_width=True)

    with c2:
        st.subheader("Sensor Distribution (x1-x5)")
        df_melt = df[['x1','x2','x3','x4','x5']].melt()
        fig_box = px.box(df_melt, x='variable', y='value', color='variable')
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# MODULE 4: AI MAINTENANCE ASSISTANT
# ==========================================
elif menu == "AI Maintenance Assistant":
    st.title("🤖 Sentinel AI Advisor")
    
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("Please add `OPENAI_API_KEY` to Streamlit Secrets to enable this.")
    else:
        client = openai.OpenAI(api_key=api_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: Why is high vibration dangerous in elevator shafts?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
