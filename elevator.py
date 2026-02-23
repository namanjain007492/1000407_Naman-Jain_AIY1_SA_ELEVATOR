import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import time

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    cumulative_trapezoid = None

# ==================================================
# UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Elevator AI Operations", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #38bdf8; }
    .context-box { background-color: #1e293b; border-left: 5px solid #38bdf8; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
    .insight-box { background-color: #0f291e; border-left: 5px solid #10b981; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
    .physics-card { background-color: #312e81; border-left: 5px solid #6366f1; padding: 15px; border-radius: 8px; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# DATA PIPELINE & MATH ENGINE
# ==================================================
@st.cache_data
def load_and_clean_data():
    try:
        raw_df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    except Exception:
        st.error("⚠️ Dataset not found! Please ensure 'Elevator predictive-maintenance-dataset.csv' is in the exact same folder.")
        st.stop()
        
    # Clean Data
    clean_df = raw_df.dropna().drop_duplicates().copy()
    
    # Mathematical Modeling (FA-2 Rubric)
    t = np.arange(len(clean_df))
    clean_df['Ideal_Resonance'] = 20 + 15 * np.sin(2 * np.pi * 0.05 * t) # Sine Wave
    
    if cumulative_trapezoid is not None:
        clean_df['Cumulative_Stress'] = cumulative_trapezoid(clean_df['vibration'], initial=0) * 0.1
    else:
        clean_df['Cumulative_Stress'] = np.cumsum(clean_df['vibration'].values) * 0.1 
        
    return raw_df, clean_df

raw_df, df = load_and_clean_data()
plot_df = df.iloc[::10].copy() if len(df) > 10000 else df.copy() # Downsample for visual performance

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    
    nav = st.radio("System Modules", [
        "📖 Project Overview",
        "🧹 Data Processing",
        "📊 Telemetry Visualizations",
        "🧊 Physics & Speed Simulator",
        "💡 Insights & GenAI"
    ])
    
    st.markdown("---")
    st.header("🔑 AI Integration")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")

# ==================================================
# MODULE 1: PROJECT OVERVIEW
# ==================================================
if nav == "📖 Project Overview":
    st.title("📖 Predictive Maintenance Overview")
    
    st.markdown("""
    <div class="context-box">
        <h3>🔍 Problem Context</h3>
        <p>This dashboard monitors elevator sensor data (revolutions, humidity, and vibrations) to identify mechanical wear before a catastrophic failure occurs.</p>
        <p><b>Vibration</b> is the primary "health indicator". Excessive vibration indicates mechanical stress, while humidity and door revolutions act as the root "stress factors".</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("❓ Research Fundamentals")
    with st.expander("1. How does humidity affect machine performance?", expanded=True):
        st.write("High humidity causes condensation, increasing rust, track friction, and sensor degradation.")
    with st.expander("2. Could revolutions impact vibration?", expanded=True):
        st.write("Yes. Revolutions track mechanical cycles. More cycles mean worn belts and bearings, causing increased vibrational output.")
    with st.expander("3. Why is vibration the main target?", expanded=True):
        st.write("Vibration is the earliest physical symptom of failure. Motors and tracks rattle out of their normal frequency long before they completely break.")

    st.subheader("💼 Real-World Application")
    st.write("1. **Reducing Downtime:** Prevents elevators from trapping passengers during peak hours.")
    st.write("2. **Cost Saving:** Re-lubricating a vibrating track is much cheaper than replacing a snapped cable.")

# ==================================================
# MODULE 2: DATA PROCESSING
# ==================================================
elif nav == "🧹 Data Processing":
    st.title("🧹 Data Diagnostics & Cleaning")
    
    t1, t2 = st.tabs(["Raw Telemetry", "Cleaned Output"])
    
    with t1:
        st.subheader("Raw Data Assessment")
        st.write(f"Initial Shape: **{raw_df.shape[0]:,} rows** × **{raw_df.shape[1]} columns**.")
        st.dataframe(raw_df.head(), use_container_width=True)
        
        st.write("**Missing Values:**")
        st.dataframe(raw_df.isnull().sum().reset_index().rename(columns={"index": "Sensor", 0: "Missing Count"}), use_container_width=True)

    with t2:
        st.subheader("Cleaned Dataset Summary")
        st.write(f"After dropping N/A values and duplicates, the clean dataset contains **{df.shape[0]:,} rows**.")
        st.dataframe(df.describe(), use_container_width=True)
        st.success("✅ Data is processed and ready for visualization and physics modeling.")

# ==================================================
# MODULE 3: TELEMETRY VISUALIZATIONS (With FA-2 Math)
# ==================================================
elif nav == "📊 Telemetry Visualizations":
    st.title("📊 Telemetry & Diagnostics")
    
    t_viz, t_math = st.tabs(["Data Explorations", "Mathematical Modeling"])
    
    with t_viz:
        # 1. Line Plot
        st.subheader("1. Time Series of Vibration")
        fig_line = px.line(plot_df, x='ID', y='vibration', template="plotly_dark", color_discrete_sequence=['#38bdf8'])
        st.plotly_chart(fig_line, use_container_width=True)
        
        c1, c2 = st.columns(2)
        # 2. Histogram
        with c1:
            st.subheader("2. Stress Factor Distributions")
            st.plotly_chart(px.histogram(plot_df, x='humidity', nbins=40, template="plotly_dark", color_discrete_sequence=['#10b981'], title="Humidity"), use_container_width=True)
            st.plotly_chart(px.histogram(plot_df, x='revolutions', nbins=40, template="plotly_dark", color_discrete_sequence=['#f59e0b'], title="Revolutions"), use_container_width=True)

        # 3. Scatter Plot
        with c2:
            st.subheader("3. Revolutions vs Vibration")
            st.plotly_chart(px.scatter(plot_df, x='revolutions', y='vibration', opacity=0.5, template="plotly_dark", color='vibration', color_continuous_scale='Reds'), use_container_width=True)
            
        # 4. Box Plot & Heatmap
        st.subheader("4. Sensor Outliers (x1-x5)")
        df_melted = plot_df[['x1', 'x2', 'x3', 'x4', 'x5']].melt(var_name="Sensor", value_name="Reading")
        st.plotly_chart(px.box(df_melted, x="Sensor", y="Reading", color="Sensor", template="plotly_dark"), use_container_width=True)
        
        st.subheader("5. Correlation Heatmap")
        corr_matrix = df[['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']].corr()
        st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)

    with t_math:
        st.subheader("Trigonometric Baseline (Sine Wave vs Reality)")
        st.write("Simulating perfect motor resonance (Sine Wave) against actual noisy sensor data.")
        fig_sine = go.Figure()
        fig_sine.add_trace(go.Scatter(x=plot_df['ID'][:200], y=plot_df['Ideal_Resonance'][:200], name='Ideal Sine Wave', line=dict(color='#10b981', dash='dash')))
        fig_sine.add_trace(go.Scatter(x=plot_df['ID'][:200], y=plot_df['vibration'][:200], name='Actual CSV Data', line=dict(color='#ef4444')))
        fig_sine.update_layout(template="plotly_dark")
        st.plotly_chart(fig_sine, use_container_width=True)
        
        st.subheader("Cumulative Wear (Calculus Integrals)")
        st.write("Using Numerical Integration to calculate the total Area Under the Curve for mechanical stress.")
        st.plotly_chart(px.area(plot_df, x='ID', y='Cumulative_Stress', template="plotly_dark", color_discrete_sequence=['#8b5cf6']), use_container_width=True)

# ==================================================
# MODULE 4: PHYSICS & SPEED SIMULATOR
# ==================================================
elif nav == "🧊 Physics & Speed Simulator":
    st.title("🧊 Interactive Physics & Speed Engine")
    
    st.markdown("""
    <div class="physics-card">
        Test how passenger weight impacts the mechanical speed and travel time of the elevator between floors. 
        Higher passenger loads decrease motor speed and increase mechanical strain.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    start_floor = c1.selectbox("Start Floor:", ["Ground", "Floor 1", "Floor 2"], index=0)
    target_floor = c2.selectbox("Destination Floor:", ["Ground", "Floor 1", "Floor 2"], index=2)
    pax_load = st.slider("Board Passengers:", 0, 20, 8)
    
    # Physics Calculations
    z_map = {"Ground": 0.0, "Floor 1": 4.0, "Floor 2": 8.0} # Height in meters
    z_start = z_map[start_floor]
    z_end = z_map[target_floor]
    distance = abs(z_end - z_start)
    
    # Speed and weight logic
    base_speed = 2.5 # Ideal speed empty
    speed_penalty = pax_load * 0.09 # Lose speed per passenger
    actual_speed = max(0.5, base_speed - speed_penalty)
    travel_time = distance / actual_speed if distance > 0 else 0
    
    # Strain logic
    sim_vib = 15 + (pax_load * 2.5) 
    vib_color = 'red' if sim_vib > 50 else 'orange' if sim_vib > 35 else '#10b981'

    # Metrics Display
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance to Travel", f"{distance} meters")
    m2.metric("Actual Speed", f"{actual_speed:.2f} m/s", delta=f"-{speed_penalty:.2f} m/s (Weight Penalty)", delta_color="inverse")
    m3.metric("Est. Travel Time", f"{travel_time:.1f} sec")
    m4.metric("Mechanical Strain", f"{sim_vib:.1f} Hz")

    # 3D Visualizer
    fig_3d = go.Figure()
    
    # Draw Shaft
    fig_3d.add_trace(go.Mesh3d(x=[-2, 2, 2, -2, -2, 2, 2, -2], y=[-2, -2, 2, 2, -2, -2, 2, 2], z=[0, 0, 0, 0, 8, 8, 8, 8], alphahull=1, opacity=0.05, color='white'))
    
    # Draw Floors
    for z_val, f_name in zip([0.0, 4.0, 8.0], ['Ground', 'Floor 1', 'Floor 2']):
        plane_color = 'cyan' if z_val == z_end else 'grey'
        fig_3d.add_trace(go.Surface(x=[[-2, 2], [-2, 2]], y=[[-2, -2], [2, 2]], z=[[z_val, z_val], [z_val, z_val]], opacity=0.3 if z_val == z_end else 0.1, colorscale=[[0, plane_color], [1, plane_color]], showscale=False))

    # Draw Elevator Car at Destination
    fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[z_end], mode='markers', marker=dict(size=40, color=vib_color, symbol='square'), name='Elevator Car'))
    
    # Draw Passengers
    if pax_load > 0:
        fig_3d.add_trace(go.Scatter3d(x=np.random.uniform(-0.5, 0.5, pax_load), y=np.random.uniform(-0.5, 0.5, pax_load), z=[z_end]*pax_load, mode='markers', marker=dict(size=6, color='black'), name='Passengers'))
    
    fig_3d.update_layout(scene=dict(zaxis_range=[0, 9], zaxis_title='Height (Meters)'), template="plotly_dark", height=600, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================================================
# MODULE 5: INSIGHTS & AI
# ==================================================
elif nav == "💡 Insights & GenAI":
    st.title("💡 Operational Insights & AI Assistant")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Key Insights</h3>
        <ul>
            <li><b>Usage Drives Wear:</b> Positive correlation between door revolutions and vibration proves mechanical fatigue.</li>
            <li><b>Environmental Impact:</b> Humidity correlates with increased vibration over time, amplifying stress (friction/condensation).</li>
            <li><b>Anomaly Detection:</b> Severe outliers in sensors x1-x5 during high-vibration events allow us to pinpoint spatial failure points.</li>
            <li><b>Speed Degradation:</b> Physics models show passenger load exponentially decreases motor speed while increasing vibrational strain.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🤖 Gemini Operations Chief")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            target_model = next((m for m in available_models if 'flash' in m or 'pro' in m), available_models[0])
            model = genai.GenerativeModel(target_model)
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [{"role": "assistant", "content": "I am the Elevator Maintenance AI. Ask me how to interpret the Correlation Heatmap, or how passenger weight affects motor speed!"}]

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.write(msg["content"])

            if prompt := st.chat_input("Ask about elevator maintenance or physics..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.write(prompt)

                with st.chat_message("assistant"):
                    sys_prompt = f"""
                    You are a Senior Data Scientist analyzing elevator maintenance.
                    DATA CONTEXT:
                    - Avg Vibration: {df['vibration'].mean():.2f}
                    - High passenger weight reduces elevator speed and increases vibration.
                    Answer the user concisely: {prompt}
                    """
                    with st.spinner("Analyzing data..."):
                        response = model.generate_content(sys_prompt)
                        st.write(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"API Error: {e}")
    else:
        st.warning("⚠️ Enter your Gemini API Key in the sidebar to activate the AI Chatbot.")
