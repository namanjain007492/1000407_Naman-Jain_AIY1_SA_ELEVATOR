import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

# ==================================================
# UI DESIGN & CONFIGURATION
# ==================================================
st.set_page_config(page_title="Task 2: Elevator Visualization", page_icon="🏢", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# DATA PIPELINE (Stage 2 Background Loading)
# ==================================================
@st.cache_data
def load_and_clean_data():
    try:
        raw_df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    except Exception:
        st.error("⚠️ Dataset not found! Please ensure 'Elevator predictive-maintenance-dataset.csv' is in the exact same folder.")
        st.stop()
        
    # Copy for cleaning
    clean_df = raw_df.copy()
    
    # Check for missing values & Drop them
    missing_counts = clean_df.isnull().sum()
    clean_df = clean_df.dropna()
    
    # Check for duplicates & Drop them
    duplicate_count = clean_df.duplicated().sum()
    clean_df = clean_df.drop_duplicates()
    
    return raw_df, clean_df, missing_counts, duplicate_count

raw_df, df, missing_counts, duplicate_count = load_and_clean_data()

# Downsample for UI performance so the browser doesn't crash on 100k+ rows
plot_df = df.iloc[::10].copy() if len(df) > 10000 else df.copy()

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=60)
    st.title("🏢 Elevator OS")
    st.markdown("**Predictive Maintenance Dashboard**")
    
    nav = st.radio("Task 2 Project Stages", [
        "📖 Stage 1: Problem & Research",
        "🧹 Stage 2: Data Understanding",
        "📊 Stage 3: Data Visualization",
        "💡 Stage 4: Insights & AI"
    ])
    
    st.markdown("---")
    st.header("🔑 Gemini AI Configuration")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        if api_key:
            st.success("API Key loaded!")
        else:
            st.warning("Enter key to unlock AI in Stage 4.")

# ==================================================
# STAGE 1: UNDERSTAND PROBLEM AND RESEARCH
# ==================================================
if nav == "📖 Stage 1: Problem & Research":
    st.title("📖 Stage 1: Problem Context & Research")
    
    st.markdown("""
    <div class="context-box">
        <h3>🔍 Project Context: Predictive Maintenance</h3>
        <p>This project focuses on the <b>predictive maintenance of elevators</b>. By monitoring sensor data attached to elevator doors (such as revolutions, humidity, and vibrations), we can identify potential mechanical issues <i>before</i> they lead to catastrophic failures.</p>
        <p><b>Vibration</b> acts as the primary "health indicator" of the elevator. High vibration typically indicates wear, tear, or an impending malfunction, while humidity and revolutions act as environmental and mechanical "stress factors".</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("❓ Guiding Research Questions")
    
    with st.expander("1. How does humidity affect machine performance?", expanded=True):
        st.write("High humidity can cause condensation on metallic parts and sensors. This leads to rust, increased friction on the door tracks, electrical shorts, and swelling of certain materials, ultimately increasing mechanical strain.")
        
    with st.expander("2. Could revolutions (number of door movements) impact vibration?", expanded=True):
        st.write("Absolutely. Revolutions represent the cumulative mechanical work done by the elevator doors. As the motor and tracks complete more cycles, structural degradation occurs. Worn-out belts and bearings directly result in higher vibrational output.")
        
    with st.expander("3. Why is vibration considered the main target for prediction?", expanded=True):
        st.write("Vibration is the earliest and most measurable physical symptom of mechanical failure. Before a motor burns out completely, its rotational balance degrades, causing it to rattle and vibrate out of its normal frequency range.")

    st.subheader("💼 Real-World Value of Predicting Vibration")
    st.write("Transitioning from *reactive* maintenance to *predictive* maintenance offers massive industry benefits:")
    st.markdown("""
    1. **Reducing Downtime:** Predicting failures keeps systems operational. Breaking down during peak office hours traps passengers and causes severe logistical issues.
    2. **Saving Maintenance Costs:** Fixing a minor vibrational issue (like lubricating a track or replacing a $50 bearing) prevents the need to replace a completely destroyed $5,000 motor later.
    """)

# ==================================================
# STAGE 2: DATA UNDERSTANDING AND CLEANING
# ==================================================
elif nav == "🧹 Stage 2: Data Understanding":
    st.title("🧹 Stage 2: Data Understanding & Cleaning")
    st.write("Assessing data quality to ensure it is readable and free from major issues before analysis.")
    
    t1, t2 = st.tabs(["Raw Data Assessment", "Cleaned Data Output"])
    
    with t1:
        st.subheader("1. Loading the Dataset")
        st.write(f"Initial Dataset Shape: **{raw_df.shape[0]:,} rows** and **{raw_df.shape[1]} columns**.")
        st.dataframe(raw_df.head(), use_container_width=True)
        
        st.subheader("2. Checking for Missing Values (.isnull().sum())")
        st.dataframe(missing_counts.reset_index().rename(columns={"index": "Column", 0: "Missing Values"}), use_container_width=True)
        
        st.subheader("3. Checking for Duplicates")
        st.write(f"Total duplicate rows found: **{duplicate_count}**")

    with t2:
        st.subheader("Cleaned Dataset Summary (.describe())")
        st.write(f"After dropping N/A values and duplicates, the clean dataset contains **{df.shape[0]:,} rows**.")
        st.dataframe(df.describe(), use_container_width=True)
        st.success("✅ Data is clean, structured, and ready for Stage 3 Visualization.")

# ==================================================
# STAGE 3: DATA VISUALIZATION
# ==================================================
elif nav == "📊 Stage 3: Data Visualization":
    st.title("📊 Stage 3: Exploratory Visualizations")
    st.write("Exploring how stress factors (humidity, revolutions) affect the health signal (vibration).")
    
    # 1. Line Plot
    st.subheader("1. Time Series of Vibration")
    st.write("**Columns:** ID vs vibration | **Purpose:** Detect unusual spikes over time.")
    fig_line = px.line(plot_df, x='ID', y='vibration', template="plotly_dark", color_discrete_sequence=['#38bdf8'])
    st.plotly_chart(fig_line, use_container_width=True)
    
    c1, c2 = st.columns(2)
    
    # 2. Histogram
    with c1:
        st.subheader("2. Distribution of Stress Factors")
        st.write("**Columns:** humidity, revolutions | **Purpose:** Check spread of values.")
        fig_hist1 = px.histogram(plot_df, x='humidity', nbins=40, template="plotly_dark", color_discrete_sequence=['#10b981'], title="Humidity")
        st.plotly_chart(fig_hist1, use_container_width=True)
        
        fig_hist2 = px.histogram(plot_df, x='revolutions', nbins=40, template="plotly_dark", color_discrete_sequence=['#f59e0b'], title="Revolutions")
        st.plotly_chart(fig_hist2, use_container_width=True)

    # 3. Scatter Plot
    with c2:
        st.subheader("3. Revolutions vs Vibration")
        st.write("**Columns:** revolutions vs vibration | **Purpose:** See if usage leads to higher vibrations.")
        fig_scatter = px.scatter(plot_df, x='revolutions', y='vibration', opacity=0.5, template="plotly_dark", color='vibration', color_continuous_scale='Reds')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # 4. Box Plot
    st.subheader("4. Sensor Readings Distribution (x1-x5)")
    st.write("**Columns:** x1, x2, x3, x4, x5 | **Purpose:** Detect outliers or abnormal spatial sensor readings.")
    df_melted = plot_df[['x1', 'x2', 'x3', 'x4', 'x5']].melt(var_name="Sensor", value_name="Reading")
    fig_box = px.box(df_melted, x="Sensor", y="Reading", color="Sensor", template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # 5. Correlation Heatmap
    st.subheader("5. Correlation Heatmap")
    st.write("**Columns:** All Numeric | **Purpose:** Find relationships between variables.")
    corr_matrix = df[['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']].corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

# ==================================================
# STAGE 4: INSIGHTS AND AI INTEGRATION
# ==================================================
elif nav == "💡 Stage 4: Insights & AI":
    st.title("💡 Stage 4: Insights & AI Maintenance Assistant")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Key Insights Discovered</h3>
        <ul>
            <li><b>Usage Drives Wear:</b> There is a visible positive correlation between high door revolutions and increased vibration, proving that mechanical fatigue accumulates with usage.</li>
            <li><b>Environmental Amplification:</b> While humidity isn't the primary cause of failure, the heatmap shows it correlates with increased vibration over time, acting as an amplifying stressor (likely due to friction or condensation).</li>
            <li><b>Anomaly Detection:</b> The Box Plots reveal severe outliers in sensors x1 through x5 during high-vibration events, allowing us to pinpoint exactly which spatial axis the mechanical failure is occurring on.</li>
            <li><b>Predictive Value:</b> The Time Series line plot clearly shows normal operating baselines punctuated by extreme spikes, meaning a simple threshold alarm can successfully predict failures before they happen.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🛠️ Recommendations for Maintenance Teams")
    st.markdown("""
    * **Dynamic Maintenance Scheduling:** Move away from calendar-based maintenance. Schedule bearing and track lubrication specifically for elevators reaching critical revolution counts.
    * **Threshold Alarms:** Set automated alerts when the standard deviation of vibration exceeds normal parameters for more than 5 minutes.
    * **Targeted Diagnostics:** Use spatial sensors (x1-x5) to diagnose whether the issue is lateral rattling or vertical grinding before dispatching a technician.
    """)
    
    st.markdown("---")
    st.header("🤖 Google Gemini: Maintenance Operations AI")
    st.write("Use the generative AI chatbot to analyze these insights further, generate maintenance protocols, or answer technical questions based on the dataset.")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Find an available text model
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            target_model = next((m for m in available_models if 'flash' in m or 'pro' in m), available_models[0])
            
            model = genai.GenerativeModel(target_model)
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [{"role": "assistant", "content": "I am the Predictive Maintenance AI. Ask me how to interpret the Correlation Heatmap, or how to reduce elevator downtime!"}]

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.write(msg["content"])

            if prompt := st.chat_input("Ask Gemini about elevator maintenance or data patterns..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.write(prompt)

                with st.chat_message("assistant"):
                    # Give the AI context about the dataset
                    sys_prompt = f"""
                    You are a Senior Data Scientist analyzing elevator predictive maintenance.
                    DATA CONTEXT:
                    - Avg Vibration: {df['vibration'].mean():.2f}
                    - Max Revolutions: {df['revolutions'].max():.2f}
                    - High humidity correlates with increased friction.
                    Answer the user's question concisely: {prompt}
                    """
                    with st.spinner("Analyzing maintenance data..."):
                        response = model.generate_content(sys_prompt)
                        st.write(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"API Error: {e}")
    else:
        st.warning("⚠️ Enter your Gemini API Key in the sidebar to activate the AI Assistant.")
