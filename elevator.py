import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# ==========================================
# PAGE CONFIGURATION (Premium Look)
# ==========================================
st.set_page_config(
    page_title="Smarter Elevator Maintenance",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DATA LOADING & CLEANING (Stage 2)
# ==========================================
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")
    
    # Store raw length
    raw_len = len(df)
    
    # Clean Data: Drop nulls in the target variable 'vibration'
    df_clean = df.dropna(subset=['vibration']).copy()
    
    # Drop duplicates if any
    df_clean = df_clean.drop_duplicates()
    
    return df, df_clean, raw_len

df_raw, df, raw_len = load_and_clean_data()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=80)
st.sidebar.title("Elevator OS")
page = st.sidebar.radio("Navigate", 
    ["📖 Project Context", "🧹 Data Overview", "📊 Visual Analytics", "💡 Key Insights", "🤖 AI Assistant (Premium)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Target Variable:** Vibration")
st.sidebar.markdown(f"**Cleaned Records:** {len(df):,}")

# ==========================================
# PAGE 1: PROJECT CONTEXT (Stage 1)
# ==========================================
if page == "📖 Project Context":
    st.title("🛗 Smarter Elevator Movement Visualization")
    st.markdown("""
    ### The Real-World Engineering Problem
    This platform addresses the **Predictive Maintenance of Elevators**. 
    Instead of waiting for an elevator door to jam or a motor to burn out, we use sensor data to identify micro-anomalies. 
    
    * **Vibration** is our primary "health indicator". Excessive shaking means heavy wear and tear.
    * **Humidity & Revolutions** act as "stress factors". We are investigating how environmental moisture and the sheer amount of mechanical movement force the elevator to vibrate out of normal ranges.
    
    **Why Companies Care:**
    1.  **Downtime Reduction:** Prevents elevators from trapping people.
    2.  **Cost Efficiency:** Predictable part replacements are cheaper than emergency disaster repairs.
    """)

# ==========================================
# PAGE 2: DATA OVERVIEW (Stage 2)
# ==========================================
elif page == "🧹 Data Overview":
    st.title("Dataset Understanding & Cleaning")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Rows", f"{raw_len:,}")
    col2.metric("Missing Vibration Values Removed", f"{raw_len - len(df):,}")
    col3.metric("Final Usable Rows", f"{len(df):,}")
    
    st.subheader("First 5 Rows of Cleaned Data")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ==========================================
# PAGE 3: VISUAL ANALYTICS (Stage 3)
# ==========================================
elif page == "📊 Visual Analytics":
    st.title("Data Visualization Hub")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", "📊 Distributions", "🟢 Scatter Analysis", "📦 Box Plots", "🔥 Correlation"
    ])
    
    # 1. Line Plot
    with tab1:
        st.subheader("Vibration Over Time")
        st.markdown("Observing how the health indicator behaves over sequential samples. *(Downsampled for performance)*")
        # Sampling every 20th row to keep the UI snappy
        fig_line = px.line(df.iloc[::20], x='ID', y='vibration', 
                           color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_line, use_container_width=True)
        
    # 2. Histogram
    with tab2:
        st.subheader("Distribution of Stress Factors")
        colA, colB = st.columns(2)
        with colA:
            fig_hist_hum = px.histogram(df, x='humidity', nbins=30, title="Humidity Distribution", color_discrete_sequence=['#3282B8'])
            st.plotly_chart(fig_hist_hum, use_container_width=True)
        with colB:
            fig_hist_rev = px.histogram(df, x='revolutions', nbins=30, title="Revolutions Distribution", color_discrete_sequence=['#0F4C75'])
            st.plotly_chart(fig_hist_rev, use_container_width=True)
            
    # 3. Scatter Plot
    with tab3:
        st.subheader("Revolutions vs Vibration")
        st.markdown("Do higher revolutions cause higher vibrations?")
        # Using a sample of 3000 points to prevent over-plotting
        fig_scatter = px.scatter(df.sample(3000, random_state=42), x='revolutions', y='vibration', 
                                 opacity=0.5, trendline="ols", color='humidity', color_continuous_scale='Viridis')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # 4. Box Plot
    with tab4:
        st.subheader("Sensor Readings Outlier Detection (x1 - x5)")
        # Melt dataframe to plot multiple boxplots easily
        df_melted = df[['x1', 'x2', 'x3', 'x4', 'x5']].melt(var_name='Sensor', value_name='Reading')
        fig_box = px.box(df_melted, x='Sensor', y='Reading', color='Sensor')
        st.plotly_chart(fig_box, use_container_width=True)
        
    # 5. Correlation Heatmap
    with tab5:
        st.subheader("Feature Correlation Heatmap")
        corr_matrix = df[['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# PAGE 4: INSIGHTS (Stage 4)
# ==========================================
elif page == "💡 Key Insights":
    st.title("Actionable Maintenance Insights")
    
    st.success("**Insight 1: Vibration Spikes Highlight Intermittent Friction**\nThe time-series analysis shows steady baselines mixed with sharp spikes. In real life, this implies sporadic mechanical blockages rather than total system failure.")
    
    st.warning("**Insight 2: Humidity Introduces Micro-Stresses**\nThe slight positive correlation between humidity and vibration indicates that environmental moisture causes tracks to expand or lubrication to degrade, making doors shudder.")
    
    st.info("**Insight 3: The Revolutions Paradox**\nContrary to basic assumptions, revolutions correlate slightly negatively with vibration. This means the elevator actually stabilizes and runs smoother at consistent speeds, while low-speed starting/stopping creates grinding.")
    
    st.markdown("### 🛠️ Recommendation for Teams")
    st.markdown("> **Establish Automated Alerts:** Maintenance teams should integrate an alert system that dispatches a technician purely for lubrication and track-cleaning whenever the rolling average of `vibration` exceeds a threshold of 40, or if local weather predicts extended periods of high humidity.")

# ==========================================
# PAGE 5: AI ASSISTANT (Premium Secret Feature)
# ==========================================
elif page == "🤖 AI Assistant (Premium)":
    st.title("Ask the Maintenance AI")
    st.markdown("Consult the AI regarding elevator repair codes, maintenance strategies, or statistical correlations.")
    
    # Securely fetching API key from secrets
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    
    if not api_key:
        st.error("🔒 Premium Feature Locked. Please add your OpenAI API Key to `.streamlit/secrets.toml` to activate the AI Assistant.")
    else:
        client = openai.OpenAI(api_key=api_key)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your predictive maintenance AI. How can I help you analyze the elevator data today?"}]

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about vibration thresholds or maintenance tips..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                )
                
                full_response = response.choices[0].message.content
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
