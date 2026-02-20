import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import time

# ==========================================
# 1. MALL OPERATIONS UI & CSS
# ==========================================
st.set_page_config(page_title="Mall Operations | Smart Elevator", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .demo-box { background-color: #0f291e; padding: 20px; border-radius: 10px; border-left: 5px solid #2ea043; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE: MAPPING CSV TO MALL REALITY
# ==========================================
@st.cache_data
def load_mall_data():
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv").dropna(subset=['vibration']).copy()
    
    # MAPPING 1: Floors (Using vertical sensor X3)
    # X3 ranges from ~0.2 to ~1.2 in the data. Let's slice it into 4 Mall Floors.
    conditions = [
        (df['x3'] < 0.5),
        (df['x3'] >= 0.5) & (df['x3'] < 0.75),
        (df['x3'] >= 0.75) & (df['x3'] < 1.0),
        (df['x3'] >= 1.0)
    ]
    floors = ['Ground Floor (Food Court)', 'Floor 1 (Apparel)', 'Floor 2 (Electronics)', 'Floor 3 (Cinema)']
    df['Mall_Floor'] = np.select(conditions, floors, default='Unknown')
    
    # MAPPING 2: Passenger Load Estimate (Using Revolutions)
    df['Estimated_Passengers'] = (df['revolutions'] / 10).astype(int).clip(0, 15)
    
    # MAPPING 3: Energy Consumption Score
    df['Energy_Draw_kW'] = (df['revolutions'] * 0.5) + (df['vibration'] * 0.2)
    
    # Rolling averages for smooth tracking
    df['vibration_smooth'] = df['vibration'].rolling(20).mean().fillna(df['vibration'])
    
    return df

try:
    df = load_mall_data()
except Exception as e:
    st.error(f"Dataset missing! Please upload the CSV. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: SECURITY & OPERATIONS
# ==========================================
with st.sidebar:
    st.title("🛍️ Mall Operations")
    st.image("https://cdn-icons-png.flaticon.com/512/3030/3030245.png", width=80)
    st.markdown("---")
    
    nav = st.radio("Access Terminals", [
        "📺 App Demo & Walkthrough",
        "🏢 3D Mall Shaft (Live Anim)", 
        "📊 Floor & Shopper Analytics", 
        "⚡ Energy & Motor Health",
        "👮 AI Security Chief"
    ])
    
    st.markdown("---")
    vib_alert = st.slider("Max Mall Safety Vibration", 10, 100, 35)

# ==========================================
# MODULE 1: APP DEMO & WALKTHROUGH
# ==========================================
if nav == "📺 App Demo & Walkthrough":
    st.title("Welcome to the Mall Elevator Control Room")
    
    st.markdown("""
    <div class="demo-box">
        <h3>🚀 Interactive Demonstration</h3>
        <p>This application transforms raw mathematical CSV data into a real-world shopping mall scenario. Here is how the AI interprets your dataset:</p>
        <ul>
            <li><b>The Floors:</b> Sensor <code>X3</code> measures vertical height. We mapped its coordinates to actual mall floors (Ground to Cinema).</li>
            <li><b>The Passengers:</b> <code>Revolutions</code> dictate motor strain. High strain means the car is full of shoppers.</li>
            <li><b>The Health:</b> <code>Vibration</code> determines if the doors or tracks are jamming.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Live Operations Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Traffic", f"{df['Estimated_Passengers'].mean():.0f} pax/trip")
    c2.metric("Busiest Floor", df['Mall_Floor'].mode()[0])
    c3.metric("Avg Shaft Humidity", f"{df['humidity'].mean():.1f}%")
    c4.metric("Critical Alerts", len(df[df['vibration'] > vib_alert]))

    st.subheader("Simulate Live Weekend Traffic")
    if st.button("▶️ Start Weekend Traffic Simulator"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        for i in range(100, 500, 10):
            progress_bar.progress(int((i/500)*100))
            sim_data = df.iloc[i-100:i]
            status_text.write(f"**Tracking Trip ID #{i}** | Current Floor: {sim_data['Mall_Floor'].iloc[-1]}")
            
            fig = px.bar(sim_data, x='ID', y='vibration', color='Mall_Floor', template="plotly_dark")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
        st.success("Simulation Complete!")

# ==========================================
# MODULE 2: 3D MALL SHAFT (WORKING ELEVATOR)
# ==========================================
elif nav == "🏢 3D Mall Shaft (Live Anim)":
    st.title("🏢 3D Mall Shaft Digital Twin")
    st.write("Watch the elevator navigate the shopping center. Press **Play** below the chart to animate the car moving between the Food Court and the Cinema.")
    
    # We slice 100 chronological rows to animate a complete journey
    anim_df = df.iloc[2000:2150].copy()
    anim_df['TimeFrame'] = range(len(anim_df))
    
    fig_3d = px.scatter_3d(
        anim_df, x='x1', y='x2', z='x3',
        animation_frame='TimeFrame',
        color='vibration',
        size='Estimated_Passengers', size_max=40,
        color_continuous_scale='Plasma',
        range_z=[0, 1.5], range_x=[df['x1'].min(), df['x1'].max()], range_y=[df['x2'].min(), df['x2'].max()],
        template="plotly_dark", height=700
    )
    
    # Drawing Virtual "Mall Floors"
    for z_val, f_name in zip([0.3, 0.6, 0.85, 1.1], ['Ground (Food)', 'Floor 1', 'Floor 2', 'Floor 3 (Cinema)']):
        fig_3d.add_trace(go.Surface(
            x=[[90, 170], [90, 170]], y=[[-60, -60], [25, 25]], z=[[z_val, z_val], [z_val, z_val]],
            opacity=0.2, colorscale='Greens', showscale=False, name=f_name
        ))
        
    fig_3d.update_layout(scene=dict(zaxis_title='Vertical Height (Floors)'))
    st.plotly_chart(fig_3d, use_container_width=True)
    st.info("🛒 The sphere size changes based on passenger load. Color changes based on mechanical vibration.")

# ==========================================
# MODULE 3: FLOOR & SHOPPER ANALYTICS
# ==========================================
elif nav == "📊 Floor & Shopper Analytics":
    st.title("📊 Shopper Traffic & Floor Health")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Which Floor Causes the Most Vibration?")
        floor_health = df.groupby('Mall_Floor')['vibration'].mean().reset_index()
        fig_bar = px.bar(floor_health, x='Mall_Floor', y='vibration', color='vibration', template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.subheader("Passenger Load vs Vibration")
        st.write("Does a heavier car vibrate more?")
        fig_scatter = px.box(df.sample(2000), x='Estimated_Passengers', y='vibration', color='Estimated_Passengers', template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# MODULE 4: ENERGY & MOTOR HEALTH
# ==========================================
elif nav == "⚡ Energy & Motor Health":
    st.title("⚡ Energy Consumption & Door Sensors")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Motor Energy Draw (kW)")
        fig_energy = px.area(df.iloc[::30], x='ID', y='Energy_Draw_kW', template="plotly_dark", color_discrete_sequence=['#ff9900'])
        st.plotly_chart(fig_energy, use_container_width=True)
        
    with c2:
        st.subheader("Door Mechanism Diagnostics (X4 & X5)")
        st.write("Analyzing internal sensor variance during door opening sequences.")
        fig_doors = px.scatter(df.sample(2000), x='x4', y='x5', color='vibration', size='revolutions', template="plotly_dark")
        st.plotly_chart(fig_doors, use_container_width=True)

# ==========================================
# MODULE 5: AI SECURITY CHIEF
# ==========================================
elif nav == "👮 AI Security Chief":
    st.title("👮 Gemini Security & Operations AI")
    st.write("Ask the AI Mall Chief about elevator maintenance, crowd control, or floor analytics.")
    
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🔑 Please add your `GOOGLE_API_KEY` to Streamlit Secrets!")
    else:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "Mall Control AI online. How can I assist with the elevator fleet today?"}]

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("E.g., Why is Floor 3 vibrating the most?"):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                sys_prompt = f"Context: Mall Elevator Data. Avg Vib: {df['vibration'].mean():.1f}. Busiest Floor: {df['Mall_Floor'].mode()[0]}. Energy Draw: {df['Energy_Draw_kW'].mean():.1f}kW. User says: {prompt}"
                response = model.generate_content(sys_prompt)
                st.write(response.text)
                st.session_state.chat.append({"role": "assistant", "content": response.text})
