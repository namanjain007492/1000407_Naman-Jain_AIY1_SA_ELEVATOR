🏢 Elevator AI Operations
Smart Predictive Maintenance System using Physics, Statistics & AI
📌 Project Overview

Elevator AI Operations is an advanced predictive maintenance dashboard built using Streamlit, combining:

📊 Data Science

🧮 Mathematical Modeling

⚙ Mechanical Engineering Concepts

🤖 Generative AI Integration

📈 Real-Time Anomaly Detection

🧊 3D Physics Simulation

The system monitors elevator telemetry data (vibration, humidity, revolutions, spatial sensors) to predict mechanical wear before catastrophic failure occurs.

This project simulates a real-world Condition-Based Predictive Maintenance (PdM) system used by global elevator manufacturers.

🎯 Core Objectives

Detect abnormal vibration patterns

Model mechanical wear mathematically

Simulate elevator physics under passenger load

Classify risk levels using statistical thresholds

Estimate Remaining Useful Life (RUL)

Provide AI-generated operational insights

Create a digital twin of an elevator system

🧠 System Architecture
1️⃣ Data Processing Engine

Removes missing values

Eliminates duplicates

Prepares structured telemetry data

Calculates cumulative mechanical stress using numerical integration

2️⃣ Mathematical Modeling

Ideal motor modeled as a sine wave

Real vibration compared against harmonic baseline

Area Under Curve (AUC) used to estimate total wear

3️⃣ Telemetry Visualization

Time series vibration tracking

Stress factor histograms

Correlation heatmap

Spatial outlier detection

Scatter plots for usage vs degradation

4️⃣ 3D Physics Simulator

Simulates vertical travel between floors

Passenger weight affects speed

Mechanical strain increases with load

Real-time 3D elevator visualization

5️⃣ AI Operations Assistant

Integrated with Google Gemini API

Context-aware maintenance analysis

Chat log export system

6️⃣ Statistical Risk Classification

Uses rolling mean & standard deviation:

🟢 Normal (≤ 1σ)

🟠 Warning (2σ–3σ)

🔴 Critical (>3σ)

7️⃣ Real-Time Anomaly Detection Engine

Rolling statistical thresholds

Automatic anomaly marking

Live health scoring

Remaining Useful Life estimation

📊 Mathematical Foundations
Harmonic Motion Model
𝑃
(
𝑡
)
=
𝑃
0
+
𝐴
sin
⁡
(
2
𝜋
𝑓
𝑡
)
P(t)=P
0
	​

+Asin(2πft)
Numerical Integration (Wear Estimation)
𝑊
𝑒
𝑎
𝑟
=
∫
𝑉
𝑖
𝑏
𝑟
𝑎
𝑡
𝑖
𝑜
𝑛
(
𝑡
)
𝑑
𝑡
Wear=∫Vibration(t)dt
Statistical Thresholding
𝑇
ℎ
𝑟
𝑒
𝑠
ℎ
𝑜
𝑙
𝑑
=
𝜇
+
3
𝜎
Threshold=μ+3σ
Physics Modeling

Based on Newton’s Second Law:

𝐹
=
𝑚
𝑎
F=ma
🏗 Industrial Relevance

This system mirrors predictive maintenance platforms used by:

Otis (Otis ONE™)

KONE

ThyssenKrupp MAX

Industrial predictive systems reduce:

50% unplanned downtime

20% component wear

Millions in operational costs

🌱 Sustainability Impact

Reduces emergency breakdowns

Extends mechanical lifespan

Minimizes energy waste

Lowers carbon footprint

Supports ESG goals

⚙ Tech Stack

Python

Streamlit

Pandas

NumPy

Plotly

SciPy

Google Generative AI (Gemini API)

🏆 Academic & Engineering Value

This project demonstrates:

Integration of mathematics with engineering

Real-world industrial simulation

Statistical modeling expertise

Applied physics knowledge

AI integration skills

Data visualization mastery

It represents a complete digital twin predictive system.

👨‍💻 Credits

Name : Naman Jain
Class : 11-IB

Elevator Predictive Maintenance Dataset
(Simulated industrial telemetry data)

📜 License

This project is developed for educational and research purposes.
