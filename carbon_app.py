import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ICADS | GenAI Carbon Manager", layout="wide")

# --- 🔑 API KEY SETUP (HARDCODED) ---
# PASTE YOUR KEY INSIDE THE QUOTES BELOW
GEMINI_API_KEY = "AIzaSyCic8-HaVQJrKJRuHnOETA0m0MAeE1A6ms" 

# Configure the AI immediately
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"API Key Error: {e}")

# --- SECTION 1: DATA LOADING & AI ENGINE ---
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_resource
def build_ai_engine(user_file=None):
    # 1. DATA LOADING
    if user_file is not None:
        # OPTION A: Load User Data
        try:
            df = pd.read_csv(user_file)
            # Ensure required columns exist
            required_cols = ['Rotational_Speed_rpm', 'Torque_Nm', 'Air_Temp_K', 'Process_Temp_K']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Error: Your CSV must contain these columns: {required_cols}")
                return None, None, None
            
            # If user data doesn't have 'Tool_Wear_min', we simulate it for the graph
            if 'Tool_Wear_min' not in df.columns:
                 df['Tool_Wear_min'] = np.linspace(0, 260, len(df))
            st.sidebar.success("✅ Custom Data Loaded!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None, None
             
    else:
        # OPTION B: Simulation (Default)
        np.random.seed(42)
        n_samples = 3000
        data = {
            'Rotational_Speed_rpm': np.random.normal(1500, 250, n_samples), 
            'Torque_Nm': np.random.normal(40, 10, n_samples), 
            'Air_Temp_K': np.random.normal(300, 2, n_samples),
            'Process_Temp_K': np.random.normal(310, 1, n_samples),
            'Tool_Wear_min': np.linspace(0, 260, n_samples) 
        }
        df = pd.DataFrame(data)

    # 2. PHYSICS & WASTE CALCULATION
    # We calculate what power *should* be based on physics
    df['Physics_Power_kW'] = (df['Torque_Nm'] * df['Rotational_Speed_rpm']) / 9549
    
    # If user didn't provide Actual Power, we simulate the waste/efficiency loss
    if 'Actual_Power_kW' not in df.columns:
        def apply_inefficiency(row):
            waste = 0
            if row['Tool_Wear_min'] > 200:
                waste = row['Physics_Power_kW'] * np.random.uniform(0.15, 0.25)
            return row['Physics_Power_kW'] + waste + np.random.normal(0, 0.02)
        df['Actual_Power_kW'] = df.apply(apply_inefficiency, axis=1)

    # 3. AI MODEL TRAINING
    # We train on the "Cleanest" 50% of the data to learn the baseline
    train_mask = df['Tool_Wear_min'] < 150
    df_train = df[train_mask]
    
    # Fallback for small datasets
    if len(df_train) < 10: 
        df_train = df 
        
    features = ['Rotational_Speed_rpm', 'Torque_Nm', 'Air_Temp_K', 'Process_Temp_K']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_train[features], df_train['Actual_Power_kW'])
    
    return model, df, features

# Pass the uploaded file to the engine
model, df, features = build_ai_engine(uploaded_file)

# Stop app if data failed to load
if df is None:
    st.stop()

# Run Predictions
df['AI_Predicted_Ideal_kW'] = model.predict(df[features])
df['Energy_Waste_kWh'] = df['Actual_Power_kW'] - df['AI_Predicted_Ideal_kW']
df['Energy_Waste_kWh'] = df['Energy_Waste_kWh'].clip(lower=0) # No negative waste

# --- SECTION 2: UI SIDEBAR ---
st.sidebar.divider()
st.sidebar.header("⚙️ Controls")
elec_cost = st.sidebar.slider("Electricity Cost ($/kWh)", 0.10, 0.50, 0.18)
emission_factor = 0.475

# Main Dashboard Header
st.title("🏭 ICADS: Generative AI Carbon Manager")

# Calculate Metrics
total_waste = df['Energy_Waste_kWh'].sum()
savings = total_waste * elec_cost
carbon_waste = total_waste * emission_factor

# Top Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Waste (kWh)", f"{total_waste:,.0f}", delta_color="inverse")
c2.metric("Potential Savings", f"${savings:,.2f}", delta_color="normal")
c3.metric("Avoidable Carbon", f"{carbon_waste:,.0f} kgCO2e", delta_color="inverse")

st.divider()

# --- SECTION 3: THE SIMPLIFIED GRAPH ---
st.subheader("📉 Visualizing the 'Carbon Gap'")

# Group data for smooth plotting
df['group'] = pd.cut(df['Tool_Wear_min'], bins=50)
chart_data = df.groupby('group', observed=True)[['Actual_Power_kW', 'AI_Predicted_Ideal_kW', 'Tool_Wear_min']].mean()

fig, ax = plt.subplots(figsize=(10, 4))

# Plot 1: The AI Baseline (Green)
ax.plot(chart_data['Tool_Wear_min'], chart_data['AI_Predicted_Ideal_kW'], 
        color='green', linestyle='--', linewidth=2, label='AI Ideal Baseline (Target)')

# Plot 2: The Actual (Red)
ax.plot(chart_data['Tool_Wear_min'], chart_data['Actual_Power_kW'], 
        color='red', linewidth=2, label='Actual Power (Current)')

# Plot 3: The Waste Zone (Shaded Area)
ax.fill_between(chart_data['Tool_Wear_min'], 
                chart_data['AI_Predicted_Ideal_kW'], 
                chart_data['Actual_Power_kW'], 
                color='red', alpha=0.1, label='Carbon Waste Zone')

ax.set_title("Forensic Analysis: When did the machine start polluting?")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Tool Wear (Minutes)")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.info("ℹ️ **Note:** The 'Red Mist' area represents pure waste. Notice how the lines separate after 200 minutes of wear.")

# --- SECTION 4: GEMINI INTEGRATION ---
st.divider()
st.subheader("🧠 Ask Gemini: How do we fix this?")

col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown("**Analyze this anomaly:**")
    st.write("The system will send this summary to Gemini:")
    st.code(f"""
    Status: High Emissions Detected
    Tool Wear: > 200 mins
    Avg Efficiency Loss: 22%
    Current Temp: {df['Process_Temp_K'].mean():.1f} K
    """, language="yaml")
    
    generate_btn = st.button("✨ Generate Recommendation", type="primary")

with col_b:
    if generate_btn:
        # Check if user forgot to paste the key
        if "PASTE_YOUR_LONG_KEY" in GEMINI_API_KEY:
            st.error("⚠️ You need to paste your real API key in the code at line 16!")
        else:
            with st.spinner("Consulting Gemini AI expert..."):
                try:
                    # Use the stable model name
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    prompt = f"""
                    Act as a Senior Manufacturing Sustainability Engineer. 
                    I have detected a carbon emission anomaly in a milling machine.
                    
                    DATA:
                    - The machine has exceeded 200 minutes of tool wear.
                    - Energy consumption is 22% higher than the AI baseline.
                    - Process temperature is averaging {df['Process_Temp_K'].mean():.1f} Kelvin.
                    
                    Please provide:
                    1. A diagnosis of the physical root cause.
                    2. Three specific maintenance actions to reduce the carbon footprint immediately.
                    3. An estimate of the long-term risk if ignored.
                    
                    Keep the tone professional and actionable.
                    """
                    
                    response = model.generate_content(prompt)
                    st.success("Analysis Complete")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Error connecting to Gemini: {e}")