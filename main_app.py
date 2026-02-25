import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fitness Data Pro | Pipeline",
    page_icon="🏆",
    layout="wide"
)

# --- USER-FRIENDLY CSS THEME ---
st.markdown("""
    <style>
    /* Professional Dark Theme */
    .stApp {
        background-color: #1a1b2e;
        color: #ffffff;
    }
    
    /* Elegant Card Design for Steps */
    .step-card {
        background-color: #2a2b45;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 30px;
        border: 1px solid #3d3e5a;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Gradient Headers */
    .main-header {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
    }

    /* Button Customization */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        padding: 12px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏋️ Fitness Health Data — Pro Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("Upload your fitness tracking CSV and let the pipeline preprocess, clean, and explore your data.")
    st.divider()

    # --- STEP 1: UPLOAD DATASET ---
    st.markdown("### 📁 Step 1 • Upload Dataset")
    with st.container():
        uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])
        
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Fitness_Health_Tracking_Dataset.csv loaded successfully! | {len(df):,} rows × {len(df.columns)} columns")
        
        # Dashboard Overview
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Total Nulls", df.isnull().sum().sum())

        st.divider()

        # --- STEP 2: CHECK NULL VALUES ---
        st.markdown("### 🔍 Step 2 • Check Null Values")
        if st.button("Check Null Values"):
            null_data = df.isnull().sum()
            if null_data.sum() > 0:
                st.warning("Null Values Detected")
                # Using a bar chart to visualize nulls (mimicking video reference)
                st.bar_chart(null_data[null_data > 0], color="#ff4b4b")
            else:
                st.success("Zero nulls remaining!")

        st.divider()

        # --- STEP 3: PREPROCESS DATA ---
        st.markdown("### ⚙️ Step 3 • Preprocess Data")
        if st.button("Run Preprocessing"):
            with st.status("Preprocessing Log", expanded=True):
                # Time Normalization
                st.write("🕒 Parsing Date column to datetime...")
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Handling Nulls
                st.write("🔧 Interpolating (linear) + ffill/bfill null values in key metrics...")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
                
                # Categorical Cleaning
                if 'Workout_Type' in df.columns:
                    st.write("📝 Filled Workout_Type nulls -> 'No Workout'")
                    df['Workout_Type'] = df['Workout_Type'].fillna("No Workout")
                
                st.session_state['cleaned_df'] = df
                st.write("✅ Preprocessing Complete")

        st.divider()

        # --- STEP 4: PREVIEW CLEANED DATASET ---
        st.markdown("### 👁️ Step 4 • Preview Cleaned Dataset")
        if 'cleaned_df' in st.session_state:
            if st.button("Preview Cleaned Data"):
                st.dataframe(st.session_state['cleaned_df'].head(20), use_container_width=True)
                
                # Download Button
                csv = st.session_state['cleaned_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name="FitPulse_Cleaned_Data.csv",
                    mime="text/csv"
                )
        else:
            st.info("Run Preprocessing first to preview cleaned data.")

        st.divider()

        # --- STEP 5: EXPLORATORY DATA ANALYSIS ---
        st.markdown("### 📊 Step 5 • Exploratory Data Analysis")
        if st.button("Run Full EDA"):
            data = st.session_state.get('cleaned_df', df)
            
            # Numeric Distributions
            st.markdown("#### Distribution of Numeric Features")
            st.line_chart(data.select_dtypes(include=[np.number]).head(100))
            
            # Correlation Matrix
            st.markdown("#### Correlation Heatmap")
            corr = data.select_dtypes(include=[np.number]).corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    else:
        # Initial Landing Screen
        st.info("Please upload your Fitness CSV file to begin the pipeline.")

if __name__ == "__main__":
    main()