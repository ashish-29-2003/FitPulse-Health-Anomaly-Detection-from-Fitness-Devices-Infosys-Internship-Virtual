import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FitPulse",
    page_icon="🏆",
    layout="wide"
)

# --- USER-FRIENDLY CSS THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #1a1b2e; color: #ffffff; }
    .main-header {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
    }
    .stButton>button {
        width: 100%; border-radius: 8px;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white; border: none; padding: 12px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">FitPulse</h1>', unsafe_allow_html=True)
    st.markdown("Automated Preprocessing, Time-Series Normalization, and Feature Extraction.")
    st.divider()

    # --- STEP 1: UPLOAD DATASET ---
    st.markdown("### 📁 Step 1 • Upload Dataset")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])
        
    if uploaded_file:
        if 'raw_df' not in st.session_state:
            st.session_state['raw_df'] = pd.read_csv(uploaded_file)
        
        df = st.session_state['raw_df']
        st.success(f"Dataset loaded: {len(df):,} rows")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Initial Nulls", df.isnull().sum().sum())
        st.divider()

    # --- STEP 2: CHECK NULL VALUES ---
        st.markdown("### 🔍 Step 2 • Check Null Values")
        if st.button("Check Null Values"):
            null_data = df.isnull().sum()
            if null_data.sum() > 0:
                st.warning("Null Values Detected")
                st.bar_chart(null_data[null_data > 0], color="#ff4b4b")
            else:
                st.success("Zero nulls detected!")
        st.divider()

    # --- STEP 3: PREPROCESS & NORMALIZE DATA (Merged) ---
        st.markdown("### ⚙️ Step 3 • Preprocess & Normalize Data")
        if st.button("Run Preprocessing Pipeline"):
            with st.status("Cleaning, Normalizing, and Extracting Features...", expanded=True):
                # 1. Date Cleaning (Fixes Nulls for Step 4)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Date'] = df['Date'].ffill().bfill() 
                
                # 2. Resampling/Normalization (Milestone 1 Requirement) 
                st.write("⏳ Resampling data to hourly intervals for consistency...")
                df_sorted = df.sort_values('Date')
                # Resampling to 1-hour granularity 
                resampled_df = df_sorted.set_index('Date').resample('H').mean(numeric_only=True).reset_index()
                st.session_state['resampled_log'] = resampled_df.head(50) 

                # 3. Feature Extraction (Milestone 2 Requirement) [cite: 26, 66]
                st.write("🧪 Extracting time-series features (Day, Weekend status)...")
                df['Day_Name'] = df['Date'].dt.day_name()
                df['Is_Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
                
                # 4. Handle Metric Nulls (Interpolation) 
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
                
                # 5. Rule-based Anomaly Detection [cite: 31, 85]
                if 'Heart_Rate' in df.columns:
                    df['HR_Anomaly'] = df['Heart_Rate'].apply(lambda x: 1 if x > 120 or x < 50 else 0)

                st.session_state['cleaned_df'] = df.copy() 
                st.write("✅ Preprocessing & Normalization Complete!")
                
                # Display the Normalized Log immediately within the status or below
                st.write("**Time-Normalized Data Log (Hourly Granularity):**")
                st.dataframe(st.session_state['resampled_log'], use_container_width=True)

        st.divider()

    # --- STEP 4: PREVIEW CLEANED DATASET ---
        st.markdown("### 👁️ Step 4 • Preview Cleaned Dataset")
        if 'cleaned_df' in st.session_state:
            cleaned_view = st.session_state['cleaned_df']
            st.dataframe(cleaned_view.head(20), use_container_width=True)
            if cleaned_view.isnull().sum().sum() == 0:
                st.success("✅ Cleaned Dataset Verified: 0 Null Values remaining.")
        else:
            st.info("Run Step 3 first.")
        st.divider()

    # --- STEP 5: EXPLORATORY DATA ANALYSIS (EDA) ---
        st.markdown("### 📊 Step 5 • Exploratory Data Analysis")
        if 'cleaned_df' in st.session_state:
            eda_data = st.session_state['cleaned_df']
            num_cols = [c for c in eda_data.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c]
            
            selected_metrics = st.multiselect("Select health metrics to view trends:", num_cols, default=num_cols[:1])
            if selected_metrics:
                # Cleaned visualization limited to 150 records for clarity
                fig_trend = px.line(eda_data.head(150), x='Date', y=selected_metrics, 
                                    template="plotly_dark", markers=True)
                fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Run Preprocessing to unlock Step 5.")

    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()