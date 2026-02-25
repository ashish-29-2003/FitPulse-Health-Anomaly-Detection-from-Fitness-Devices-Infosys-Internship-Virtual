import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fitness Data Pro | Pipeline",
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
    /* Style for Step Containers */
    .step-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #464b5d;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏋️ Fitness Health Data — Pro Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("Automated Preprocessing and Anomaly Detection (Milestone 2)")
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

        # --- STEP 3: PREPROCESS DATA ---
        st.markdown("### ⚙️ Step 3 • Preprocess Data")
        if st.button("Run Preprocessing"):
            with st.status("Cleaning and Extracting Features...", expanded=True):
                # 1. Date Cleaning (CRITICAL: FIXES NULLS IN STEP 4)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Date'] = df['Date'].ffill().bfill() 
                
                # 2. Feature Extraction (Milestone 2)
                df['Day_Name'] = df['Date'].dt.day_name()
                df['Is_Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
                
                # 3. Handle Metric Nulls (Interpolation)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
                
                # 4. Anomaly Detection Logic
                if 'Heart_Rate' in df.columns:
                    df['HR_Anomaly'] = df['Heart_Rate'].apply(lambda x: 1 if x > 120 or x < 50 else 0)

                st.session_state['cleaned_df'] = df.copy() 
                st.write("✅ Preprocessing Complete!")
        st.divider()

        # --- STEP 4: PREVIEW CLEANED DATASET ---
        st.markdown("### 👁️ Step 4 • Preview Cleaned Dataset")
        if 'cleaned_df' in st.session_state:
            cleaned_view = st.session_state['cleaned_df']
            st.dataframe(cleaned_view.head(20), use_container_width=True)
            
            # Verify 0 Nulls
            if cleaned_view.isnull().sum().sum() == 0:
                st.success("✅ Cleaned Dataset Verified: 0 Null Values (including Date).")
            
            csv = cleaned_view.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Cleaned CSV", data=csv, file_name="Cleaned_Fitness_Data.csv")
        else:
            st.info("Run Step 3 first.")
        st.divider()

        # --- STEP 5: EXPLORATORY DATA ANALYSIS (EDA) ---
        st.markdown("### 📊 Step 5 • Exploratory Data Analysis")
        # Fix: Pull data directly from state to prevent refresh loops
        if 'cleaned_df' in st.session_state:
            eda_data = st.session_state['cleaned_df']
            
            # Clean Trend Visualization
            st.markdown("#### Health Metric Trends")
            num_cols = eda_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove anomaly flags from trend list to keep it clean
            clean_num_cols = [c for c in num_cols if 'Anomaly' not in c]
            
            selected_metrics = st.multiselect("Select metrics to compare:", clean_num_cols, default=clean_num_cols[:1])
            
            if selected_metrics:
                # Use a subset of data (e.g., first 150 rows) for a cleaner graph
                fig_trend = px.line(eda_data.head(150), x='Date', y=selected_metrics, 
                                    markers=True, template="plotly_dark",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_trend.update_layout(hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_trend, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if 'Workout_Type' in eda_data.columns:
                    st.markdown("#### Workout Distribution")
                    fig_pie = px.pie(eda_data, names='Workout_Type', hole=0.4, 
                                     color_discrete_sequence=px.colors.qualitative.Safe)
                    fig_pie.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("#### Feature Correlation")
                corr = eda_data[clean_num_cols].corr()
                fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", 
                                     color_continuous_scale='RdBu_r', template="plotly_dark")
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Run Preprocessing to unlock Step 5.")

    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()