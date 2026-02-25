import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px  # Added for more attractive visualizations

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
        st.success(f"Dataset loaded successfully! | {len(df):,} rows × {len(df.columns)} columns")
        
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
                st.bar_chart(null_data[null_data > 0], color="#ff4b4b")
            else:
                st.success("Zero nulls remaining!")

        st.divider()

        # --- STEP 3: PREPROCESS DATA (Milestone 2 Updates) ---
        st.markdown("### ⚙️ Step 3 • Preprocess Data")
        if st.button("Run Preprocessing"):
            with st.status("Preprocessing Log", expanded=True):
                # Time Normalization
                st.write("🕒 Parsing Date column and extracting Time-Series features...")
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # Feature Extraction (Milestone 2)
                df['Day'] = df['Date'].dt.day_name()
                df['Month'] = df['Date'].dt.month_name()
                df['Is_Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
                
                # Handling Nulls (Deep Cleaning)
                st.write("🔧 Cleaning and interpolating metrics...")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
                
                # Final safety fill for numeric and categorical
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                cat_cols = df.select_dtypes(include=['object']).columns
                df[cat_cols] = df[cat_cols].fillna("Not Recorded")
                
                # Anomaly Detection Logic (Milestone 2)
                st.write("🚨 Detecting Health Anomalies...")
                # Rule-based: High Heart Rate
                if 'Heart_Rate' in df.columns:
                    df['HR_Anomaly'] = df['Heart_Rate'].apply(lambda x: 1 if x > 120 or x < 50 else 0)
                
                # Statistical-based: Z-Score for Sleep Duration
                if 'Sleep_Duration' in df.columns:
                    mean_sleep = df['Sleep_Duration'].mean()
                    std_sleep = df['Sleep_Duration'].std()
                    df['Sleep_Anomaly'] = df['Sleep_Duration'].apply(
                        lambda x: 1 if (x < mean_sleep - 2*std_sleep) or (x > mean_sleep + 2*std_sleep) else 0
                    )
                
                st.session_state['cleaned_df'] = df
                st.write("✅ Preprocessing & Feature Extraction Complete")

        st.divider()

        # --- STEP 4: PREVIEW CLEANED DATASET ---
        st.markdown("### 👁️ Step 4 • Preview Cleaned Dataset")
        if 'cleaned_df' in st.session_state:
            if st.button("Preview Cleaned Data"):
                cleaned_data = st.session_state['cleaned_df']
                st.dataframe(cleaned_data.head(20), use_container_width=True)
                
                # Final Verification for User
                null_check = cleaned_data.isnull().sum().sum()
                if null_check == 0:
                    st.success("✅ Verification: 0 Null Values found in cleaned dataset.")
                
                # Download Button
                csv = cleaned_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name="FitPulse_Cleaned_Data_Pro.csv",
                    mime="text/csv"
                )
        else:
            st.info("Run Preprocessing first to preview cleaned data.")

        st.divider()

        # --- STEP 5: EXPLORATORY DATA ANALYSIS (Attractive Visuals) ---
        st.markdown("### 📊 Step 5 • Exploratory Data Analysis")
        if st.button("Run Full EDA"):
            data = st.session_state.get('cleaned_df', df)
            
            # 1. Anomaly Summary Cards
            st.markdown("#### Health Anomaly Insights")
            col_a, col_b = st.columns(2)
            if 'HR_Anomaly' in data.columns:
                hr_count = data['HR_Anomaly'].sum()
                col_a.metric("Abnormal Heart Rate Flags", hr_count, delta="Alerts Found", delta_color="inverse")
            if 'Sleep_Anomaly' in data.columns:
                sl_count = data['Sleep_Anomaly'].sum()
                col_b.metric("Sleep Pattern Anomalies", sl_count, delta="Alerts Found", delta_color="inverse")

            # 2. Interactive Distribution Plot
            st.markdown("#### Feature Distribution")
            numeric_options = data.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Select metric to view distribution:", numeric_options)
            fig_dist = px.histogram(data, x=selected_col, nbins=30, color_discrete_sequence=['#6a11cb'], marginal="box")
            fig_dist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # 3. Correlation Heatmap
            st.markdown("#### Metric Correlation Matrix")
            corr = data.select_dtypes(include=[np.number]).corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    else:
        st.info("Please upload your Fitness CSV file to begin the pipeline.")

if __name__ == "__main__":
    main()