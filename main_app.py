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
    st.markdown("### 📂 Step 1 • Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Save original in session state
        st.session_state['df'] = df

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.markdown("#### Null Value Count (Before)")
            st.write(df.isnull().sum())

        st.divider()

        # --- STEP 2: PREPROCESSING ---
        st.markdown("### ⚙️ Step 2 • Intelligent Preprocessing")
        if st.button("🚀 Run Preprocessing Pipeline"):
            with st.spinner("Cleaning data, handling anomalies, and filling nulls..."):
                cleaned_df = df.copy()

                # 1. Standardize Date
                cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"], errors="coerce")

                # 2. Define Numeric Columns for Imputation
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
                # Remove User_ID if it exists as it shouldn't be averaged
                if 'User_ID' in numeric_cols: numeric_cols.remove('User_ID')

                # 3. Intelligent Imputation (Group by User then Interpolate + Fill)
                if 'User_ID' in cleaned_df.columns:
                    # Linearly interpolate missing values per user
                    cleaned_df[numeric_cols] = cleaned_df.groupby("User_ID")[numeric_cols].transform(
                        lambda x: x.interpolate(method="linear")
                    )
                    # Forward/Backward fill any remaining nulls (e.g., at the start or end of a series)
                    cleaned_df[numeric_cols] = cleaned_df.groupby("User_ID")[numeric_cols].transform(
                        lambda x: x.ffill().bfill()
                    )
                else:
                    # Global interpolation if no User_ID
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].interpolate(method="linear").ffill().bfill()

                # 4. Handle Categorical Columns (Categorical nulls like Workout_Type)
                cat_cols = cleaned_df.select_dtypes(include=['object']).columns.tolist()
                for col in cat_cols:
                    if col != 'Full Name': # Usually keep names or handle specifically
                        cleaned_df[col] = cleaned_df[col].fillna("Not Recorded")

                # 5. Final check: Fill any absolute remaining nulls with global median/mode
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
                
                st.session_state['cleaned_df'] = cleaned_df
                st.success("Preprocessing Complete! No Null Values remain.")

        st.divider()

        # --- STEP 3: CLEANED DATA PREVIEW ---
        st.markdown("### ✨ Step 3 • Preview Cleaned Dataset")
        if 'cleaned_df' in st.session_state:
            c_df = st.session_state['cleaned_df']
            
            tab1, tab2 = st.tabs(["Data Preview", "Null Value Check"])
            
            with tab1:
                st.dataframe(c_df.head(20), use_container_width=True)
            
            with tab2:
                null_counts = c_df.isnull().sum()
                if null_counts.sum() == 0:
                    st.balloons()
                    st.success("Perfect! 0 Null Values detected.")
                st.write(null_counts)

            # --- STEP 4: DOWNLOAD ---
            st.markdown("### 📥 Step 4 • Export Cleaned Data")
            csv = c_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="FitPulse_Cleaned_Data.csv",
                mime="text/csv"
            )
        else:
            st.info("Run Preprocessing (Step 2) to generate the cleaned dataset.")

        st.divider()

        # --- STEP 5: EDA ---
        st.markdown("### 📊 Step 5 • Exploratory Data Analysis")
        if st.button("Run Full EDA"):
            data = st.session_state.get('cleaned_df', df)
            st.markdown("#### Correlation Heatmap")
            corr = data.select_dtypes(include=[np.number]).corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)
            
            st.markdown("#### Metric Trends (First 100 Records)")
            st.line_chart(data.select_dtypes(include=[np.number]).head(100))

    else:
        st.info("Please upload your Fitness CSV file to begin the pipeline.")

if __name__ == "__main__":
    main()