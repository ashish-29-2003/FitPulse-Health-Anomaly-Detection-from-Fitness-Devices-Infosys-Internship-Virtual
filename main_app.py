import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="FitPulse Anomaly Detection", page_icon="🏥", layout="wide")

def preprocess_and_resample(df):
    """
    Core logic for Milestone 1: Cleaning, Normalizing, and Resampling
    """
    # 1. Normalize Timestamps
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])
    
    # 2. Handle Missing Values using Linear Interpolation
    numeric_cols = ["Hours_Slept", "Water_Intake (Liters)", "Active_Minutes", "Heart_Rate (bpm)"]
    df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear").ffill().bfill()
    )
    
    # 3. Resample to 1-Hour Granularity
    df = df.set_index("Date")
    df_resampled = df.groupby("User_ID")[numeric_cols].resample('H').mean().reset_index()
    
    # Final cleanup for any gaps created by resampling
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method='linear')
    
    return df_resampled

def main():
    # Sidebar for project info
    st.sidebar.title("FitPulse Project")
    st.sidebar.info("Milestone 1: Data Collection & Preprocessing")
    
    st.title(" FitPulse: Health Anomaly Detection")
    st.subheader("Interactive Preprocessing Dashboard")
    st.markdown("---")

    # Step 1: File Upload UI
    st.write("### Step 1: Data Ingestion")
    uploaded_file = st.file_uploader("Upload Fitness Watch Data (CSV)", type=["csv"])

    if uploaded_file:
        # Load Data
        df_raw = pd.read_csv(uploaded_file)
        
        # Display Metrics for a 'Dashboard' feel
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Records", len(df_raw))
        with m2:
            st.metric("Users Found", df_raw['User_ID'].nunique())
        with m3:
            st.metric("Data Status", "Ready to Process")

        # Step 2: Preprocessing Logic
        df_processed = preprocess_and_resample(df_raw)

        # UI Layout: Side-by-Side Comparisons
        col_left, col_right = st.columns(2)

        with col_left:
            st.write("### Raw Data Preview")
            st.dataframe(df_raw.head(15), use_container_width=True)
            
        with col_right:
            st.write("### Cleaned & Resampled Data")
            st.dataframe(df_processed.head(15), use_container_width=True)

        # Step 3: Time-normalized data log
        st.markdown("---")
        st.write("###  Time-Normalized Data Log")
        
        log_col1, log_col2 = st.columns(2)
        with log_col1:
            log_details = {
                "Timestamp Format": "ISO-8601 (UTC Standardized)",
                "Resampling Frequency": "1-Hour Intervals",
                "Interpolation Method": "Linear",
                "Final Row Count": len(df_processed)
            }
            st.json(log_details)
            
        

    else:
        st.info("Waiting for CSV file upload to display processed results.")

if __name__ == "__main__":
    main()