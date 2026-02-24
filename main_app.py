import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="FitPulse", page_icon="🏥", layout="wide")

def preprocess_and_resample(df):
    """
    Core logic for Milestone 1: Cleaning and Daily Resampling
    """
    # Step 1: Normalize Timestamps
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])
    
    # Step 2: Handle Missing Values using Linear Interpolation
    numeric_cols = ["Hours_Slept", "Water_Intake (Liters)", "Active_Minutes", "Heart_Rate (bpm)"]
    df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear").ffill().bfill()
    )
    
    # Step 3: Resample to Daily Intervals ('D')
    df = df.set_index("Date")
    # Resampling by Day to keep row counts consistent with your original dataset
    df_resampled = df.groupby("User_ID")[numeric_cols].resample('D').mean().reset_index()
    
    # Final cleanup for any gaps created by daily resampling
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method='linear')
    
    return df_resampled

def main():
    st.title("🏥 FitPulse")
   

    # --- STEP 1: DATA INGESTION ---
    # Requirement: Import health data from fitness trackers in CSV format
    with st.expander("Step 1: Data Ingestion", expanded=True):
        st.subheader("Upload Fitness Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file)
            st.info(f"Raw Data Loaded: {len(df_raw)} records")
            st.dataframe(df_raw.head(50), use_container_width=True)
        else:
            st.warning("Please upload the dataset to begin.")
            return

    # --- STEP 2: CLEANING & NORMALIZATION ---
    # Requirement: Clean and normalize timestamps, interpolate missing values
    with st.expander("Step 2: Cleaning & Normalization", expanded=False):
        st.subheader("Data Processing Results")
        
        # Process the raw data
        df_processed = preprocess_and_resample(df_raw)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Processing Steps Applied:**")
            st.write("- Standardized Date Formats")
            st.write("- Filled Missing Health Metrics")
        
        with col2:
            st.markdown("**Processed Data Preview:**")
            st.dataframe(df_processed.head(50), use_container_width=True)

    # --- STEP 3: TIME-SERIES RESAMPLING ---
    # Requirement: Align time intervals to consistent frequency
    with st.expander("Step 3: Time-Series Resampling (Daily)", expanded=False):
        st.subheader("Consistent Interval View")
        st.write("Data aligned to consistent daily intervals.")
        
        # Display the final resampled dataframe
        st.dataframe(df_processed, use_container_width=True)
        
        # Summary metrics
        m1, m2 = st.columns(2)
        m1.metric("Final Row Count", len(df_processed))
        m2.metric("Target Frequency", "Daily ('D')")

   # --- STEP 4: REPORT GENERATION ---
    with st.expander("Step 4: Generate Downloadable Reports", expanded=True):
        st.subheader("Export Preprocessing Summary")
        
        # ... (Report Summary Table code)
        
        # Convert processed data to CSV for download
        # Now df_processed includes 'User_ID' and 'Date' as columns
        csv = df_processed.to_csv(index=False).encode('utf-8')
        
       
        
        
        # Streamlit Download Button
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv,
            file_name='fitpulse_preprocessed_report.csv',
            mime='text/csv',
            help="Click to download the time-normalized and cleaned fitness data."
        )

if __name__ == "__main__":
    main()