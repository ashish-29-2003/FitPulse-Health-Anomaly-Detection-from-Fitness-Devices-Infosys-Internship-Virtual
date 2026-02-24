import streamlit as st
import pandas as pd
import numpy as np

def preprocess_data(df):
    """Basic cleaning based on preprocessing_data.ipynb"""
    # Convert Date to datetime objects
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    
    # Identify numeric columns for interpolation
    numeric_cols = ["Hours_Slept", "Water_Intake (Liters)", "Active_Minutes", "Heart_Rate (bpm)"]
    
    # Handle missing values using interpolation and fill methods
    df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear").ffill().bfill()
    )
    
    # Fill categorical missing values
    if "Workout_Type" in df.columns:
        df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")
        
    return df

def main():
    st.set_page_config(page_title="FitPulse Health Anomaly Detection", layout="wide")
    
    st.title("🏥 FitPulse: Health Anomaly Detection")
    st.markdown("### Milestone 1: Data Collection & Preprocessing")
    
    # File Uploader Widget
    uploaded_file = st.file_uploader("Upload your Fitness Health Tracking Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load the raw data
        df_raw = pd.read_csv(uploaded_file)
        
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(10))
        
        # Process the data
        with st.spinner('Cleaning and normalizing data...'):
            df_cleaned = preprocess_data(df_raw)
            
        st.success("Data Preprocessing Complete!")
        
        # Cleaned Data Preview
        st.subheader("Cleaned Dataset Preview")
        st.write(f"Total Records: {df_cleaned.shape[0]} | Columns: {df_cleaned.shape[1]}")
        st.dataframe(df_cleaned.head(10))
        
        # Show Missing Values Count to verify cleaning
        if st.checkbox("Show Missing Values Count"):
            st.write(df_cleaned.isnull().sum())
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()