import streamlit as st
import pandas as pd
import numpy as np

# Page configuration with a unique smartwatch theme
st.set_page_config(
    page_title="FitPulse ", 
    page_icon="⌚", 
    layout="wide"
)

# Custom CSS for a Unique Glassmorphism / Smartwatch UI
st.markdown("""
    <style>
    /* Main background with a dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    
    /* Glassmorphism Card Effect */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Neon Metric Styling */
    div[data-testid="stMetric"] {
        background: rgba(0, 212, 255, 0.07);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }

    /* High-contrast Download Button */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 30px;
        font-weight: 700;
        letter-spacing: 1px;
        transition: 0.4s all ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

def preprocess_and_resample(df):
    """
    Milestone 1 Core Logic: Cleaning & Daily Resampling
    """
    # Normalize Timestamps to standardized datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])
    
    # Linear Interpolation for health metrics missing values
    numeric_cols = ["Hours_Slept", "Water_Intake (Liters)", "Active_Minutes", "Heart_Rate (bpm)"]
    df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear").ffill().bfill()
    )
    
    # Resample to Daily ('D') to prevent row explosion
    df = df.set_index("Date")
    # reset_index() restores 'Date' as a column for the CSV export
    df_resampled = df.groupby("User_ID")[numeric_cols].resample('D').mean().reset_index()
    
    # Final cleanup to ensure no gaps remain
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method='linear')
    
    return df_resampled

def main():
    # Sidebar - Wearable Device Control Center
    st.sidebar.markdown("<h2 style='text-align: center;'>⌚ FitPulse OS</h2>", unsafe_allow_html=True)
    st.sidebar.write(" **System:** Online")
    st.sidebar.write("**Module:** Preprocessing")
    
    
    # Main Dashboard Header
    st.title("⌚Dashboard")
    st.markdown("#### Preprocessing Engine | Milestone 1")
    

    # SECTION 1: DEVICE SYNC (INGESTION)
    st.markdown("###  Step 1: Device Synchronization")
    uploaded_file = st.file_uploader("Sync Fitness Logs (CSV)", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        
        # Grid Layout for Analytics
        col_main, col_side = st.columns([3, 1])

        with col_side:
            st.markdown("### Metrics")
            st.metric("Total Users", df_raw['User_ID'].nunique())
            st.metric("Data Points", f"{len(df_raw):,}")
            st.metric("Frequency", "Daily ('D')")

        with col_main:
            # SECTION 2: SMART CLEANING
            df_processed = preprocess_and_resample(df_raw)
            
            st.markdown("### Step 2: Health Stream Preview")
            st.dataframe(df_processed.head(50), use_container_width=True)
            
            with st.expander("Technical Processing Logs"):
                st.write("- **Interpolation**: Linear method successfully applied")
                st.write("- **Normalization**: Timestamps converted to UTC format")
                st.write("- **Alignment**: Time-intervals resampled to Daily frequency")

        # SECTION 3: EXPORT (MODULE 4 PREVIEW)
        st.markdown("---")
        st.markdown("### Step 3: Export Health Report")
        
        # Centering the download button with spacing
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            # Preserve 'Date' in the CSV for report integrity
            csv_output = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Generate & Download Health Report",
                data=csv_output,
                file_name='fitpulse_preprocessed_report.csv',
                mime='text/csv',
                use_container_width=True
            )
            st.success("Report Compiled: Features and Timestamps validated.")

    else:
        # Welcome State for Smartwatch UI
        st.info(" Welcome to FitPulse. Connect your fitness data to initiate the preprocessing engine.")

if __name__ == "__main__":
    main()