import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
from datetime import datetime
try:
    import plotly.express as px
except ImportError:
    st.error("Plotly is required. Please install it using 'pip install plotly'.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FitPulse Pro | AI Anomaly Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL UI THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; border-right: 1px solid #30363D; }
    .main-header {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #1C2128; border: 1px solid #30363D;
        padding: 20px; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: 0.3s;
    }
    .metric-card:hover { border-color: #4FACFE; transform: translateY(-5px); }
    .report-box {
        background-color: #0D1117; border-radius: 10px; padding: 25px;
        border: 1px solid #4FACFE; margin-top: 15px;
    }
    .stButton>button {
        width: 100%; border-radius: 10px;
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
        color: #0E1117; border: none; padding: 12px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # --- SIDEBAR: WORKFLOW ---
    with st.sidebar:
        st.markdown("<h2 style='color: #4FACFE;'>⚡ FitPulse Engine</h2>", unsafe_allow_html=True)
        st.caption("Infosys Springboard Virtual Internship")
        st.divider()
        menu = st.radio("Project Workflow:", 
                        ["Step 1: Data Ingestion", 
                         "Step 2: Neural Exploratory Analysis", 
                         "Step 3: Neural Preprocessing", 
                         "Step 4: Anomaly Intelligence", 
                         "Step 5: Final Audit & Export"])
        
        st.divider()
        st.markdown("### 🏆 Milestone Tracker")
        st.write("✅ Preprocessing & Static Consistency")
        st.write("✅ Time-Series Normalization")
        st.write("✅ Exploratory Analysis")
        st.write("✅ AI Anomaly Logic")
        st.divider()
        st.info("System Engine: Python v3.10")

    st.markdown('<h1 class="main-header">🩺 FitPulse Intelligence Dashboard</h1>', unsafe_allow_html=True)

    # --- STEP 1: DATA INGESTION ---
    if menu == "Step 1: Data Ingestion":
        st.markdown("### 📥 1.0 Data Stream Ingestion")
        uploaded_file = st.file_uploader("Upload Health Telemetry CSV", type=["csv"])
        if uploaded_file:
            st.session_state['raw_df'] = pd.read_csv(uploaded_file)
            df = st.session_state['raw_df']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><h5>Raw Records</h5><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><h5>Feature Count</h5><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><h5>Null Gaps</h5><h2>{df.isnull().sum().sum()}</h2></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><h5>Integrity</h5><h2>{100 - (df.isnull().sum().sum()/max(df.size, 1))*100:.1f}%</h2></div>', unsafe_allow_html=True)
            
            st.write("### Initial Telemetry Preview")
            st.dataframe(df.head(10), use_container_width=True)

    # --- STEP 2: NEURAL EXPLORATORY ANALYSIS (EDA) ---
    elif menu == "Step 2: Neural Exploratory Analysis":
        st.markdown("### 🔍 2.0 Neural Exploratory Analysis")
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            st.markdown("#### 📊 Descriptive Stats")
            st.dataframe(df.describe().T, use_container_width=True)
            
            col1, col2 = st.columns(2)
            numeric_df = df.select_dtypes(include=[np.number])
            
            with col1:
                st.markdown("#### 🌡️ Heatmap Correlation")
                if not numeric_df.empty:
                    fig_corr = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
            with col2:
                st.markdown("#### 📈 Distribution Analysis")
                if not numeric_df.empty:
                    feat = st.selectbox("Select Feature:", numeric_df.columns)
                    fig_dist = px.histogram(df, x=feat, marginal="box", color_discrete_sequence=['#4FACFE'])
                    st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Please upload a dataset in Step 1.")

    # --- STEP 3: NEURAL PREPROCESSING (FIXED USER_ID & NULLS) ---
    elif menu == "Step 3: Neural Preprocessing":
        st.markdown("### ⚙️ 3.0 Preprocessing & Resampling")
        if 'raw_df' in st.session_state:
            if st.button("🚀 Execute Neural Pipeline"):
                df = st.session_state['raw_df'].copy()
                with st.status("Hardening Data Structure & Fixing User IDs...", expanded=True) as status:
                    
                    # 1. Date Format Correction
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    
                    # 2. SELECTIVE AGGREGATION
                    agg_map = {}
                    # Define columns that must remain integers and non-averaged
                    id_cols = ['User_ID', 'Age']
                    static_cols = ['Gender', 'Height', 'Weight']
                    
                    for col in df.columns:
                        if col == 'Date': continue
                        if col in id_cols or col in static_cols:
                            agg_map[col] = 'first' # Do not average IDs or Gender
                        elif pd.api.types.is_numeric_dtype(df[col]):
                            agg_map[col] = 'mean'
                        else:
                            agg_map[col] = 'first'
                    
                    # Hourly Resampling
                    resampled = df.sort_values('Date').set_index('Date').resample('H').agg(agg_map).reset_index()
                    
                    # 3. ADVANCED NULL PURGING
                    num_cols = resampled.select_dtypes(include=[np.number]).columns
                    resampled[num_cols] = resampled[num_cols].interpolate(method='linear')
                    resampled = resampled.ffill().bfill()
                    
                    # 4. DATA TYPE HARDENING (Fixes Decimal User IDs)
                    for col in id_cols:
                        if col in resampled.columns:
                            resampled[col] = resampled[col].astype(int) # Force back to clean integers
                    
                    # 5. Path Creation
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    
                    # 6. Feature Extraction
                    resampled['Hour'] = resampled['Date'].dt.hour
                    
                    # 7. Statistical Anomaly Baseline (Z-Score)
                    target_col = 'Heart_Rate' if 'Heart_Rate' in resampled.columns else (resampled.select_dtypes(include=[np.number]).columns[0] if not resampled.select_dtypes(include=[np.number]).empty else None)
                    
                    if target_col:
                        resampled['Z_Score'] = np.abs(stats.zscore(resampled[target_col]))
                        resampled['Is_Anomaly'] = (resampled['Z_Score'] > 2.5).astype(int)
                    
                    st.session_state['cleaned_df'] = resampled.copy()
                    status.update(label="Neural Purge Complete! IDs Restored.", state="complete")
                
                st.success("✅ Dataset Sanitized. User IDs restored to integer format.")
                
                p1, p2, p3 = st.columns(3)
                p1.markdown(f'<div class="metric-card"><h5>Clean Rows</h5><h2>{len(resampled):,}</h2></div>', unsafe_allow_html=True)
                p2.markdown(f'<div class="metric-card"><h5>Residual Nulls</h5><h2>{resampled.isnull().sum().sum()}</h2></div>', unsafe_allow_html=True)
                anom_count = resampled['Is_Anomaly'].sum() if 'Is_Anomaly' in resampled else 0
                p3.markdown(f'<div class="metric-card"><h5>AI Anomalies</h5><h2>{anom_count}</h2></div>', unsafe_allow_html=True)
                
                st.write("#### Data Preview (Look at User_ID column)")
                st.dataframe(resampled.head(20), use_container_width=True)
        else:
            st.info("Please upload a dataset in Step 1.")

    # --- STEP 4 & 5 REMAINS THE SAME AS PREVIOUS ---
    elif menu == "Step 4: Anomaly Intelligence":
        st.markdown("### 📊 4.0 AI Health Analytics")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            metrics = [c for c in df.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c and 'Z_Score' not in c and 'Hour' not in c]
            selected = st.multiselect("Select Telemetry:", metrics, default=metrics[:1] if metrics else [])
            if selected:
                st.plotly_chart(px.line(df.head(500), x='Date', y=selected, template="plotly_dark"), use_container_width=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                y_ax = 'Heart_Rate' if 'Heart_Rate' in df.columns else (metrics[0] if metrics else None)
                if y_ax:
                    st.plotly_chart(px.scatter(df.head(500), x='Date', y=y_ax, color='Is_Anomaly', color_continuous_scale=['#4FACFE', '#FF4B4B']), use_container_width=True)
            with c2:
                if 'Is_Anomaly' in df.columns:
                    st.plotly_chart(px.pie(df, names='Is_Anomaly', hole=0.5, color_discrete_map={0:'#4FACFE', 1:'#FF4B4B'}), use_container_width=True)
        else:
            st.info("Execute Preprocessing first.")

    elif menu == "Step 5: Final Audit & Export":
        st.markdown("### 📋 5.0 Comprehensive Technical Audit")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            aud_c1, aud_c2, aud_c3 = st.columns(3)
            aud_c1.metric("RESIDUAL NULLS", f"{df.isnull().sum().sum()}", delta="CLEAN")
            aud_c2.metric("ANOMALIES", f"{df['Is_Anomaly'].sum()}")
            # Verify IDs are still ints
            id_status = "INT" if pd.api.types.is_integer_dtype(df['User_ID']) else "FLOAT"
            aud_c3.metric("ID FORMAT", id_status)
            st.markdown('</div>', unsafe_allow_html=True)
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button("💾 Download Clean Dataset", df.to_csv(index=False).encode('utf-8'), "FitPulse_Final.csv", "text/csv")

if __name__ == "__main__":
    main()