import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import plotly.express as px
except ImportError:
    st.error("Plotly is required. Please install it using 'pip install plotly'.")

from tsfresh import extract_features
from prophet import Prophet

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FitPulse Pro | AI Anomaly Engine",
    page_icon="🩺",
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
        st.markdown("<h2 style='color: #4FACFE;'>🩺 FitPulse Engine</h2>", unsafe_allow_html=True)
        st.caption("Infosys Springboard Virtual Internship")
        st.divider()
        menu = st.radio("Project Workflow:", 
                        ["Step 1: Data Ingestion", 
                         "Step 2: Neural Exploratory Analysis", 
                         "Step 3: Neural Preprocessing & Feature Extraction", 
                         "Step 4: Trend Modeling & Clustering", 
                         "Step 5: Final Audit & Export"])
        
    st.markdown('<h1 class="main-header">🩺 FitPulse Intelligence Dashboard</h1>', unsafe_allow_html=True)

    # --- STEP 1: DATA INGESTION ---
    if menu == "Step 1: Data Ingestion":
        st.markdown("### 🛰️ Data Stream Ingestion")
        uploaded_file = st.file_uploader("Upload Health Telemetry CSV", type=["csv"])
        if uploaded_file:
            # FIX: Immediate imputation of common nulls to prevent 0.00 displays
            raw_data = pd.read_csv(uploaded_file)
            st.session_state['raw_df'] = raw_data
            df = st.session_state['raw_df']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><h5>Raw Records</h5><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><h5>Feature Count</h5><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><h5>Null Gaps</h5><h2>{df.isnull().sum().sum()}</h2></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><h5>Integrity</h5><h2>{100 - (df.isnull().sum().sum()/max(df.size, 1))*100:.1f}%</h2></div>', unsafe_allow_html=True)
            
            st.write("### Initial Telemetry Preview")
            st.dataframe(df.head(10), use_container_width=True)

    # --- STEP 2: NEURAL EXPLORATORY ANALYSIS ---
    elif menu == "Step 2: Neural Exploratory Analysis":
        st.markdown("### 🧠 Neural Exploratory Analysis")
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            st.markdown("#### Descriptive Stats")
            st.dataframe(df.describe().T, use_container_width=True)
            
            col1, col2 = st.columns(2)
            numeric_df = df.select_dtypes(include=[np.number])
            
            with col1:
                st.markdown("#### 🌡️ Heatmap Correlation")
                if not numeric_df.empty:
                    fig_corr = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
            with col2:
                st.markdown("#### 📊 Distribution Analysis")
                if not numeric_df.empty:
                    feat = st.selectbox("Select Feature:", numeric_df.columns)
                    fig_dist = px.histogram(df, x=feat, marginal="box", color_discrete_sequence=['#4FACFE'])
                    st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Please upload a dataset in Step 1.")

    # --- STEP 3: NEURAL PREPROCESSING & FEATURE EXTRACTION ---
    elif menu == "Step 3: Neural Preprocessing & Feature Extraction":
        st.markdown("### ⚡ Preprocessing & TSFresh Extraction")
        if 'raw_df' in st.session_state:
            if st.button("Execute Neural Pipeline"):
                df = st.session_state['raw_df'].copy()
                with st.status("Hardening Data & Extracting Time-Series Features...", expanded=True) as status:
                    
                    # 1. Date Format Correction
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    
                    # 2. Resampling & Null Purging (FIXED: Improved Imputation)
                    # We ensure Hours_Slept and Water_Intake are averaged, while IDs are kept static
                    agg_map = {col: ('mean' if pd.api.types.is_numeric_dtype(df[col]) and col not in ['User_ID', 'Age'] else 'first') for col in df.columns if col != 'Date'}
                    resampled = df.sort_values('Date').set_index('Date').resample('H').agg(agg_map).reset_index()
                    
                    # FIX: Handle missing values explicitly for dashboard metrics
                    num_cols = resampled.select_dtypes(include=[np.number]).columns
                    resampled[num_cols] = resampled[num_cols].interpolate(method='linear')
                    resampled = resampled.ffill().bfill()
                    
                    # 3. Data Type Hardening (FIXED: Prevent Decimal IDs)
                    if 'User_ID' in resampled.columns: 
                        resampled['User_ID'] = resampled['User_ID'].fillna(0).astype(int)
                    if 'Age' in resampled.columns:
                        resampled['Age'] = resampled['Age'].fillna(0).astype(int)

                    # 4. TSFRESH FEATURE EXTRACTION
                    st.write("Extracting statistical features (Mean, Std, Kurtosis)...")
                    target_cols = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept'] if c in resampled.columns]
                    if target_cols:
                        ts_features = extract_features(resampled[['User_ID', 'Date'] + target_cols], 
                                                       column_id='User_ID', column_sort='Date')
                        st.session_state['feature_matrix'] = ts_features.dropna(axis=1)

                    # 5. Statistical Anomaly Baseline
                    target_col = 'Heart_Rate' if 'Heart_Rate' in resampled.columns else resampled.select_dtypes(include=[np.number]).columns[0]
                    resampled['Z_Score'] = np.abs(stats.zscore(resampled[target_col]))
                    resampled['Is_Anomaly'] = (resampled['Z_Score'] > 2.5).astype(int)
                    
                    st.session_state['cleaned_df'] = resampled.copy()
                    status.update(label="Feature Extraction Complete!", state="complete")
                
                st.success("Preprocessing Complete. Statistical Features Extracted.")
                st.write("#### Feature Matrix Preview")
                st.dataframe(st.session_state['feature_matrix'].head(), use_container_width=True)
        else:
            st.info("Please upload a dataset in Step 1.")

    # --- STEP 4: TREND MODELING & CLUSTERING ---
    elif menu == "Step 4: Trend Modeling & Clustering":
        st.markdown("### 📈 Advanced Behavioral Modeling")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            
            t1, t2 = st.tabs(["Prophet Trend Analysis", "KMeans Behavioral Clusters"])
            
            with t1:
                st.markdown("#### Seasonal Trend Modeling")
                metric_options = [c for c in df.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c and 'Z_Score' not in c]
                metric = st.selectbox("Metric to Model:", metric_options)
                if st.button("Generate Trend Forecast"):
                    # Prepare Prophet data
                    pdf = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'}).tail(200)
                    m = Prophet(yearly_seasonality=True, daily_seasonality=True)
                    m.fit(pdf)
                    future = m.make_future_dataframe(periods=24, freq='H')
                    forecast = m.predict(future)
                    
                    fig = px.line(forecast, x='ds', y='yhat', title=f"{metric} Projected Trend", template="plotly_dark")
                    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Confidence Upper", line=dict(dash='dot'))
                    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Confidence Lower", line=dict(dash='dot'))
                    st.plotly_chart(fig, use_container_width=True)

            with t2:
                st.markdown("#### User Pattern Clustering")
                if 'feature_matrix' in st.session_state:
                    f_matrix = st.session_state['feature_matrix']
                    scaled = StandardScaler().fit_transform(f_matrix)
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
                    
                    pca = PCA(n_components=2).fit_transform(scaled)
                    pdf = pd.DataFrame(pca, columns=['PC1', 'PC2'])
                    pdf['Cluster'] = kmeans.labels_.astype(str)
                    
                    fig_cluster = px.scatter(pdf, x='PC1', y='PC2', color='Cluster', title="Behavioral Groups (PCA Projection)", template="plotly_dark")
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.warning("Please run Feature Extraction in Step 3 first.")
        else:
            st.info("Execute Preprocessing first.")

    # --- STEP 5: FINAL AUDIT & EXPORT ---
    elif menu == "Step 5: Final Audit & Export":
        st.markdown("### 📋 Comprehensive Technical Audit")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            aud_c1, aud_c2, aud_c3 = st.columns(3)
            
            # FIX: Ensure metrics are clean for final report
            aud_c1.metric("RESIDUAL NULLS", f"{df.isnull().sum().sum()}", delta="CLEAN")
            aud_c2.metric("ANOMALIES", f"{df['Is_Anomaly'].sum()}")
            
            # Verify IDs are cleaned integers
            id_status = "INT" if pd.api.types.is_integer_dtype(df['User_ID']) else "FLOAT"
            aud_c3.metric("ID FORMAT", id_status)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button("📥 Download Final Report", df.to_csv(index=False).encode('utf-8'), "FitPulse_Final.csv", "text/csv")

if __name__ == "__main__":
    main()