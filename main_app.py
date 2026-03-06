import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Advanced UI components for prominent visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is required. Please install it using 'pip install plotly'.")

# Library imports for Neural Prediction and Feature Extraction
try:
    from tsfresh import extract_features
    from prophet import Prophet
except ImportError:
    st.error("Missing AI libraries! Run: pip install tsfresh prophet")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FitPulse Pro | Wearable AI Hub",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROMINENT UI THEME & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; border-right: 1px solid #30363D; }
    
    .device-header {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background-color: #1C2128; border: 1px solid #30363D;
        padding: 25px; border-radius: 15px; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.4); transition: 0.4s;
    }
    .metric-card:hover { 
        border-color: #4FACFE; 
        transform: translateY(-8px); 
        box-shadow: 0 15px 30px rgba(79, 172, 254, 0.2); 
    }
    
    .insight-box {
        background: rgba(79, 172, 254, 0.08); border-left: 6px solid #4FACFE;
        padding: 20px; border-radius: 10px; margin: 15px 0; font-family: 'Segoe UI', sans-serif;
        font-size: 0.95rem; line-height: 1.6; color: #E0E0E0;
    }
    
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.2em;
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
        color: #0E1117; border: none; font-weight: bold; font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    
    .pulse {
        display: inline-block; width: 12px; height: 12px;
        background-color: #FF4B4B; border-radius: 50%;
        margin-right: 8px; animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def map_biometric_columns(df):
    mapping = {}
    cols = df.columns
    hr_matches = [c for c in cols if 'heart' in c.lower() or 'bpm' in c.lower()]
    if hr_matches: mapping['Heart_Rate'] = hr_matches[0]
    step_matches = [c for c in cols if 'step' in c.lower()]
    if step_matches: mapping['Steps_Taken'] = step_matches[0]
    sleep_matches = [c for c in cols if 'sleep' in c.lower() or 'slp' in c.lower()]
    if sleep_matches: mapping['Hours_Slept'] = sleep_matches[0]
    return mapping

def generate_radar_chart(df, categories):
    normal_avg = df[df['Is_Anomaly'] == 0][categories].mean()
    anomaly_avg = df[df['Is_Anomaly'] == 1][categories].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=normal_avg, theta=categories, fill='toself', name='Healthy Average', line_color='#4FACFE'))
    fig.add_trace(go.Scatterpolar(r=anomaly_avg, theta=categories, fill='toself', name='Abnormal Event', line_color='#FF4B4B'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, gridcolor="#30363D")),
                      template="plotly_dark", showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def main():
    with st.sidebar:
        st.markdown("<h2 style='color: #4FACFE;'>🩺 FitPulse Settings</h2>", unsafe_allow_html=True)
        st.caption("AI Watch Synchronization Active")
        st.divider()
        menu = st.radio("Intelligence Modules:", 
                        ["Device Connectivity", "Spatial Bio-Mapping", "Neural Hardening", "Behavioral Inference", "Integrity Audit"])
        st.divider()
        st.subheader("Sensitivity Settings")
        hr_threshold = st.slider("Alert if Heart Rate >", 60, 200, 120, help="Manually set your high heart rate alert limit.")
        z_score_threshold = st.slider("AI Sensitivity (Z-Score)", 1.0, 5.0, 2.5, help="Lower = More alerts. Higher = Only major anomalies.")

    st.markdown('<h1 class="device-header"> 🩺 FitPulse Intelligence Hub</h1>', unsafe_allow_html=True)

    # --- DEVICE CONNECTIVITY ---
    if menu == "Device Connectivity":
        st.markdown("### <span class='pulse'></span>Data Integration Station", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your watch data (CSV or JSON)", type=["csv", "json"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
            initial_count, initial_nulls = len(df), df.isnull().sum().sum()
            df = df.drop_duplicates().reset_index(drop=True)
            st.session_state['raw_df'] = df
            
            st.markdown("#### ✅ File Upload Successful")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><h5>Total Readings</h5><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><h5>Metrics Tracked</h5><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><h5>Fixed Errors</h5><h2>{initial_nulls}</h2></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><h5>Data Quality</h5><h2>{100 - (initial_nulls/max(df.size,1))*100:.1f}%</h2></div>', unsafe_allow_html=True)
            
            st.markdown("""<div class="insight-box"><b>User Guide:</b> Your data has been synced. We fixed any missing sensor gaps 
            so the AI can analyze your health without interruptions. You can see a preview of your raw data below.</div>""", unsafe_allow_html=True)
            st.dataframe(df.head(15), use_container_width=True)

    # --- SPATIAL BIO-MAPPING ---
    elif menu == "Spatial Bio-Mapping":
        st.markdown("### Health Pattern Visualization")
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.subheader("Your Health in 3D Space")
            if len(numeric_cols) >= 3:
                fig_3d = px.scatter_3d(df.head(1000), x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], 
                                      color=numeric_cols[0], template="plotly_dark", height=700, color_continuous_scale='Viridis',
                                      labels={numeric_cols[0]: 'Primary Metric', numeric_cols[1]: 'Secondary Metric', numeric_cols[2]: 'Tertiary Metric'})
                st.plotly_chart(fig_3d, use_container_width=True)
                st.info("Tip: Look for clusters! If your data forms tight groups, your health habits are consistent.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Metric Consistency")
                feat = st.selectbox("Select a metric to examine:", numeric_cols)
                fig_hist = px.histogram(df, x=feat, marginal="violin", template="plotly_dark", color_discrete_sequence=['#4FACFE'])
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                st.markdown("#### Vital Metric Connections")
                st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=True, template="plotly_dark", color_continuous_scale='RdBu_r'), use_container_width=True)
        else:
            st.info("Please sync your device first in the 'Device Connectivity' module.")

    # --- NEURAL HARDENING (The Processing Engine) ---
    elif menu == "Neural Hardening":
        st.markdown("### 🧠 AI Health Calibration")
        if 'raw_df' in st.session_state:
            df_worker = st.session_state['raw_df'].copy()
            mapped_cols = map_biometric_columns(df_worker)
            granularity = st.select_slider("Time View Detail:", options=['1min', '30min', 'H', 'D'], value='H')
            
            if st.button("Start AI Calibration"):
                with st.status("Executing Processing Engine...", expanded=True) as status:
                    df_worker['Date'] = pd.to_datetime(df_worker['Date'], errors='coerce')
                    df_worker = df_worker.dropna(subset=['Date'])
                    
                    numeric_features = df_worker.select_dtypes(include=[np.number]).columns.tolist()
                    st.session_state['numeric_features'] = numeric_features
                    st.session_state['stage1_raw'] = df_worker.head(200).copy()
                    
                    # --- STAGE 2: CLEANING (Handling NaNs properly) ---
                    df_worker[numeric_features] = df_worker.groupby('User_ID', group_keys=False)[numeric_features].apply(
                        lambda x: x.interpolate(method='linear').ffill().bfill()
                    )
                    # Safety: If interpolation left any NaNs (e.g., all values in a column were NaN), fill with 0
                    df_worker[numeric_features] = df_worker[numeric_features].fillna(0)
                    st.session_state['stage2_cleaned'] = df_worker.head(200).copy()
                    
                    # --- STAGE 3: NORMALIZATION ---
                    scaler_mm = MinMaxScaler()
                    df_norm_viz = df_worker.copy()
                    df_norm_viz[numeric_features] = scaler_mm.fit_transform(df_worker[numeric_features])
                    st.session_state['stage3_normalized'] = df_norm_viz.head(200).copy()
                    
                    # --- RESAMPLING & FEATURES ---
                    agg_map = {col: ('mean' if pd.api.types.is_numeric_dtype(df_worker[col]) and col not in ['User_ID', 'Age'] else 'first') 
                               for col in df_worker.columns if col != 'Date'}
                    resampled = df_worker.sort_values('Date').set_index('Date').resample(granularity).agg(agg_map).reset_index()
                    
                    hr_col = mapped_cols.get('Heart_Rate', 'Heart_Rate')
                    if hr_col in resampled.columns:
                        resampled['HR_Rolling_Avg'] = resampled[hr_col].rolling(window=7, min_periods=1).mean()
                        resampled['Z_Score'] = np.abs(stats.zscore(resampled[hr_col].fillna(resampled[hr_col].mean())))
                        resampled['Is_Anomaly'] = ((resampled['Z_Score'] > z_score_threshold) | (resampled[hr_col] > hr_threshold)).astype(int)

                    resampled = resampled.fillna(0)
                    st.session_state['cleaned_df'] = resampled.copy()
                    st.session_state['mapped_cols'] = mapped_cols
                    
                    # --- PRE-CLEANED FEATURE MATRIX FOR CLUSTERING ---
                    # Ensure no NaNs or Infs exist in the matrix sent to the next module
                    f_mat = resampled.select_dtypes(include=[np.number]).drop(columns=['Z_Score', 'Is_Anomaly'], errors='ignore')
                    f_mat = f_mat.replace([np.inf, -np.inf], np.nan).fillna(0)
                    st.session_state['feature_matrix'] = f_mat
                    
                    status.update(label="Milestone 1 & 2 Complete!", state="complete")

            # --- PREPROCESSING OVERVIEW ---
            if 'cleaned_df' in st.session_state and 'numeric_features' in st.session_state:
                st.success("✅ Workflow Alignment Confirmed")
                with st.expander("🔍 Milestone 1 & 2: Preprocessing Overview", expanded=True):
                    view_col = st.selectbox("Select metric to track across stages:", st.session_state['numeric_features'])
                    c1, c2, c3 = st.columns(3)
                    with c1: st.line_chart(st.session_state['stage1_raw'][view_col], color="#FF4B4B")
                    with c2: st.line_chart(st.session_state['stage2_cleaned'][view_col], color="#4FACFE")
                    with c3: st.line_chart(st.session_state['stage3_normalized'][view_col], color="#00F2FE")
                st.dataframe(st.session_state['cleaned_df'].head(10), use_container_width=True)
        else:
            st.info("Please sync your device first.")

    # --- BEHAVIORAL INFERENCE (Activity Clusters Fix) ---
    elif menu == "Behavioral Inference":
        st.markdown("### AI Diagnostics & Future Trends")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            tab1, tab2, tab3 = st.tabs(["Health Signature", "Activity Clusters", "Future Predictions"])
            
            with tab1:
                st.markdown("#### Normal vs. Abnormal Comparison")
                biometrics = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept'] if c in df.columns]
                if biometrics:
                    scaled_df = df.copy().fillna(0)
                    scaled_df[biometrics] = StandardScaler().fit_transform(scaled_df[biometrics])
                    st.plotly_chart(generate_radar_chart(scaled_df, biometrics), use_container_width=True)

            with tab2:
                st.markdown("#### Lifestyle Groups")
                if 'feature_matrix' in st.session_state:
                    f_matrix = st.session_state['feature_matrix'].copy()
                    
                    # --- THE FINAL FIX FOR PCA NaN ISSUE ---
                    # 1. Drop columns that are entirely NaN
                    f_matrix = f_matrix.dropna(axis=1, how='all')
                    # 2. Fill any remaining individual NaNs with 0
                    f_matrix = f_matrix.fillna(0)
                    
                    if not f_matrix.empty and len(f_matrix) > 1:
                        # 3. Standardize and check for finite values
                        scaled = StandardScaler().fit_transform(f_matrix)
                        scaled = np.nan_to_num(scaled) # Safety check for infinite values
                        
                        pca = PCA(n_components=2).fit_transform(scaled)
                        pdf = pd.DataFrame(pca, columns=['Dimension A', 'Dimension B'])
                        pdf['Group'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(scaled).labels_.astype(str)
                        st.plotly_chart(px.scatter(pdf, x='Dimension A', y='Dimension B', color='Group', template="plotly_dark"), use_container_width=True)
                    else:
                        st.warning("Insufficient data variation for clustering.")

            with tab3:
                st.markdown("#### 24-Hour Predictive Forecast")
                # Ensure we only select columns that exist and have data
                forecast_cols = [c for c in df.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c and 'Z_Score' not in c]
                metric = st.selectbox("Predict for:", forecast_cols)
                if st.button("Calculate Forecast"):
                    pdf = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'}).tail(200).dropna()
                    if len(pdf) > 2:
                        m = Prophet(daily_seasonality=True).fit(pdf)
                        forecast = m.predict(m.make_future_dataframe(periods=24, freq='H'))
                        st.plotly_chart(px.line(forecast, x='ds', y='yhat', template="plotly_dark"), use_container_width=True)
                    else:
                        st.error("Not enough historical data for prediction.")

    # --- BEHAVIORAL INFERENCE ---
    elif menu == "Behavioral Inference":
        st.markdown("### AI Diagnostics & Future Trends")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            tab1, tab2, tab3 = st.tabs(["Health Signature", "Activity Clusters", "Future Predictions"])
            
            with tab1:
                st.markdown("#### Normal vs. Abnormal Comparison")
                biometrics = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept'] if c in df.columns]
                if biometrics:
                    scaled_df = df.copy()
                    scaled_df[biometrics] = StandardScaler().fit_transform(df[biometrics])
                    st.plotly_chart(generate_radar_chart(scaled_df, biometrics), use_container_width=True)

            with tab2:
                st.markdown("#### Lifestyle Groups")
                if 'feature_matrix' in st.session_state:
                    f_matrix = st.session_state['feature_matrix']
                    if not f_matrix.empty and len(f_matrix) > 1:
                        scaled = StandardScaler().fit_transform(f_matrix)
                        pca = PCA(n_components=2).fit_transform(scaled)
                        pdf = pd.DataFrame(pca, columns=['Dimension A', 'Dimension B'])
                        pdf['Group'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(scaled).labels_.astype(str)
                        st.plotly_chart(px.scatter(pdf, x='Dimension A', y='Dimension B', color='Group', template="plotly_dark"), use_container_width=True)

            with tab3:
                st.markdown("####  24-Hour Predictive Forecast")
                metric = st.selectbox("Predict for:", [c for c in df.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c])
                if st.button("Calculate Forecast"):
                    pdf = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'}).tail(200)
                    m = Prophet(daily_seasonality=True).fit(pdf)
                    forecast = m.predict(m.make_future_dataframe(periods=24, freq='H'))
                    fig_prophet = px.line(forecast, x='ds', y='yhat', template="plotly_dark", title=f"Expected {metric} Trend")
                    st.plotly_chart(fig_prophet, use_container_width=True)
        else:
            st.info("Complete 'Neural Hardening' to see predictions.")

    # --- INTEGRITY AUDIT ---
    elif menu == "Integrity Audit":
        st.markdown("### Health Security Audit")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df'].copy()
            mapped_cols = st.session_state.get('mapped_cols', {})
            hr_col = mapped_cols.get('Heart_Rate', 'Heart_Rate')
            
            df['Date'] = pd.to_datetime(df['Date'])
            anomalies = df[df['Is_Anomaly'] == 1].copy()
            
            st.markdown("#### Critical Safety Alerts")
            critical_events = df[df[hr_col] > (hr_threshold/200)] 
            if not critical_events.empty:
                st.error(f"ALERT: Detected {len(critical_events)} events exceeding your {hr_threshold} BPM limit.")
                with st.expander("View Alarm History"):
                    st.table(critical_events[['Date', hr_col]].tail(10))

            if not anomalies.empty:
                st.markdown("#### Continuous Anomaly Tracker")
                fig_detect = px.line(df, x='Date', y=hr_col, title="Your Pulse Stream", template="plotly_dark")
                fig_detect.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies[hr_col], mode='markers', name='Flagged Event', marker=dict(color='red', size=10)))
                st.plotly_chart(fig_detect, use_container_width=True)

                anomalies['Day'] = anomalies['Date'].dt.day_name()
                anomalies['Hour'] = anomalies['Date'].dt.hour.astype(str) + ":00"
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("####  When do Anomalies happen?")
                    fig_sun = px.sunburst(anomalies, path=['Day', 'Hour'], values='Z_Score', color=hr_col, 
                                          template="plotly_dark", height=600, color_continuous_scale='Reds')
                    st.plotly_chart(fig_sun, use_container_width=True)
                with c2:
                    st.metric("Abnormal Events", len(anomalies))
                
                st.divider()
                st.download_button("Download Medical Audit (CSV)", anomalies.to_csv(index=False).encode('utf-8'), "Health_Audit_Report.csv", "text/csv")
        else:
            st.info("Run 'Neural Hardening' first to generate your audit.")

if __name__ == "__main__":
    main()