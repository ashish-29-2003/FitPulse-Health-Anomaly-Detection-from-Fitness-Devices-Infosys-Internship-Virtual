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
    from tsfresh.feature_extraction import MinimalFCParameters
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
        hr_threshold = st.slider("Alert if Heart Rate >", 60, 200, 120)
        z_score_threshold = st.slider("AI Sensitivity (Z-Score)", 1.0, 5.0, 2.5)

    st.markdown('<h1 class="device-header"> 🩺 FitPulse Intelligence Hub</h1>', unsafe_allow_html=True)

    # --- DEVICE CONNECTIVITY (Cleaned via Notebook logic) ---
    if menu == "Device Connectivity":
        st.markdown("### <span class='pulse'></span>Data Integration Station", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your watch data (CSV or JSON)", type=["csv", "json"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
            initial_count, initial_nulls = len(df), df.isnull().sum().sum()
            
            # Integrated Notebook Preprocessing Logic
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(lambda x: x.interpolate(method="linear"))
            df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(lambda x: x.ffill().bfill())
            if "Workout_Type" in df.columns:
                df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")
            
            st.session_state['raw_df'] = df
            st.success(f"File Upload Successful. Repaired {initial_nulls} data gaps using user-grouped interpolation.")
            st.dataframe(df.head(15), use_container_width=True)

    # --- SPATIAL BIO-MAPPING ---
    elif menu == "Spatial Bio-Mapping":
        st.markdown("### Health Pattern Visualization")
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                fig_3d = px.scatter_3d(df.head(1000), x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], 
                                      color=numeric_cols[0], template="plotly_dark", height=700)
                st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Please sync your device first.")

    # --- NEURAL HARDENING ---
    elif menu == "Neural Hardening":
        st.markdown("### 🧠 AI Health Calibration")
        if 'raw_df' in st.session_state:
            df_worker = st.session_state['raw_df'].copy()
            mapped_cols = map_biometric_columns(df_worker)
            if st.button("Start AI Calibration"):
                with st.status("Analyzing Health Trends...") as status:
                    hr_col = mapped_cols.get('Heart_Rate', 'Heart_Rate (bpm)')
                    df_worker['Z_Score'] = np.abs(stats.zscore(df_worker[hr_col].fillna(df_worker[hr_col].mean())))
                    df_worker['Is_Anomaly'] = ((df_worker['Z_Score'] > z_score_threshold) | (df_worker[hr_col] > hr_threshold)).astype(int)
                    st.session_state['cleaned_df'] = df_worker
                    status.update(label="Calibration Complete!", state="complete")
        else:
            st.info("Please sync your device first.")

    # --- BEHAVIORAL INFERENCE (Integrated TSFresh logic) ---
    elif menu == "Behavioral Inference":
        st.markdown("### AI Diagnostics & Future Trends")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            tab1, tab2 = st.tabs(["Lifestyle Clusters", "Future Predictions"])
            
            with tab1:
                st.markdown("#### Activity Clusters (TSFresh Statistical Features)")
                # Minimal Feature Set from Notebook
                ts_input = df[["User_ID", "Date", "Heart_Rate (bpm)"]].rename(columns={"User_ID": "id", "Date": "time", "Heart_Rate (bpm)": "value"})
                features = extract_features(ts_input, column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters())
                
                # Stability logic from notebook
                features = features.dropna(axis=1, how="all").fillna(0)
                scaled = StandardScaler().fit_transform(features)
                scaled = np.nan_to_num(scaled)
                
                pca = PCA(n_components=2).fit_transform(scaled)
                pdf = pd.DataFrame(pca, columns=['Dim A', 'Dim B'])
                pdf['Group'] = KMeans(n_clusters=3, random_state=42).fit(scaled).labels_.astype(str)
                st.plotly_chart(px.scatter(pdf, x='Dim A', y='Dim B', color='Group', title="User Lifestyle Segmentation"), use_container_width=True)

            with tab2:
                st.markdown("#### 24-Hour Predictive Forecast")
                metric = st.selectbox("Predict for:", [c for c in df.columns if df[c].dtype != object and "Date" not in c])
                if st.button("Forecast"):
                    pdf = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'}).tail(200)
                    m = Prophet().fit(pdf)
                    forecast = m.predict(m.make_future_dataframe(periods=24, freq='H'))
                    st.plotly_chart(px.line(forecast, x='ds', y='yhat', title=f"Predicted {metric} Path"), use_container_width=True)
        else:
            st.info("Run 'Neural Hardening' first.")

    # --- INTEGRITY AUDIT ---
    elif menu == "Integrity Audit":
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            anomalies = df[df['Is_Anomaly'] == 1]
            st.error(f"ALERT: Detected {len(anomalies)} events requiring review.")
            st.table(anomalies[['Date', 'Heart_Rate (bpm)', 'Z_Score']].tail(10))
        else:
            st.info("Complete calibration to generate audit.")

if __name__ == "__main__":
    main()