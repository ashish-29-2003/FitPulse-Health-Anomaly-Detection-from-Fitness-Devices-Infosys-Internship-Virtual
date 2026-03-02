import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    
    /* Device Header Styling */
    .device-header {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; margin-bottom: 0.5rem;
    }
    
    /* Smartwatch-style Metric Cards */
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
    }
    
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.2em;
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
        color: #0E1117; border: none; font-weight: bold; font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    
    /* Pulse Animation for Live Connection */
    .pulse {
        display: inline-block; width: 12px; height: 12px;
        background-color: #FF4B4B; border-radius: 50%;
        margin-right: 8px; animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

def generate_radar_chart(df, categories):
    """Displays the 'Biometric Signature' of Anomaly vs Normal states"""
    normal_avg = df[df['Is_Anomaly'] == 0][categories].mean()
    anomaly_avg = df[df['Is_Anomaly'] == 1][categories].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=normal_avg, theta=categories, fill='toself', name='Baseline Health', line_color='#4FACFE'))
    fig.add_trace(go.Scatterpolar(r=anomaly_avg, theta=categories, fill='toself', name='Anomaly Event', line_color='#FF4B4B'))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 2], gridcolor="#30363D")),
        template="plotly_dark", showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def main():
    with st.sidebar:
        st.markdown("<h2 style='color: #4FACFE;'>🩺 FitPulse Watch</h2>", unsafe_allow_html=True)
        st.caption("Ecosystem Synchronized")
        st.divider()
        # Simplified and renamed navigation modules
        menu = st.radio("Intelligence Modules:", 
                        ["Device Connectivity", "Spatial Bio-Mapping", "Neural Hardening", "Behavioral Inference", "Integrity Audit"])
        
    st.markdown('<h1 class="device-header"> 🩺 FitPulse Intelligence Hub</h1>', unsafe_allow_html=True)

    # --- DEVICE CONNECTIVITY ---
    if menu == "Device Connectivity":
        st.markdown("### <span class='pulse'></span>Telemetry Data Stream", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Sync Device Data (CSV)", type=["csv"])
        if uploaded_file:
            st.session_state['raw_df'] = pd.read_csv(uploaded_file)
            df = st.session_state['raw_df']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><h5>Sync Points</h5><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><h5>Vital Sensors</h5><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><h5>Null Gaps</h5><h2>{df.isnull().sum().sum()}</h2></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><h5>Sync Integrity</h5><h2>{100 - (df.isnull().sum().sum()/max(df.size, 1))*100:.1f}%</h2></div>', unsafe_allow_html=True)
            
            st.dataframe(df.head(15), use_container_width=True)

    # --- SPATIAL BIO-MAPPING ---
    elif menu == "Spatial Bio-Mapping":
        st.markdown("### 3D Vital Feature Mapping")
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.markdown("#### Multi-Dimensional Biometric Relationship")
            if len(numeric_cols) >= 3:
                fig_3d = px.scatter_3d(
                    df.head(1000), x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], 
                    color=numeric_cols[0], template="plotly_dark", height=700,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Metric Distribution Analysis")
                feat = st.selectbox("Select Vital Metric:", numeric_cols)
                st.plotly_chart(px.histogram(df, x=feat, marginal="violin", color_discrete_sequence=['#4FACFE'], template="plotly_dark"), use_container_width=True)
            with col2:
                st.markdown("#### Neural Correlation Heatmap")
                st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)
        else:
            st.info("Sync device connectivity to access mapping.")

    # --- NEURAL HARDENING ---
    elif menu == "Neural Hardening":
        st.markdown("### Preprocessing & AI Hardening")
        if 'raw_df' in st.session_state:
            if st.button("Initialize Neural Pipeline"):
                df = st.session_state['raw_df'].copy()
                with st.status("Executing Neural Pipeline...", expanded=True) as status:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    
                    agg_map = {col: ('mean' if pd.api.types.is_numeric_dtype(df[col]) and col not in ['User_ID', 'Age'] else 'first') for col in df.columns if col != 'Date'}
                    resampled = df.sort_values('Date').set_index('Date').resample('H').agg(agg_map).reset_index()
                    
                    resampled = resampled.interpolate(method='linear').ffill().bfill()
                    if 'User_ID' in resampled.columns: resampled['User_ID'] = resampled['User_ID'].fillna(0).astype(int)

                    target_cols = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept'] if c in resampled.columns]
                    if target_cols:
                        st.session_state['feature_matrix'] = extract_features(resampled[['User_ID', 'Date'] + target_cols], column_id='User_ID', column_sort='Date').dropna(axis=1)

                    target_col = target_cols[0] if target_cols else resampled.select_dtypes(include=[np.number]).columns[0]
                    resampled['Z_Score'] = np.abs(stats.zscore(resampled[target_col]))
                    resampled['Is_Anomaly'] = (resampled['Z_Score'] > 2.5).astype(int)
                    
                    st.session_state['cleaned_df'] = resampled.copy()
                    status.update(label="Hardening Complete!", state="complete")
                st.success("Watch data sanitized and calibrated.")
        else:
            st.info("Sync device connectivity to initialize hardening.")

    # --- BEHAVIORAL INFERENCE ---
    elif menu == "Behavioral Inference":
        st.markdown("### Behavioral Intelligence & Forecasting")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            
            anom_total = df['Is_Anomaly'].sum()
            st.markdown(f'<div class="insight-box"><b>AI DIAGNOSTIC:</b> {anom_total} anomalies detected. Predictive models suggest irregular patterns in vital telemetry.</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Health Signature", "Cluster Analysis", "Neural Forecast"])
            
            with tab1:
                st.markdown("#### Multi-Variate Anomaly Signature")
                biometrics = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept', 'Water_Intake'] if c in df.columns]
                scaled_df = df.copy()
                scaled_df[biometrics] = StandardScaler().fit_transform(df[biometrics])
                st.plotly_chart(generate_radar_chart(scaled_df, biometrics), use_container_width=True)

            with tab2:
                if 'feature_matrix' in st.session_state:
                    f_matrix = st.session_state['feature_matrix']
                    scaled = StandardScaler().fit_transform(f_matrix)
                    pca = PCA(n_components=2).fit_transform(scaled)
                    pdf = pd.DataFrame(pca, columns=['PC1', 'PC2'])
                    pdf['Cluster'] = KMeans(n_clusters=3, random_state=42).fit(scaled).labels_.astype(str)
                    st.plotly_chart(px.scatter(pdf, x='PC1', y='PC2', color='Cluster', title="Behavioral Clusters (PCA)", template="plotly_dark"), use_container_width=True)
                else:
                    st.warning("Initialize Neural Hardening first.")

            with tab3:
                metric = st.selectbox("Select Vital for Prediction:", [c for c in df.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c])
                if st.button("Generate Forecast"):
                    pdf = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'}).tail(200)
                    m = Prophet(daily_seasonality=True).fit(pdf)
                    forecast = m.predict(m.make_future_dataframe(periods=24, freq='H'))
                    st.plotly_chart(px.line(forecast, x='ds', y='yhat', title=f"Predicted 24h {metric} Trend", template="plotly_dark"), use_container_width=True)
        else:
            st.info("Initialize Neural Hardening to access diagnostics.")

    # --- INTEGRITY AUDIT ---
    elif menu == "Integrity Audit":
        st.markdown("### Technical Health Audit")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            
            st.markdown("#### Temporal Anomaly Hotspots")
            df['Day'] = df['Date'].dt.day_name()
            df['Hour'] = df['Date'].dt.hour.astype(str) + ":00"
            fig_sun = px.sunburst(df[df['Is_Anomaly']==1], path=['Day', 'Hour'], values='Z_Score', 
                                  color='Z_Score', template="plotly_dark", height=650, color_continuous_scale='Reds')
            st.plotly_chart(fig_sun, use_container_width=True)
            
            st.download_button("📥 Export Health Report", df.to_csv(index=False).encode('utf-8'), "Vital_Audit.csv", "text/csv")
        else:
            st.info("Initialize Neural Hardening to run audit.")

if __name__ == "__main__":
    main()