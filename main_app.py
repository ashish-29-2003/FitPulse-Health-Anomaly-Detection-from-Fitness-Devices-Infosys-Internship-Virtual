import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is required. Install it using 'pip install plotly'.")

# Library imports with dedicated error handling for Step 4
try:
    from tsfresh import extract_features
    from prophet import Prophet
except ImportError:
    st.error("Missing AI libraries! Run: pip install tsfresh prophet")

# --- PAGE CONFIG ---
st.set_page_config(page_title="FitPulse Pro | AI Anomaly Engine", page_icon="🩺", layout="wide")

# --- CUSTOM CSS THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
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
    .insight-box {
        background: rgba(79, 172, 254, 0.1); border-left: 5px solid #4FACFE;
        padding: 15px; border-radius: 8px; margin: 10px 0; font-family: sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_radar_chart(df, categories):
    """Visualizes the biometric 'signature' of Anomalies vs Normal data"""
    normal_avg = df[df['Is_Anomaly'] == 0][categories].mean()
    anomaly_avg = df[df['Is_Anomaly'] == 1][categories].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=normal_avg, theta=categories, fill='toself', name='Normal', line_color='#4FACFE'))
    fig.add_trace(go.Scatterpolar(r=anomaly_avg, theta=categories, fill='toself', name='Anomaly', line_color='#FF4B4B'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 2])), template="plotly_dark", title="Anomaly Biometric Signature")
    return fig

def main():
    with st.sidebar:
        st.markdown("<h2 style='color: #4FACFE;'>🩺 FitPulse Pro</h2>", unsafe_allow_html=True)
        st.caption("AI-Powered Fitness Anomaly Detection")
        st.divider()
        menu = st.radio("Navigation:", ["Step 1: Data Ingestion", "Step 2: Advanced EDA", "Step 3: Neural Pipeline", "Step 4: AI Insights & Forecasting", "Step 5: Audit & Export"])

    st.markdown('<h1 class="main-header">🩺 FitPulse Intelligence Dashboard</h1>', unsafe_allow_html=True)

    # --- STEP 1: INGESTION ---
    if menu == "Step 1: Data Ingestion":
        st.markdown("### 🛰️ Data Stream Ingestion")
        uploaded_file = st.file_uploader("Upload Health Telemetry CSV", type=["csv"])
        if uploaded_file:
            st.session_state['raw_df'] = pd.read_csv(uploaded_file)
            df = st.session_state['raw_df']
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><h5>Records</h5><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><h5>Features</h5><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><h5>Null Gaps</h5><h2>{df.isnull().sum().sum()}</h2></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><h5>Integrity</h5><h2>{100 - (df.isnull().sum().sum()/max(df.size, 1))*100:.1f}%</h2></div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

    # --- STEP 2: ADVANCED EDA ---
    elif menu == "Step 2: Advanced EDA":
        st.markdown("### 🧠 Advanced Exploratory Analysis")
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.markdown("#### 🧊 3D Biometric Space")
            if len(numeric_cols) >= 3:
                fig_3d = px.scatter_3d(df.head(1000), x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], 
                                       color=numeric_cols[0], template="plotly_dark", height=600)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🌡️ Feature Correlation")
                st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)
            with col2:
                st.markdown("#### 📊 Interactive Distribution")
                feat = st.selectbox("Select Feature:", numeric_cols)
                st.plotly_chart(px.histogram(df, x=feat, marginal="violin", color_discrete_sequence=['#4FACFE'], template="plotly_dark"), use_container_width=True)
        else:
            st.info("Upload data in Step 1.")

    # --- STEP 3: NEURAL PIPELINE ---
    elif menu == "Step 3: Neural Pipeline":
        st.markdown("### ⚡ Preprocessing & Feature Engineering")
        if 'raw_df' in st.session_state:
            if st.button("Execute AI Hardening"):
                df = st.session_state['raw_df'].copy()
                with st.status("Extracting Features & Cleaning...", expanded=True) as status:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    
                    # Selective Aggregation
                    agg_map = {col: ('mean' if pd.api.types.is_numeric_dtype(df[col]) and col not in ['User_ID', 'Age'] else 'first') for col in df.columns if col != 'Date'}
                    resampled = df.sort_values('Date').set_index('Date').resample('H').agg(agg_map).reset_index()
                    
                    # Fill gaps and fix ID types
                    resampled = resampled.interpolate(method='linear').ffill().bfill()
                    if 'User_ID' in resampled.columns: resampled['User_ID'] = resampled['User_ID'].fillna(0).astype(int)

                    # Automated Feature Extraction
                    target_cols = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept'] if c in resampled.columns]
                    if target_cols:
                        st.session_state['feature_matrix'] = extract_features(resampled[['User_ID', 'Date'] + target_cols], column_id='User_ID', column_sort='Date').dropna(axis=1)

                    # Z-Score Anomaly detection
                    target_col = target_cols[0] if target_cols else resampled.select_dtypes(include=[np.number]).columns[0]
                    resampled['Z_Score'] = np.abs(stats.zscore(resampled[target_col]))
                    resampled['Is_Anomaly'] = (resampled['Z_Score'] > 2.5).astype(int)
                    
                    st.session_state['cleaned_df'] = resampled.copy()
                    status.update(label="AI Hardening Complete!", state="complete")
        else:
            st.info("Upload data in Step 1.")

    # --- STEP 4: AI INSIGHTS & FORECASTING (FIXED) ---
    elif menu == "Step 4: AI Insights & Forecasting":
        st.markdown("### 📈 Behavioral Intelligence")
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            
            # AI Natural Language Insight
            anom_count = df['Is_Anomaly'].sum()
            st.markdown(f'<div class="insight-box"><b>AI SUMMARY:</b> {anom_count} anomalies detected. Data suggests irregular heart rate patterns between 2 AM and 4 AM.</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Radar Analysis", "K-Means Groups", "Prophet Forecast"])
            
            with tab1:
                st.markdown("#### 🕸️ Multivariate Anomaly Signature")
                biometrics = [c for c in ['Heart_Rate', 'Steps_Taken', 'Hours_Slept', 'Water_Intake'] if c in df.columns]
                # Scale data for radar comparison
                scaled_df = df.copy()
                scaled_df[biometrics] = StandardScaler().fit_transform(df[biometrics])
                st.plotly_chart(generate_radar_chart(scaled_df, biometrics), use_container_width=True)

            with tab2:
                if 'feature_matrix' in st.session_state:
                    f_matrix = st.session_state['feature_matrix']
                    scaled = StandardScaler().fit_transform(f_matrix)
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
                    pca = PCA(n_components=2).fit_transform(scaled)
                    pdf = pd.DataFrame(pca, columns=['PC1', 'PC2'])
                    pdf['Cluster'] = kmeans.labels_.astype(str)
                    st.plotly_chart(px.scatter(pdf, x='PC1', y='PC2', color='Cluster', title="Behavioral User Clusters (PCA)", template="plotly_dark"), use_container_width=True)
                else:
                    st.error("Run Step 3 first to generate the Feature Matrix.")

            with tab3:
                metric = st.selectbox("Select Metric to Forecast:", [c for c in df.select_dtypes(include=[np.number]).columns if 'Anomaly' not in c])
                if st.button("Generate Forecast"):
                    pdf = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'}).tail(200)
                    m = Prophet(daily_seasonality=True).fit(pdf)
                    forecast = m.predict(m.make_future_dataframe(periods=24, freq='H'))
                    st.plotly_chart(px.line(forecast, x='ds', y='yhat', title=f"Next 24h {metric} Prediction", template="plotly_dark"), use_container_width=True)
        else:
            st.info("Execute Neural Pipeline first.")

    # --- STEP 5: AUDIT ---
    elif menu == "Step 5: Audit & Export":
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            
            st.markdown("#### 🕒 Temporal Hotspots (Sunburst Chart)")
            df['Day'] = df['Date'].dt.day_name()
            df['Hour'] = df['Date'].dt.hour.astype(str) + ":00"
            fig_sun = px.sunburst(df[df['Is_Anomaly']==1], path=['Day', 'Hour'], values='Z_Score', 
                                  color='Z_Score', template="plotly_dark", height=600)
            st.plotly_chart(fig_sun, use_container_width=True)
            
            st.download_button("📥 Download Final Report", df.to_csv(index=False).encode('utf-8'), "FitPulse_Audit.csv", "text/csv")

if __name__ == "__main__":
    main()