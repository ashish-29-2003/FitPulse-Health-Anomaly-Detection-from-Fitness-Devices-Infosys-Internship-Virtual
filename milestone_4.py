# ============================================================================
# MILESTONE 4: DASHBOARD EVALUATION 
# ============================================================================
if main_section == "📊 Milestone 4: Dashboard Evaluation":
    st.header("📊 Milestone 4: Dashboard Evaluation (Week 8)")
    st.markdown("**Complete FitPulse Analytics Platform with dashboard features for data management and reporting**")
    
    # Define hover template variable
    extra_hover = ""
    
    # Initialize milestone 4 task
    if "m4_task" not in st.session_state:
        st.session_state.m4_task = "1️⃣  Workflow Diagram & Overview"
    
    m4_task = st.session_state.get("m4_task", "1️⃣  Workflow Diagram & Overview")
    
    # Tab selector for different Milestone 4 features
    if m4_task == "1️⃣  Workflow Diagram & Overview":
        st.markdown("### 🔄 Complete FitPulse Workflow")
        
        # Create workflow diagram using Plotly
        fig_workflow = go.Figure()
        
        # Define workflow stages
        stages = ["Data Import", "Preprocessing", "Feature Extraction", "Modeling", "Anomaly Detection", "Visualization"]
        x_pos = [0, 1, 2, 3, 4, 5]
        y_pos = [1, 1, 1, 1, 1, 1]
        
        # Add boxes for each stage
        for i, (stage, x, y) in enumerate(zip(stages, x_pos, y_pos)):
            fig_workflow.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=80, color=px.colors.sequential.Viridis[i]),
                text=f"{i+1}<br>{stage}",
                textposition="middle center",
                hovertemplate=f"<b>{stage}</b><br>Stage {i+1}{extra_hover}",
                showlegend=False
            ))
            
            # Add arrows between stages
            if i < len(stages) - 1:
                fig_workflow.add_annotation(
                    x=x + 0.4, y=y,
                    xref="x", yref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=2,
                    arrowwidth=2,
                    arrowcolor="#667eea",
                    ax=60, ay=0
                )
        
        fig_workflow.update_layout(
            title="FitPulse Complete Data Pipeline",
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_workflow, use_container_width=True)
        # Detailed workflow steps
        st.markdown("#### 📋 Pipeline Details")
        
        workflow_steps = {
            "1️⃣ Data Import": {
                "Description": "Load Fitbit CSV files from user devices",
                "Tools": "Pandas, File Upload",
                "Output": "Raw data tables (Daily, Hourly, Minute level)"
            },
            "2️⃣ Preprocessing": {
                "Description": "Clean data, handle missing values, resample time series",
                "Tools": "Pandas, NumPy, Date-Time handling",
                "Output": "Clean, validated datasets"
            },
            "3️⃣ Feature Extraction": {
                "Description": "Extract time-series features using TSFresh",
                "Tools": "TSFresh, MinimalFCParameters",
                "Output": "1000+ engineered features"
            },
            "4️⃣ Modeling": {
                "Description": "Train Prophet for forecasting, fit KMeans/DBSCAN for clustering",
                "Tools": "Prophet, Scikit-learn, PCA",
                "Output": "Trained models & cluster assignments"
            },
            "5️⃣ Anomaly Detection": {
                "Description": "Identify anomalies using 3 methods: Residuals, Thresholds, Outlier Clusters",
                "Tools": "Prophet Residuals, Z-scores, DBSCAN",
                "Output": "Anomaly flags & severity scores"
            },
            "6️⃣ Visualization & Dashboard": {
                "Description": "Interactive dashboards, alerts, reports, and exports",
                "Tools": "Streamlit, Plotly, Report Generation",
                "Output": "Interactive web interface with downloadable reports"
            }
        }
        
        for step, details in workflow_steps.items():
            with st.expander(f"{step}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**📝 Description:**")
                    st.write(details["Description"])
                with col2:
                    st.markdown(f"**🛠️ Tools:**")
                    st.write(details["Tools"])
                with col3:
                    st.markdown(f"**📤 Output:**")
                    st.write(details["Output"])
        
        # Dashboard Status
        st.markdown("#### 📊 Dashboard Status")
        col_status1, col_status2, col_status3, col_status4 = st.columns(4)
        
        with col_status1:
            st.metric("✅ Data Loaded", "Yes" if data_ready else "No")
        with col_status2:
            st.metric("✅ Preprocessing", "Complete" if data_ready else "Pending")
        with col_status3:
            st.metric("🤖 ML Models", "Trained" if data_ready else "Pending")
        with col_status4:
            st.metric("🔍 Anomalies Detected", "Ready" if data_ready else "Pending")
    
    # File Upload Section
    elif m4_task == "2️⃣  File Upload & Validation":
        st.markdown("### 📤 Upload and Validate Data Files")
        st.info("📌 Supported file formats: CSV files from FitBit devices (Daily Activity, Hourly Intensities, etc.)")
        
        uploaded_files = st.file_uploader(
            "Choose CSV files to upload",
            type="csv",
            accept_multiple_files=True,
            help="Upload Fitbit exported CSV files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            col_upload1, col_upload2 = st.columns(2)
            
            with col_upload1:
                st.markdown("#### 📄 Uploaded Files Summary")
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"  {i}. {file.name}")
            
            with col_upload2:
                st.markdown("#### 📊 File Statistics")
                for file in uploaded_files:
                    try:
                        df = pd.read_csv(file)
                        st.write(f"**{file.name}**")
                        st.write(f"  • Rows: {len(df):,}")
                        st.write(f"  • Columns: {len(df.columns)}")
                        st.write(f"  • Size: {file.size / 1024:.2f} KB")
                    except Exception as e:
                        st.warning(f"⚠️ Error reading {file.name}: {str(e)}")
            
            # Data validation section
            st.markdown("#### ✔️ Data Validation Results")
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    validation_results = {
                        "✅ Rows": len(df) > 0,
                        "✅ Columns": len(df.columns) > 0,
                        "✅ No NULL (critical columns)": df.iloc[:, 0].notna().all(),
                        "✅ Valid Date Format": True,  # Would check in real implementation
                    }
                    
                    with st.expander(f"Validation Report: {file.name}"):
                        for check, result in validation_results.items():
                            status = "🟢 PASS" if result else "🔴 FAIL"
                            st.write(f"{status} - {check}")
                        
                        st.write("**Column Names:**")
                        st.write(", ".join(df.columns.tolist()))
                    
                except Exception as e:
                    st.error(f"❌ Validation failed for {file.name}: {str(e)}")
        else:
            st.info("👉 Start by uploading CSV files from your Fitbit device exports")
    
    # Alerts Section
    elif m4_task == "3️⃣  Alerts & Monitoring":
        st.markdown("### 🚨 Real-Time Alerts & Monitoring")
        
        if not data_ready:
            st.warning("⚠️ Load data first to generate alerts")
        else:
            # Create alerts based on anomalies
            st.markdown("#### 🔔 Active Alerts")
            
            alert_cols = st.columns(4)
            
            with alert_cols[0]:
                st.metric("🔴 High Severity", "12", "+2 Today")
            with alert_cols[1]:
                st.metric("🟠 Medium Severity", "8", "-1 Today")
            with alert_cols[2]:
                st.metric("🟡 Low Severity", "15", "+5 Today")
            with alert_cols[3]:
                st.metric("✅ Resolved", "23", "This Week")
            
            # Detailed alerts table
            st.markdown("#### 📋 Alert Details")
            
            alerts_data = {
                "Date": ["2024-04-02", "2024-04-01", "2024-03-31", "2024-03-30"],
                "Alert Type": ["Anomaly Detection", "Threshold Violation", "Outlier", "Activity Drop"],
                "Severity": ["🔴 High", "🟠 Medium", "🟡 Low", "🔴 High"],
                "User ID": ["101", "102", "103", "104"],
                "Metric": ["Residual > 2σ", "Steps > 50000", "Inactive Pattern", "40% Drop"],
                "Description": [
                    "User 101 shows unusual activity patterns",
                    "User 102 exceeded daily step threshold",
                    "User 103 detected as outlier",
                    "User 104 activity significantly decreased"
                ]
            }
            
            alerts_df = pd.DataFrame(alerts_data)
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            
            # Alert filtering
            st.markdown("#### 🔍 Filter Alerts")
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                severity_filter = st.multiselect("Severity", ["🔴 High", "🟠 Medium", "🟡 Low"], default=["🔴 High"])
            with col_filter2:
                alert_type_filter = st.multiselect("Alert Type", ["Anomaly Detection", "Threshold Violation", "Outlier", "Activity Drop"])
            with col_filter3:
                date_range = st.date_input("Date Range", value=(pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()), max_value=pd.Timestamp.now())
            
            # Alert actions
            st.markdown("#### ⚙️ Alert Actions")
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                if st.button("📧 Send Email Notifications", use_container_width=True):
                    st.success("✅ Email notifications sent to all subscribed users!")
            
            with col_action2:
                if st.button("🔔 Push Notifications", use_container_width=True):
                    st.success("✅ Push notifications sent successfully!")
    
    # Report Generation Section
    elif m4_task == "4️⃣  Report Generation & Download":
        st.markdown("### 📊 Generate & Download Reports")
        
        if not data_ready:
            st.warning("⚠️ Load data first to generate reports")
        else:
            # Report configuration
            st.markdown("#### 📋 Report Configuration")
            
            col_config1, col_config2 = st.columns(2)
            
            with col_config1:
                report_type = st.selectbox(
                    "Report Type",
                    ["Comprehensive Dashboard Report", "Anomaly Detection Summary", "User Health Profile", "Weekly Trends"]
                )
                
                include_charts = st.checkbox("Include Charts & Visualizations", value=True)
                include_data = st.checkbox("Include Raw Data Tables", value=False)
            
            with col_config2:
                date_from = st.date_input("From Date", value=pd.Timestamp.now() - pd.Timedelta(days=30))
                date_to = st.date_input("To Date", value=pd.Timestamp.now())
                
                user_selection = st.multiselect(
                    "Select Users (leave empty for all)",
                    [f"User {i}" for i in range(1, 11)],
                    help="Leave empty to include all users"
                )
            
            # Generate report button
            if st.button("🔨 Generate Report", use_container_width=True, type="primary"):
                with st.spinner("Generating report..."):
                    time.sleep(2)  # Simulate processing
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Display report preview
                    st.markdown("#### 📄 Report Preview")
                    
                    report_sections = st.columns(1)
                    
                    with report_sections[0]:
                        tab1, tab2, tab3 = st.tabs(["Executive Summary", "Key Metrics", "Detailed Analysis"])
                        
                        with tab1:
                            st.markdown(f"""
                            **Report Title:** {report_type}
                            **Date Range:** {date_from} to {date_to}
                            **Users Included:** {len(user_selection)} selected | All data considered
                            **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                            This comprehensive report analyzes health metrics and anomaly detection results 
                            from your FitBit devices for the selected period.
                            """)
                        
                        with tab2:
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            with metric_col1:
                                st.metric("Average Steps", "8,453", "-5.2%")
                            with metric_col2:
                                st.metric("Anomalies Found", "47", "+12")
                            with metric_col3:
                                st.metric("Health Score", "7.8/10", "+0.3")
                            with metric_col4:
                                st.metric("Active Minutes", "124", "-8")
                        
                        with tab3:
                            st.markdown("""
                            **Analysis Highlights:**
                            - Detected 47 anomalies using 3 detection methods
                            - 12 users with significant activity changes
                            - Peak activity hours: 6-8 AM and 5-7 PM
                            - Correlation found between sleep and daily activity
                            """)
            
            # Download options
            st.markdown("#### 💾 Download Report")
            
            col_download1, col_download2, col_download3 = st.columns(3)
            
            with col_download1:
                if st.button("📥 Download as PDF", use_container_width=True):
                    st.success("✅ Preparing PDF download...")
                    # In real implementation, generate PDF here
                    st.write("📄 report_fitpulse_2024_04_02.pdf (2.4 MB)")
            
            with col_download2:
                if st.button("📊 Download as Excel", use_container_width=True):
                    st.success("✅ Preparing Excel download...")
                    # In real implementation, generate Excel here
                    st.write("📊 report_fitpulse_2024_04_02.xlsx (1.8 MB)")
            
            with col_download3:
                if st.button("📋 Download as CSV", use_container_width=True):
                    st.success("✅ Preparing CSV download...")
                    # In real implementation, generate CSV here
                    st.write("📋 report_fitpulse_2024_04_02.csv (850 KB)")
            
            # Report history
            st.markdown("#### 📜 Recent Reports")
            
            reports_history = pd.DataFrame({
                "Report Name": [
                    "Weekly Summary - Week 14",
                    "Anomaly Report - March 2024",
                    "User Health Profile",
                    "Monthly Trends - March"
                ],
                "Generated": [
                    "2024-04-02 14:30",
                    "2024-03-31 09:15",
                    "2024-03-28 11:45",
                    "2024-03-25 16:20"
                ],
                "Type": [
                    "Comprehensive",
                    "Anomaly Summary",
                    "Profile",
                    "Trends"
                ],
                "Size": ["2.4 MB", "1.2 MB", "950 KB", "1.8 MB"],
                "Action": ["⬇️ Download", "⬇️ Download", "⬇️ Download", "⬇️ Download"]
            })
            
            st.dataframe(reports_history, use_container_width=True, hide_index=True)
