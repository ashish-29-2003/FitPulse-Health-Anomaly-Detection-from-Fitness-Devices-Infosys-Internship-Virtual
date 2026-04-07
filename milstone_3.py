# --- MILESTONE 3: ANOMALY DETECTION & VISUALIZATION ---
if main_section == "🔍 Milestone 3: Anomaly Detection":
    st.markdown("<h1>🔍 Milestone 3: Anomaly Detection & Visualization</h1>", unsafe_allow_html=True)
    st.markdown("Tasks 28-35: Advanced Anomaly Detection - Outliers, Threshold Violations & Interactive Alerts")
    st.markdown("---")
    
    # Task 28-30: Model Residuals Based Anomaly Detection
    if milestone3_task == "1️⃣  Task 28-30: Model Residuals Anomaly":
        st.markdown("""
        <div class='task-box'>
        <h3>📊 Task 28-30: Detect Anomalies via Model Residuals</h3>
        <p><b>Objective:</b> Use Prophet model residuals to identify unusual patterns in time series data</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶️ Execute Residual-Based Anomaly Detection"):
            with st.spinner("🔄 Analyzing model residuals for anomalies..."):
                try:
                    # Prepare heart rate data
                    if "Time" in hr.columns and "Value" in hr.columns:
                        hr_clean = hr[hr['Value'] > 0].copy()
                        hr_clean["Date"] = pd.to_datetime(hr_clean["Time"], errors='coerce').dt.date
                        hr_clean = hr_clean.dropna(subset=["Date"])
                        
                        # Daily average heart rate
                        hr_daily = hr_clean.groupby("Date")['Value'].mean().reset_index()
                        hr_daily.columns = ['ds', 'y']
                        hr_daily['ds'] = pd.to_datetime(hr_daily['ds'])
                        hr_daily = hr_daily.sort_values('ds').reset_index(drop=True)
                        
                        if len(hr_daily) >= 10:
                            # Fit Prophet model
                            m = Prophet(interval_width=0.85, daily_seasonality=False)
                            m.fit(hr_daily)
                            forecast = m.predict(hr_daily[['ds']])
                            
                            # Calculate residuals
                            residuals = hr_daily['y'].values - forecast['yhat'].values
                            residual_std = np.std(residuals)
                            residual_mean = np.mean(residuals)
                            
                            # Identify anomalies (> 2 standard deviations from mean)
                            threshold = 2.0
                            anomaly_mask = np.abs(residuals - residual_mean) > (threshold * residual_std)
                            
                            # Add anomaly info to dataframe
                            hr_daily['residual'] = residuals
                            hr_daily['is_anomaly'] = anomaly_mask
                            hr_daily['residual_zscore'] = np.abs((residuals - residual_mean) / residual_std)
                            
                            anomaly_count = anomaly_mask.sum()
                            anomaly_pct = (anomaly_count / len(hr_daily)) * 100
                            
                            # Display metrics
                            st.success(f"✅ Detected {anomaly_count} anomalies ({anomaly_pct:.1f}%) in heart rate data")
                            
                            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                            with col_metric1:
                                st.metric("📊 Total Days", len(hr_daily))
                            with col_metric2:
                                st.metric("🚨 Anomalies Found", anomaly_count)
                            with col_metric3:
                                st.metric("📈 Residual Std", f"{residual_std:.2f}")
                            with col_metric4:
                                st.metric("📉 Threshold (σ)", f"±{threshold:.1f}")
                            
                            # Visualization: Time series with residuals and anomalies highlighted
                            st.markdown("#### Residual Analysis Chart")
                            
                            fig_residuals = go.Figure()
                            
                            # Original data
                            fig_residuals.add_trace(go.Scatter(
                                x=hr_daily['ds'],
                                y=hr_daily['y'],
                                mode='lines+markers',
                                name='Actual HR',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6)
                            ))
                            
                            # Forecast line
                            fig_residuals.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['yhat'],
                                mode='lines',
                                name='Prophet Forecast',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            # Anomalies highlighted
                            anomalies = hr_daily[hr_daily['is_anomaly']]
                            if len(anomalies) > 0:
                                fig_residuals.add_trace(go.Scatter(
                                    x=anomalies['ds'],
                                    y=anomalies['y'],
                                    mode='markers',
                                    name='🚨 Anomalies',
                                    marker=dict(
                                        size=12,
                                        color='#FF6B6B',
                                        symbol='diamond',
                                        line=dict(width=2, color='#FFB700')
                                    ),
                                    text=[f"Anomaly at {d.strftime('%Y-%m-%d')}<br>HR: {v:.0f} BPM<br>Z-score: {z:.2f}" 
                                          for d, v, z in zip(anomalies['ds'], anomalies['y'], anomalies['residual_zscore'])],
                                    hovertemplate='%{text}<extra></extra>'
                                ))
                            
                            # Confidence interval
                            fig_residuals.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['yhat_upper'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,100,80,0)',
                                showlegend=False
                            ))
                            fig_residuals.add_trace(go.Scatter(
                                x=forecast['ds'],
                                y=forecast['yhat_lower'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,100,80,0)',
                                name='85% Confidence Interval',
                                fillcolor='rgba(31, 119, 180, 0.2)'
                            ))
                            
                            fig_residuals.update_layout(
                                title="<b>❤️ Heart Rate: Residual-Based Anomaly Detection</b><br><sub>Points outside confidence interval = anomalies</sub>",
                                xaxis_title="Date",
                                yaxis_title="Heart Rate (BPM)",
                                template='plotly_dark',
                                hovermode='x unified',
                                height=600,
                                font=dict(size=12)
                            )
                            st.plotly_chart(fig_residuals, use_container_width=True)
                            
                            # Residual distribution chart - SIMPLIFIED FOR CLARITY
                            st.markdown("#### Residual Distribution Analysis")
                            
                            # Single comprehensive chart combining distribution and time series
                            fig_combined = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=("📊 Residual Distribution", "📈 Anomalies Over Time"),
                                specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
                                vertical_spacing=0.15
                            )
                            
                            # Histogram in top subplot
                            fig_combined.add_trace(
                                go.Histogram(
                                    x=residuals,
                                    nbinsx=25,
                                    name='Distribution',
                                    marker=dict(color='#4ECDC4', line=dict(color='#2A9D8F', width=1)),
                                    opacity=0.8,
                                    showlegend=True
                                ),
                                row=1, col=1
                            )
                            
                            # Add threshold band for top chart
                            upper_threshold = residual_mean + (threshold * residual_std)
                            lower_threshold = residual_mean - (threshold * residual_std)
                            
                            fig_combined.add_vline(x=upper_threshold, line_dash="dash", line_color="#FF6B6B", row=1, col=1)
                            fig_combined.add_vline(x=lower_threshold, line_dash="dash", line_color="#FF6B6B", row=1, col=1)
                            fig_combined.add_vline(x=residual_mean, line_dash="dot", line_color="#FFD700", row=1, col=1)
                            
                            # Time series in bottom subplot
                            fig_combined.add_trace(
                                go.Scatter(
                                    x=hr_daily['ds'],
                                    y=hr_daily['residual_zscore'],
                                    mode='lines',
                                    name='Z-Score Trend',
                                    line=dict(color='#45B7D1', width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(69, 183, 209, 0.2)',
                                    showlegend=True
                                ),
                                row=2, col=1
                            )
                            
                            # Highlight anomalies clearly
                            anomalies_zscore = hr_daily[hr_daily['is_anomaly']]
                            if len(anomalies_zscore) > 0:
                                fig_combined.add_trace(
                                    go.Scatter(
                                        x=anomalies_zscore['ds'],
                                        y=anomalies_zscore['residual_zscore'],
                                        mode='markers',
                                        name='🚨 Anomalies',
                                        marker=dict(size=12, color='#FF6B6B', symbol='star', 
                                                  line=dict(color='#FFB700', width=2)),
                                        showlegend=True
                                    ),
                                    row=2, col=1
                                )
                            
                            # Add threshold line to bottom chart
                            fig_combined.add_hline(y=threshold, line_dash="dash", line_color="#FF6B6B",
                                                  annotation_text=f"Anomaly Threshold ({threshold}σ)", 
                                                  annotation_position="right", row=2, col=1)
                            
                            fig_combined.update_xaxes(title_text="Residual Value", row=1, col=1)
                            fig_combined.update_xaxes(title_text="Date", row=2, col=1)
                            fig_combined.update_yaxes(title_text="Frequency", row=1, col=1)
                            fig_combined.update_yaxes(title_text="Z-Score", row=2, col=1)
                            
                            fig_combined.update_layout(
                                title="<b>📊 Residual Analysis: Distribution & Time Series</b><br><sub>Top: How residuals are distributed | Bottom: When anomalies occurred</sub>",
                                template='plotly_dark',
                                height=700,
                                font=dict(size=11),
                                hovermode='x unified',
                                showlegend=True
                            )
                            st.plotly_chart(fig_combined, use_container_width=True)
                            
                            # Anomalies details table
                            if len(anomalies) > 0:
                                st.markdown("#### 🚨 Detected Anomalies Details")
                                anomaly_details = anomalies[['ds', 'y', 'residual', 'residual_zscore']].copy()
                                anomaly_details.columns = ['Date', 'HR (BPM)', 'Residual', 'Z-Score']
                                anomaly_details['Date'] = anomaly_details['Date'].dt.strftime('%Y-%m-%d')
                                st.dataframe(anomaly_details.reset_index(drop=True), use_container_width=True, hide_index=True)
                        else:
                            st.warning("⚠️ Need at least 10 days of heart rate data for residual analysis")
                    else:
                        st.error("❌ Heart rate data not available")
                except Exception as e:
                    st.error(f"Error in residual analysis: {str(e)}")
    
    # Task 31-33: Threshold-Based Anomaly Detection
    elif milestone3_task == "2️⃣  Task 31-33: Threshold Violations":
        st.markdown("""
        <div class='task-box'>
        <h3>⚠️ Task 31-33: Detect Anomalies via Threshold Violations</h3>
        <p><b>Objective:</b> Flag activity and health metrics that exceed normal ranges</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶️ Execute Threshold-Based Anomaly Detection"):
            with st.spinner("🔄 Analyzing threshold violations..."):
                try:
                    # Define thresholds based on health guidelines
                    thresholds = {
                        'daily_steps': {'low': 3000, 'high': 25000},
                        'calories': {'low': 1200, 'high': 5000},
                        'very_active_min': {'low': 0, 'high': 120},
                        'sedentary_min': {'low': 0, 'high': 1200}
                    }
                    
                    # Analyze daily activity
                    anomalies_list = []
                    
                    for idx, row in daily.iterrows():
                        warnings = []
                        
                        # Steps check
                        if row['TotalSteps'] < thresholds['daily_steps']['low']:
                            warnings.append(f"Very low steps ({row['TotalSteps']:.0f})")
                        elif row['TotalSteps'] > thresholds['daily_steps']['high']:
                            warnings.append(f"Exceptionally high steps ({row['TotalSteps']:.0f})")
                        
                        # Calories check
                        if row['Calories'] > thresholds['calories']['high']:
                            warnings.append(f"Unusually high calories ({row['Calories']:.0f})")
                        elif row['Calories'] < thresholds['calories']['low'] and row['TotalSteps'] > 5000:
                            warnings.append(f"Low calories despite high activity")
                        
                        # Activity level check
                        if row['VeryActiveMinutes'] > thresholds['very_active_min']['high']:
                            warnings.append(f"Excessive intense activity ({row['VeryActiveMinutes']:.0f} min)")
                        
                        # Sedentary check
                        if row['SedentaryMinutes'] > thresholds['sedentary_min']['high']:
                            warnings.append(f"Excessive sedentary time ({row['SedentaryMinutes']:.0f} min)")
                        
                        if warnings:
                            anomalies_list.append({
                                'Date': row.get('ActivityDate', row.get('Date', idx)),
                                'Steps': row['TotalSteps'],
                                'Calories': row['Calories'],
                                'VeryActive': row['VeryActiveMinutes'],
                                'Sedentary': row['SedentaryMinutes'],
                                'Warnings': ' | '.join(warnings),
                                'Severity': len(warnings)
                            })
                    
                    anomalies_df = pd.DataFrame(anomalies_list)
                    
                    if len(anomalies_df) > 0:
                        anomalies_df = anomalies_df.sort_values('Severity', ascending=False)
                        
                        # Display metrics
                        st.success(f"✅ Detected {len(anomalies_df)} days with threshold violations")
                        
                        col_thresh1, col_thresh2, col_thresh3, col_thresh4 = st.columns(4)
                        with col_thresh1:
                            st.metric("📊 Total Days Analyzed", len(daily))
                        with col_thresh2:
                            st.metric("⚠️ Anomalies", len(anomalies_df))
                        with col_thresh3:
                            severity_high = len(anomalies_df[anomalies_df['Severity'] >= 2])
                            st.metric("🔴 High Severity", severity_high)
                        with col_thresh4:
                            anomaly_pct = (len(anomalies_df) / len(daily)) * 100
                            st.metric("📈 Anomaly %", f"{anomaly_pct:.1f}%")
                        
                        st.markdown("---")
                        
                        # Interactive timeline visualization - IMPROVED CLARITY
                        st.markdown("#### Threshold Violations Timeline")
                        
                        daily_copy = daily.copy()
                        date_col = "ActivityDate" if "ActivityDate" in daily_copy.columns else "Date"
                        daily_copy['Date_str'] = pd.to_datetime(daily_copy[date_col]).dt.strftime('%Y-%m-%d')
                        daily_copy['is_anomaly'] = daily_copy['Date_str'].isin(anomalies_df['Date'].astype(str).values)
                        daily_copy = daily_copy.sort_values(date_col)
                        
                        # Create clearer bar chart
                        fig_timeline = go.Figure()
                        
                        # Background bars for all days (colored by status)
                        colors = ['#FF6B6B' if x else '#4ECDC4' for x in daily_copy['is_anomaly']]
                        
                        fig_timeline.add_trace(go.Bar(
                            x=pd.to_datetime(daily_copy[date_col]),
                            y=daily_copy['TotalSteps'],
                            marker=dict(
                                color=colors,
                                line=dict(color=['#FFB700' if x else '#2A9D8F' for x in daily_copy['is_anomaly']], width=2)
                            ),
                            name='Daily Steps',
                            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Steps: %{y:.0f}<extra></extra>',
                            opacity=0.8
                        ))
                        
                        # Add trend line overlay
                        normal_days = daily_copy[~daily_copy['is_anomaly']]
                        fig_timeline.add_trace(go.Scatter(
                            x=pd.to_datetime(normal_days[date_col]),
                            y=normal_days['TotalSteps'],
                            mode='lines',
                            name='Normal Trend',
                            line=dict(color='#FFD700', width=2, dash='dash'),
                            hoverinfo='skip'
                        ))
                        
                        fig_timeline.update_layout(
                            title="<b>📈 Daily Steps: Normal vs Violations</b><br><sub>🔵 Blue = Normal Days | 🔴 Red = Threshold Violations</sub>",
                            xaxis_title="Date",
                            yaxis_title="Daily Steps",
                            template='plotly_dark',
                            hovermode='x unified',
                            height=500,
                            font=dict(size=11),
                            legend=dict(x=0.01, y=0.99)
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Threshold violation details
                        st.markdown("#### Alert Details (Sorted by Severity)")
                        st.dataframe(
                            anomalies_df[['Date', 'Steps', 'Calories', 'VeryActive', 'Sedentary', 'Warnings']],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.success("✅ No threshold violations detected! All metrics within normal ranges.")
                        
                except Exception as e:
                    st.error(f"Error in threshold analysis: {str(e)}")
    
    # Task 34-35: Outlier Cluster Based Anomaly Detection
    elif milestone3_task == "3️⃣  Task 34-35: Outlier Cluster Detection":
        st.markdown("""
        <div class='task-box'>
        <h3>🎯 Task 34-35: Detect Anomalies via Clustering</h3>
        <p><b>Objective:</b> Use DBSCAN to identify outliers in activity patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶️ Execute Cluster-Based Anomaly Detection"):
            with st.spinner("🔄 Running cluster-based anomaly detection..."):
                try:
                    # Prepare clustering features
                    cluster_features = daily.groupby("Id")[["TotalSteps", "Calories", "VeryActiveMinutes", 
                                                            "FairlyActiveMinutes", "LightlyActiveMinutes", 
                                                            "SedentaryMinutes"]].mean().reset_index()
                    
                    X = cluster_features[["TotalSteps", "Calories", "VeryActiveMinutes", 
                                         "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]].values
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply DBSCAN
                    dbscan = DBSCAN(eps=1.5, min_samples=2)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_anomalies = list(labels).count(-1)
                    anomaly_pct = (n_anomalies / len(labels)) * 100 if len(labels) > 0 else 0
                    
                    # Display metrics
                    st.success(f"✅ Detected {n_anomalies} anomalous users ({anomaly_pct:.1f}%)")
                    
                    col_cluster1, col_cluster2, col_cluster3, col_cluster4 = st.columns(4)
                    with col_cluster1:
                        st.metric("👥 Total Users", len(cluster_features))
                    with col_cluster2:
                        st.metric("🎯 Normal Clusters", n_clusters)
                    with col_cluster3:
                        st.metric("🚨 Outliers", n_anomalies)
                    with col_cluster4:
                        st.metric("Outlier %", f"{anomaly_pct:.1f}%")
                    
                    st.markdown("---")
                    
                    # PCA visualization
                    st.markdown("#### Anomalous Users Visualization (PCA 2D)")
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    viz_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Label': ['🚨 Anomaly' if x == -1 else f'Cluster' for x in labels],
                        'Steps': cluster_features['TotalSteps'].values,
                        'Calories': cluster_features['Calories'].values,
                        'User_ID': cluster_features['Id'].values,
                        'is_anomaly': labels == -1
                    })
                    
                    fig_pca = go.Figure()
                    
                    # Normal clusters
                    normal_df = viz_df[~viz_df['is_anomaly']]
                    fig_pca.add_trace(go.Scatter(
                        x=normal_df['PC1'],
                        y=normal_df['PC2'],
                        mode='markers',
                        name='Normal Users',
                        marker=dict(
                            size=10,
                            color='#4ECDC4',
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        text=[f"User {int(uid)}<br>Steps: {s:.0f}<br>Calories: {c:.0f}" 
                              for uid, s, c in zip(normal_df['User_ID'], normal_df['Steps'], normal_df['Calories'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    # Anomalies
                    anomaly_df = viz_df[viz_df['is_anomaly']]
                    if len(anomaly_df) > 0:
                        fig_pca.add_trace(go.Scatter(
                            x=anomaly_df['PC1'],
                            y=anomaly_df['PC2'],
                            mode='markers',
                            name='🚨 Anomalies',
                            marker=dict(
                                size=14,
                                color='#FF6B6B',
                                symbol='diamond',
                                opacity=0.9,
                                line=dict(width=2, color='#FFB700')
                            ),
                            text=[f"⚠️ User {int(uid)}<br>Steps: {s:.0f}<br>Calories: {c:.0f}<br>STATUS: ANOMALY" 
                                  for uid, s, c in zip(anomaly_df['User_ID'], anomaly_df['Steps'], anomaly_df['Calories'])],
                            hovertemplate='%{text}<extra></extra>'
                        ))
                    
                    fig_pca.update_layout(
                        title="<b>Outlier Detection via DBSCAN Clustering</b><br><sub>Users with unusual activity patterns highlighted</sub>",
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                        template='plotly_dark',
                        height=600,
                        hovermode='closest',
                        font=dict(size=11)
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Anomalies comparison chart - CLEARER REPRESENTATION
                    if len(anomaly_df) > 0:
                        st.markdown("#### Anomalous Users Profile Analysis")
                        
                        anom_users = cluster_features.iloc[labels == -1]
                        normal_users = cluster_features.iloc[labels != -1]
                        
                        # More understandable grouped bar chart instead of radar
                        if len(anom_users) > 0:
                            anom_avg = anom_users[["TotalSteps", "Calories", "VeryActiveMinutes", "SedentaryMinutes"]].mean()
                            normal_avg = normal_users[["TotalSteps", "Calories", "VeryActiveMinutes", "SedentaryMinutes"]].mean()
                            
                            # Create comparison dataframe
                            comparison_data = pd.DataFrame({
                                'Metric': ['Daily Steps', 'Calories Burned', 'Active Minutes', 'Sedentary Minutes'],
                                'Anomalous Users': [anom_avg['TotalSteps'], anom_avg['Calories'], 
                                                   anom_avg['VeryActiveMinutes'], anom_avg['SedentaryMinutes']],
                                'Normal Users': [normal_avg['TotalSteps'], normal_avg['Calories'], 
                                               normal_avg['VeryActiveMinutes'], normal_avg['SedentaryMinutes']]
                            })
                            
                            # Create grouped bar chart - EASY TO READ
                            fig_compare = go.Figure()
                            
                            fig_compare.add_trace(go.Bar(
                                x=comparison_data['Metric'],
                                y=comparison_data['Anomalous Users'],
                                name='🚨 Anomalous Users',
                                marker=dict(color='#FF6B6B', line=dict(width=2, color='#FFB700')),
                                hovertemplate='<b>%{x}</b><br>Value: %{y:.0f}<extra></extra>'
                            ))
                            
                            fig_compare.add_trace(go.Bar(
                                x=comparison_data['Metric'],
                                y=comparison_data['Normal Users'],
                                name='✅ Normal Users',
                                marker=dict(color='#4ECDC4', line=dict(width=2, color='#2A9D8F')),
                                hovertemplate='<b>%{x}</b><br>Value: %{y:.0f}<extra></extra>'
                            ))
                            
                            fig_compare.update_layout(
                                title="<b>Profile Comparison: Anomalous vs Normal Users</b><br><sub>Easy comparison of key metrics</sub>",
                                barmode='group',
                                xaxis_title="Health Metrics",
                                yaxis_title="Average Value",
                                template='plotly_dark',
                                height=500,
                                font=dict(size=11),
                                hovermode='x unified',
                                legend=dict(x=0.01, y=0.99)
                            )
                            st.plotly_chart(fig_compare, use_container_width=True)
                            
                            # Add summary statistics for clarity
                            st.markdown("**📊 Profile Differences at a Glance:**")
                            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                            
                            with col_stats1:
                                diff = ((anom_avg['TotalSteps'] - normal_avg['TotalSteps']) / normal_avg['TotalSteps'] * 100)
                                st.metric("Steps Difference", f"{diff:+.1f}%", "vs Normal")
                            
                            with col_stats2:
                                diff = ((anom_avg['Calories'] - normal_avg['Calories']) / normal_avg['Calories'] * 100)
                                st.metric("Calories Difference", f"{diff:+.1f}%", "vs Normal")
                            
                            with col_stats3:
                                diff = ((anom_avg['VeryActiveMinutes'] - normal_avg['VeryActiveMinutes']) / normal_avg['VeryActiveMinutes'] * 100)
                                st.metric("Active Min Difference", f"{diff:+.1f}%", "vs Normal")
                            
                            with col_stats4:
                                diff = ((anom_avg['SedentaryMinutes'] - normal_avg['SedentaryMinutes']) / normal_avg['SedentaryMinutes'] * 100)
                                st.metric("Sedentary Difference", f"{diff:+.1f}%", "vs Normal")
                        
                        # Detailed anomalies table
                        st.markdown("#### Anomalous Users Details")
                        anom_display = cluster_features.iloc[labels == -1][["Id", "TotalSteps", "Calories", "VeryActiveMinutes", "SedentaryMinutes"]].copy()
                        anom_display.columns = ["User ID", "Avg Steps", "Avg Calories", "Avg Active Min", "Avg Sedentary Min"]
                        st.dataframe(anom_display.reset_index(drop=True), use_container_width=True, hide_index=True)
                        
                        # Reason analysis
                        st.markdown("#### Why These Users Are Anomalies?")
                        for idx, (user_id, row) in enumerate(zip(anom_display["User ID"], anom_display.values)):
                            if idx >= 5:  # Show only first 5
                                st.info(f"... and {len(anom_display)-5} more anomalous users")
                                break
                            
                            reason = []
                            if row[1] < 2000:
                                reason.append("Very sedentary (< 2000 steps)")
                            elif row[1] > 30000:
                                reason.append("Extremely active (> 30000 steps)")
                            
                            if row[2] < 1000:
                                reason.append("Anomalously low calories")
                            elif row[2] > 6000:
                                reason.append("Unusually high calorie burn")
                            
                            if row[3] > 120:
                                reason.append("Excessive very active time")
                            
                            if len(reason) == 0:
                                reason.append("Unique pattern not matching normal profiles")
                            
                            st.warning(f"👤 User {int(user_id)}: {' | '.join(reason)}")
                    else:
                        st.success("✅ No outliers detected! All users follow normal activity patterns.")
                        
                except Exception as e:
                    st.error(f"Error in cluster-based analysis: {str(e)}")