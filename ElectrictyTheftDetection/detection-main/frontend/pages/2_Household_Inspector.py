import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Resolve project root (two levels up from pages/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

st.set_page_config(page_title="Household Inspector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0f172a 100%); }
    
    .case-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255,255,255,0.05);
        margin: 8px 0;
    }
    .case-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .case-value { color: #e2e8f0; font-size: 1.3rem; font-weight: 600; }
    .risk-critical { color: #ef5350 !important; }
    .risk-high { color: #ff9800 !important; }
    .risk-suspicious { color: #ffc107 !important; }
    .risk-normal { color: #66bb6a !important; }
    
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #e2e8f0;
        margin: 20px 0 10px 0; padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(t=40, l=40, r=20, b=40),
)


def get_risk_class(tier):
    return {
        'Theft': 'risk-critical',
        'High Risk': 'risk-high',
        'Suspicious': 'risk-suspicious',
        'Normal': 'risk-normal'
    }.get(tier, '')


def main():
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔍 Household Deep-Dive Inspector
            </h1>
            <p style="color: #94a3b8;">Investigate individual consumer consumption patterns and model predictions</p>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    risk_path = os.path.join(PROJECT_ROOT, "data", "processed", "final_risk_scores.csv")
    ts_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_sgcc.csv")
    ensemble_path = os.path.join(PROJECT_ROOT, "data", "processed", "test_ensemble_scores.csv")
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features_sgcc.csv")
    
    if not os.path.exists(risk_path):
        st.error("❌ Missing data. Please run the training pipeline first.")
        return

    df = pd.read_csv(risk_path)
    ensemble_df = pd.read_csv(ensemble_path) if os.path.exists(ensemble_path) else None
    features_df = pd.read_csv(features_path) if os.path.exists(features_path) else None
    
    # ─────────────────────────────────────────────────
    # Sidebar: Consumer Selection
    # ─────────────────────────────────────────────────
    st.sidebar.markdown("### 🎯 Select Consumer")
    
    filter_tier = st.sidebar.multiselect(
        "Filter by Risk Tier:",
        ['Theft', 'High Risk', 'Suspicious', 'Normal'],
        default=['Theft', 'High Risk']
    )
    
    filtered_df = df[df['Risk_Tier'].isin(filter_tier)] if filter_tier else df
    
    if filtered_df.empty:
        st.warning("No consumers match the selected filters.")
        return
    
    # Sort by risk score 
    filtered_df = filtered_df.sort_values('Final_Risk_Score', ascending=False)
    consumer_list = filtered_df['CONS_NO'].tolist()
    
    selected_cons = st.sidebar.selectbox(
        f"Consumer ID ({len(consumer_list)} found):",
        consumer_list
    )
    
    case = df[df['CONS_NO'] == selected_cons].iloc[0]
    
    # ─────────────────────────────────────────────────
    # Panel 1: Case Summary Cards
    # ─────────────────────────────────────────────────
    risk_class = get_risk_class(case.get('Risk_Tier', 'Normal'))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''<div class="case-card">
            <div class="case-label">Consumer ID</div>
            <div class="case-value">{selected_cons}</div>
        </div>''', unsafe_allow_html=True)
    with col2:
        score_val = case.get('Final_Risk_Score', 0)
        st.markdown(f'''<div class="case-card">
            <div class="case-label">Risk Score</div>
            <div class="case-value {risk_class}">{score_val:.1f} / 100</div>
        </div>''', unsafe_allow_html=True)
    with col3:
        tier = case.get('Risk_Tier', 'Unknown')
        st.markdown(f'''<div class="case-card">
            <div class="case-label">Classification</div>
            <div class="case-value {risk_class}">{tier}</div>
        </div>''', unsafe_allow_html=True)
    with col4:
        rule = case.get('Top_Flagged_Rule', 'None')
        actual = "🔴 Theft" if case.get('FLAG', 0) == 1 else "🟢 Normal"
        st.markdown(f'''<div class="case-card">
            <div class="case-label">Ground Truth</div>
            <div class="case-value">{actual}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────────────
    # Panel 2: Model Scores Radar + Individual Scores
    # ─────────────────────────────────────────────────
    if ensemble_df is not None and selected_cons in ensemble_df['CONS_NO'].values:
        consumer_scores = ensemble_df[ensemble_df['CONS_NO'] == selected_cons].iloc[0]
        
        c1, c2 = st.columns([2, 3])
        
        with c1:
            st.markdown('<div class="section-header">🤖 Model Predictions</div>', unsafe_allow_html=True)
            
            model_data = {}
            if 'XGBoost_Probability' in consumer_scores.index:
                model_data['XGBoost'] = consumer_scores['XGBoost_Probability']
            if 'LSTM_Reconstruction_Error' in consumer_scores.index:
                lstm_val = consumer_scores['LSTM_Reconstruction_Error']
                # Normalize LSTM score
                lstm_min = ensemble_df['LSTM_Reconstruction_Error'].min()
                lstm_max = ensemble_df['LSTM_Reconstruction_Error'].max()
                model_data['LSTM'] = (lstm_val - lstm_min) / (lstm_max - lstm_min + 1e-6)
            if 'IF_Score' in consumer_scores.index:
                if_val = consumer_scores['IF_Score']
                if_min = ensemble_df['IF_Score'].min()
                if_max = ensemble_df['IF_Score'].max()
                model_data['Isolation Forest'] = (if_val - if_min) / (if_max - if_min + 1e-6)
            if 'LOF_Score' in consumer_scores.index:
                lof_val = consumer_scores['LOF_Score']
                lof_min = ensemble_df['LOF_Score'].min()
                lof_max = ensemble_df['LOF_Score'].max()
                model_data['LOF'] = (lof_val - lof_min) / (lof_max - lof_min + 1e-6)
            
            if model_data:
                # Radar chart
                categories = list(model_data.keys())
                values = list(model_data.values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(99, 102, 241, 0.2)',
                    line=dict(color='#6366f1', width=2),
                    name='Anomaly Score'
                ))
                fig_radar.update_layout(
                    **DARK_LAYOUT,
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], color='#475569'),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    height=350,
                    title="Model Score Radar"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        with c2:
            st.markdown('<div class="section-header">📊 Score Details</div>', unsafe_allow_html=True)
            
            score_display = []
            raw_cols = {
                'XGBoost_Probability': ('XGBoost', '#6366f1'),
                'LSTM_Reconstruction_Error': ('LSTM AE', '#f472b6'),
                'IF_Score': ('Isolation Forest', '#34d399'),
                'LOF_Score': ('LOF', '#fbbf24'),
            }
            
            for col, (name, color) in raw_cols.items():
                if col in consumer_scores.index:
                    val = consumer_scores[col]
                    # What percentile is this score?
                    all_scores = ensemble_df[col].values
                    percentile = (all_scores < val).sum() / len(all_scores) * 100
                    score_display.append({
                        'Model': name,
                        'Raw Score': f"{val:.6f}",
                        'Percentile': f"{percentile:.1f}%",
                        'Interpretation': "🔴 High" if percentile > 75 else ("🟡 Medium" if percentile > 50 else "🟢 Low")
                    })
            
            if score_display:
                st.dataframe(pd.DataFrame(score_display), use_container_width=True, hide_index=True)
            
            if rule and rule != 'None':
                st.warning(f"🚩 **Flagged Rule:** {rule}")

    # ─────────────────────────────────────────────────
    # Panel 3: Time-Series Consumption Plot
    # ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Consumption Time-Series Pattern</div>', unsafe_allow_html=True)
    
    if os.path.exists(ts_path):
        ts_df = pd.read_csv(ts_path)
        user_series = ts_df[ts_df['CONS_NO'] == selected_cons]
        
        if not user_series.empty:
            dates = [c for c in ts_df.columns if c not in ['CONS_NO', 'FLAG']]
            vals = user_series[dates].values.flatten()
            
            parsed_dates = pd.to_datetime(dates, errors='coerce')
            valid_mask = ~parsed_dates.isna()
            
            fig_ts = go.Figure()
            
            # Main consumption line
            fig_ts.add_trace(go.Scatter(
                x=parsed_dates[valid_mask],
                y=vals[valid_mask],
                mode='lines',
                name='Daily Consumption',
                line=dict(color='#6366f1', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))
            
            # Rolling average
            rolling_window = 7
            if len(vals[valid_mask]) > rolling_window:
                rolling_avg = pd.Series(vals[valid_mask]).rolling(rolling_window).mean()
                fig_ts.add_trace(go.Scatter(
                    x=parsed_dates[valid_mask],
                    y=rolling_avg,
                    mode='lines',
                    name=f'{rolling_window}-Day Moving Avg',
                    line=dict(color='#f472b6', width=2, dash='dot')
                ))
            
            # Highlight zero readings
            zero_mask = vals[valid_mask] == 0
            if zero_mask.any():
                fig_ts.add_trace(go.Scatter(
                    x=parsed_dates[valid_mask][zero_mask],
                    y=vals[valid_mask][zero_mask],
                    mode='markers',
                    name='Zero Readings ⚠️',
                    marker=dict(color='#ef5350', size=6, symbol='x')
                ))
            
            fig_ts.update_layout(
                **DARK_LAYOUT,
                title=f"Consumption Pattern — {selected_cons}",
                xaxis_title="Date",
                yaxis_title="Consumption (normalized)",
                height=450,
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No time-series data found for this consumer ID.")
    else:
        st.info("Clean time-series data not available.")

    # ─────────────────────────────────────────────────
    # Panel 4: Feature Profile
    # ─────────────────────────────────────────────────
    if features_df is not None and selected_cons in features_df['CONS_NO'].values:
        st.markdown("---")
        st.markdown('<div class="section-header">🧬 Feature Profile</div>', unsafe_allow_html=True)
        
        user_features = features_df[features_df['CONS_NO'] == selected_cons].iloc[0]
        feat_cols = [c for c in features_df.columns if c not in ['CONS_NO', 'FLAG']]
        
        # Show top features vs population
        feat_data = []
        for col in feat_cols[:15]:  # Top 15 features
            user_val = user_features[col]
            pop_mean = features_df[col].mean()
            pop_std = features_df[col].std()
            z_score = (user_val - pop_mean) / (pop_std + 1e-6)
            
            feat_data.append({
                'Feature': col,
                'User Value': f"{user_val:.4f}",
                'Population Mean': f"{pop_mean:.4f}",
                'Z-Score': f"{z_score:+.2f}",
                'Anomalous': "⚠️ Yes" if abs(z_score) > 2 else "✅ No"
            })
        
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────
    # Panel 5: Investigation Actions
    # ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📝 Investigation Actions</div>', unsafe_allow_html=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        visit_date = st.date_input("Schedule Site Visit")
        finding = st.selectbox("Investigation Finding", [
            "Pending Visit", "Meter Bypass Hook", "Magnetic Interference", 
            "Inside Meter Tamper", "Failed Component (No Theft)", "Normal / Error"
        ])
    with c_right:
        priority = st.selectbox("Priority Level", ["Critical", "High", "Medium", "Low"])
        notes = st.text_area("Inspector Notes")
    
    if st.button("✅ Submit Investigation Report", type="primary"):
        st.success(f"✅ Investigation report submitted for **{selected_cons}** — scheduled for {visit_date}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 15px; color: #475569;">
            <p>🔍 Household Inspector • Deep-dive analysis for revenue protection officers</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
