import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Setup page configuration (Must be first streamlit command)
st.set_page_config(
    page_title="⚡ AI Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add frontend root for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from components.data_loader import DashboardDataLoader
from components.risk_map import render_geographic_map, render_risk_hierarchy
from components.inference import InferenceEngine

# Premium Dark Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0f172a 100%);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .metric-green .metric-value {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-red .metric-value {
        background: linear-gradient(135deg, #ff1744 0%, #f44336 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-orange .metric-value {
        background: linear-gradient(135deg, #ff9100 0%, #ffab40 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-purple .metric-value {
        background: linear-gradient(135deg, #7c4dff 0%, #b388ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    .glassmorphism {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
    }
    
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .tier-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .tier-normal { background: rgba(76, 175, 80, 0.2); color: #66bb6a; border: 1px solid rgba(76, 175, 80, 0.3); }
    .tier-suspicious { background: rgba(255, 193, 7, 0.2); color: #ffd54f; border: 1px solid rgba(255, 193, 7, 0.3); }
    .tier-high { background: rgba(255, 152, 0, 0.2); color: #ffb74d; border: 1px solid rgba(255, 152, 0, 0.3); }
    .tier-theft { background: rgba(244, 67, 54, 0.2); color: #ef5350; border: 1px solid rgba(244, 67, 54, 0.3); }
</style>
""", unsafe_allow_html=True)


DARK_LAYOUT = {
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "template": "plotly_dark",
    "font": {"family": "Inter", "color": "#e2e8f0"},
    "margin": {"t": 40, "l": 40, "r": 20, "b": 40},
}

TIER_COLORS = {
    "Normal": "#4CAF50",
    "Suspicious": "#FFC107",
    "High Risk": "#FF9800",
    "Theft": "#F44336"
}


def render_metric_card(label, value, style_class=""):
    st.markdown(f'''
        <div class="metric-card {style_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    ''', unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <span style="font-size: 3rem;">⚡</span>
            <h2 style="color: #e2e8f0; margin: 10px 0 5px 0;">AI Theft Detection</h2>
            <p style="color: #64748b; font-size: 0.85rem;">Revenue Protection System v2.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("🔌 **Operational Dashboard** for Power Grid Revenue Protection, powered by 4-model ML Ensemble.")

    # Main Header
    st.markdown("""
        <div style="text-align: center; padding: 20px 0 30px 0;">
            <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ Electricity Theft Detection System
            </h1>
            <p style="color: #94a3b8; font-size: 1.1rem;">Real-time anomaly detection across the power distribution network</p>
        </div>
    """, unsafe_allow_html=True)

    # Data Upload / Loading Section
    st.sidebar.markdown('<div class="section-header">📂 Data Source</div>', unsafe_allow_html=True)
    
    loader = DashboardDataLoader()
    df = loader.load_risk_csv()
    
    if df is not None:
        st.sidebar.success("✅ Using cached pre-processed data.")
    else:
        st.markdown("""
            <div style="text-align: center; padding: 100px 0;">
                <span style="font-size: 5rem;">📥</span>
                <h2>Data Not Found</h2>
                <p style="color: #94a3b8;">Please make sure the processed risk data is available.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Model scores for distribution charts
    test_scores_df = df.copy() # Use processed df as fallback for charts

    # ─────────────────────────────────────────────────
    # TOP-LEVEL KPI METRICS
    # ─────────────────────────────────────────────────
    total_meters = len(df)
    theft_count = len(df[df['Risk_Tier'] == 'Theft']) if 'Risk_Tier' in df.columns else 0
    high_risk_count = len(df[df['Risk_Tier'].isin(['High Risk', 'Theft'])]) if 'Risk_Tier' in df.columns else 0
    suspicious_count = len(df[df['Risk_Tier'] == 'Suspicious']) if 'Risk_Tier' in df.columns else 0
    normal_count = len(df[df['Risk_Tier'] == 'Normal']) if 'Risk_Tier' in df.columns else 0
    theft_rate = (high_risk_count / total_meters) * 100 if total_meters > 0 else 0
    avg_score = df['Final_Risk_Score'].mean() if 'Final_Risk_Score' in df.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Meters", f"{total_meters:,}", "metric-green")
    with col2:
        render_metric_card("Confirmed Theft", f"{theft_count:,}", "metric-red")
    with col3:
        render_metric_card("High Risk", f"{high_risk_count:,}", "metric-orange")
    with col4:
        render_metric_card("Network Theft %", f"{theft_rate:.1f}%", "metric-purple")
    with col5:
        render_metric_card("Avg Risk Score", f"{avg_score:.1f}", "")

    st.markdown("---")

    # ─────────────────────────────────────────────────
    # ROW 1: Risk Distribution + Tier Breakdown
    # ─────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown('<div class="section-header">📊 Risk Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="Final_Risk_Score", color="Risk_Tier",
            color_discrete_map=TIER_COLORS,
            nbins=60, barmode="overlay", opacity=0.8
        )
        fig_hist.update_layout(**DARK_LAYOUT, title=None, xaxis_title="Risk Score", yaxis_title="Number of Households")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🎯 Tier Classification</div>', unsafe_allow_html=True)
        tier_counts = df['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        
        fig_pie = px.pie(
            tier_counts, values='Count', names='Tier',
            color='Tier', color_discrete_map=TIER_COLORS,
            hole=0.55
        )
        fig_pie.update_layout(**DARK_LAYOUT, showlegend=True, title=None)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                              textfont_size=12, pull=[0.05 if t == 'Theft' else 0 for t in tier_counts['Tier']])
        st.plotly_chart(fig_pie, use_container_width=True)

    # ─────────────────────────────────────────────────
    # ROW 2: Individual Model Scores Comparison
    # ─────────────────────────────────────────────────
    if test_scores_df is not None and 'FLAG' in test_scores_df.columns:
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 Individual Model Score Distributions (Test Set)</div>', unsafe_allow_html=True)
        
        model_cols_map = {
            'XGBoost_Probability': 'XGBoost',
            'LSTM_Reconstruction_Error': 'LSTM Autoencoder',
            'IF_Score': 'Isolation Forest',
            'LOF_Score': 'LOF'
        }
        
        available_models = {k: v for k, v in model_cols_map.items() if k in test_scores_df.columns}
        
        if available_models:
            cols = st.columns(len(available_models))
            for idx, (col_name, model_name) in enumerate(available_models.items()):
                with cols[idx]:
                    fig_box = px.box(
                        test_scores_df, y=col_name, color='FLAG',
                        color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                        labels={'FLAG': 'Label', col_name: 'Score'},
                        category_orders={'FLAG': [0, 1]}
                    )
                    fig_box.update_layout(
                        **DARK_LAYOUT,
                        title=dict(text=model_name, font=dict(size=14)),
                        showlegend=False, height=350,
                        yaxis_title="Score"
                    )
                    # Update legend names
                    fig_box.for_each_trace(lambda t: t.update(name="Normal" if t.name == "0" else "Theft"))
                    st.plotly_chart(fig_box, use_container_width=True)

    # ─────────────────────────────────────────────────
    # ROW 3: Network Hierarchy + Top Flagged Rules
    # ─────────────────────────────────────────────────
    st.markdown("---")
    c3, c4 = st.columns([3, 2])
    
    with c3:
        st.markdown('<div class="section-header">🌐 Geographic Risk Distribution (Google Maps Integration)</div>', unsafe_allow_html=True)
        render_geographic_map(df)

    with c4:
        st.markdown('<div class="section-header">🚩 Top Flagged Domain Rules</div>', unsafe_allow_html=True)
        if 'Top_Flagged_Rule' in df.columns:
            rule_counts = df[df['Top_Flagged_Rule'] != 'None']['Top_Flagged_Rule'].value_counts().head(10).reset_index()
            rule_counts.columns = ['Rule', 'Cases']
            
            fig_rules = px.bar(
                rule_counts, x='Cases', y='Rule', orientation='h',
                color='Cases', color_continuous_scale='Reds'
            )
            fig_rules.update_layout(**DARK_LAYOUT, title=None, coloraxis_showscale=False, height=400)
            st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.info("Domain rules not available yet.")

    # ─────────────────────────────────────────────────
    # ROW 4: Top 20 Most Suspicious Consumers
    # ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔴 Top 20 Highest Risk Consumers</div>', unsafe_allow_html=True)
    
    if 'Final_Risk_Score' in df.columns and 'CONS_NO' in df.columns:
        top20 = df.nlargest(20, 'Final_Risk_Score')[['CONS_NO', 'Final_Risk_Score', 'Risk_Tier', 'Top_Flagged_Rule']].reset_index(drop=True)
        top20.index = top20.index + 1
        
        st.dataframe(
            top20.style.background_gradient(subset=['Final_Risk_Score'], cmap='RdYlGn_r'),
            height=400
        )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: #475569;">
            <p>⚡ AI Electricity Theft Detection System • 4-Model Ensemble (IF + LOF + LSTM + XGBoost)</p>
            <p style="font-size: 0.8rem;">Powered by Machine Learning & Domain Expert Rules</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
