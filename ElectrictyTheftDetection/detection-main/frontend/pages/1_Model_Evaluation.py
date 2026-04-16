import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Resolve project root (two levels up from pages/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0f172a 100%); }
    
    .model-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 12px;
        text-align: center;
    }
    .model-name {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }
    .model-auc {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 8px 0;
    }
    .auc-excellent { color: #00e676; }
    .auc-good { color: #66bb6a; }
    .auc-fair { color: #ffc107; }
    .auc-poor { color: #ff5722; }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(t=50, l=50, r=20, b=50),
)

MODEL_COLORS = {
    'XGBoost': '#6366f1',
    'LSTM Autoencoder': '#f472b6',
    'Isolation Forest': '#34d399',
    'LOF': '#fbbf24',
    'Ensemble': '#f43f5e'
}


def get_auc_class(auc_val):
    if auc_val >= 0.9: return "auc-excellent"
    elif auc_val >= 0.75: return "auc-good"
    elif auc_val >= 0.6: return "auc-fair"
    else: return "auc-poor"


def normalize_scores(scores):
    """Min-max normalize scores to [0, 1]"""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def compute_ensemble_score(df):
    """Compute weighted ensemble from available model scores."""
    score = np.zeros(len(df))
    weights_used = 0
    
    if 'XGBoost_Probability' in df.columns:
        score += df['XGBoost_Probability'].values * 0.50
        weights_used += 0.50
    if 'LSTM_Reconstruction_Error' in df.columns:
        score += normalize_scores(df['LSTM_Reconstruction_Error'].values) * 0.20
        weights_used += 0.20
    if 'IF_Score' in df.columns:
        score += normalize_scores(df['IF_Score'].values) * 0.15
        weights_used += 0.15
    if 'LOF_Score' in df.columns:
        score += normalize_scores(df['LOF_Score'].values) * 0.15
        weights_used += 0.15
    
    if weights_used > 0:
        score = score / weights_used  # Re-normalize by total weight
    return score


def main():
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📊 Model Performance & Evaluation Metrics
            </h1>
            <p style="color: #94a3b8; font-size: 1rem;">Comprehensive analysis of all 4 ML models + ensemble</p>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    test_path = os.path.join(PROJECT_ROOT, "data", "processed", "test_ensemble_scores.csv")
    train_path = os.path.join(PROJECT_ROOT, "data", "processed", "train_ensemble_scores.csv")
    
    if not os.path.exists(test_path):
        st.error("❌ No test scores found. Please run the training pipeline first: `python src/pipeline/train_pipeline.py`")
        return
    
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    
    if 'FLAG' not in test_df.columns:
        st.error("❌ FLAG column not found in test scores. Cannot compute evaluation metrics.")
        return
    
    y_test = test_df['FLAG'].values

    # ─────────────────────────────────────────────────
    # SECTION 1: Model AUC-ROC Cards
    # ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Model AUC-ROC Scores</div>', unsafe_allow_html=True)
    
    model_scores = {}
    score_columns = {
        'XGBoost_Probability': 'XGBoost',
        'LSTM_Reconstruction_Error': 'LSTM Autoencoder',
        'IF_Score': 'Isolation Forest',
        'LOF_Score': 'LOF'
    }
    
    available_models = {}
    for col, name in score_columns.items():
        if col in test_df.columns:
            scores = test_df[col].values
            try:
                auc_val = roc_auc_score(y_test, scores)
                model_scores[name] = {'auc': auc_val, 'scores': scores, 'col': col}
                available_models[name] = scores
            except:
                pass
    
    # Add ensemble
    ensemble_scores = compute_ensemble_score(test_df)
    try:
        ensemble_auc = roc_auc_score(y_test, ensemble_scores)
        model_scores['Ensemble'] = {'auc': ensemble_auc, 'scores': ensemble_scores, 'col': 'ensemble'}
    except:
        pass
    
    # Render AUC cards
    n_models = len(model_scores)
    if n_models > 0:
        cols = st.columns(n_models)
        for idx, (name, info) in enumerate(model_scores.items()):
            with cols[idx]:
                auc_class = get_auc_class(info['auc'])
                emoji = "🥇" if info['auc'] == max(m['auc'] for m in model_scores.values()) else ""
                st.markdown(f'''
                    <div class="model-card">
                        <div class="model-name">{name} {emoji}</div>
                        <div class="model-auc {auc_class}">{info['auc']:.4f}</div>
                        <div style="color: #64748b; font-size: 0.8rem;">AUC-ROC</div>
                    </div>
                ''', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────
    # SECTION 2: ROC Curves (All Models Overlaid)
    # ─────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="section-header">📈 ROC Curves (All Models)</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        
        # Random baseline
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='rgba(255,255,255,0.3)', width=1),
            name='Random (AUC=0.50)', showlegend=True
        ))
        
        for name, info in model_scores.items():
            fpr, tpr, _ = roc_curve(y_test, info['scores'])
            color = MODEL_COLORS.get(name, '#ffffff')
            line_width = 3 if name == 'Ensemble' else 2
            dash = 'solid' if name != 'Ensemble' else 'dot'
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name} (AUC={info['auc']:.4f})",
                line=dict(color=color, width=line_width, dash=dash)
            ))
        
        fig_roc.update_layout(
            **DARK_LAYOUT,
            title="Receiver Operating Characteristic",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0.5)"),
            height=500
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with c2:
        st.markdown('<div class="section-header">📉 Precision-Recall Curves</div>', unsafe_allow_html=True)
        fig_pr = go.Figure()
        
        for name, info in model_scores.items():
            precision, recall, _ = precision_recall_curve(y_test, info['scores'])
            ap = average_precision_score(y_test, info['scores'])
            color = MODEL_COLORS.get(name, '#ffffff')
            line_width = 3 if name == 'Ensemble' else 2
            
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision, mode='lines',
                name=f"{name} (AP={ap:.4f})",
                line=dict(color=color, width=line_width)
            ))
        
        # Baseline
        baseline = y_test.sum() / len(y_test)
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline], mode='lines',
            line=dict(dash='dash', color='rgba(255,255,255,0.3)', width=1),
            name=f'Baseline ({baseline:.2f})', showlegend=True
        ))
        
        fig_pr.update_layout(
            **DARK_LAYOUT,
            title="Precision vs Recall",
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(x=0.01, y=0.05, bgcolor="rgba(0,0,0,0.5)"),
            height=500
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ─────────────────────────────────────────────────
    # SECTION 3: Confusion Matrices
    # ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔢 Confusion Matrices (Threshold = 0.5)</div>', unsafe_allow_html=True)
    
    confusion_models = {k: v for k, v in model_scores.items() if k in ['XGBoost', 'Ensemble']}
    # Also add models with a threshold based on median
    for name in ['LSTM Autoencoder', 'Isolation Forest', 'LOF']:
        if name in model_scores:
            confusion_models[name] = model_scores[name]
    
    cm_cols = st.columns(len(confusion_models))
    
    for idx, (name, info) in enumerate(confusion_models.items()):
        with cm_cols[idx]:
            scores = info['scores']
            
            # Use median-based threshold for anomaly scores, 0.5 for probabilities
            if name in ['XGBoost', 'Ensemble']:
                threshold = 0.5
            else:
                # Find optimal threshold using Youden's J statistic
                fpr, tpr, thresholds = roc_curve(y_test, scores)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else np.median(scores)
            
            y_pred = (scores >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            labels = ['Normal (0)', 'Theft (1)']
            fig_cm = px.imshow(
                cm, x=labels, y=labels,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect='equal'
            )
            fig_cm_layout = DARK_LAYOUT.copy()
            fig_cm_layout.update(dict(
                title=dict(text=f"{name}", font=dict(size=13)),
                xaxis_title="Predicted",
                yaxis_title="Actual",
                coloraxis_showscale=False,
                height=350,
                margin=dict(t=40, l=40, r=10, b=40)
            ))
            fig_cm.update_layout(**fig_cm_layout)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ─────────────────────────────────────────────────
    # SECTION 4: Detailed Metric Table
    # ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Comprehensive Metrics Table</div>', unsafe_allow_html=True)
    
    metrics_data = []
    for name, info in model_scores.items():
        scores = info['scores']
        
        # Optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
        if name in ['XGBoost', 'Ensemble']:
            threshold = 0.5
        else:
            threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else np.median(scores)
        
        y_pred = (scores >= threshold).astype(int)
        
        try:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            ap = average_precision_score(y_test, scores)
            
            metrics_data.append({
                'Model': name,
                'AUC-ROC': f"{info['auc']:.4f}",
                'Avg Precision': f"{ap:.4f}",
                'Accuracy': f"{acc:.4f}",
                'Precision': f"{prec:.4f}",
                'Recall': f"{rec:.4f}",
                'F1-Score': f"{f1:.4f}",
                'Threshold': f"{threshold:.4f}"
            })
        except Exception as e:
            metrics_data.append({
                'Model': name,
                'AUC-ROC': f"{info['auc']:.4f}",
                'Error': str(e)
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────
    # SECTION 5: Score Correlation Heatmap
    # ─────────────────────────────────────────────────
    st.markdown("---")
    c5, c6 = st.columns(2)
    
    with c5:
        st.markdown('<div class="section-header">🔗 Model Score Correlations</div>', unsafe_allow_html=True)
        
        corr_cols = [col for col in ['XGBoost_Probability', 'LSTM_Reconstruction_Error', 'IF_Score', 'LOF_Score'] 
                     if col in test_df.columns]
        
        if len(corr_cols) >= 2:
            corr_df = test_df[corr_cols].copy()
            corr_df.columns = [score_columns.get(c, c) for c in corr_cols]
            
            corr_matrix = corr_df.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.3f',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='equal'
            )
            fig_corr.update_layout(**DARK_LAYOUT, title="Inter-Model Score Correlation", height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with c6:
        st.markdown('<div class="section-header">📊 AUC-ROC Comparison Bar Chart</div>', unsafe_allow_html=True)
        
        auc_data = pd.DataFrame([
            {'Model': name, 'AUC-ROC': info['auc']} 
            for name, info in model_scores.items()
        ]).sort_values('AUC-ROC', ascending=True)
        
        fig_bar = px.bar(
            auc_data, x='AUC-ROC', y='Model', orientation='h',
            color='Model',
            color_discrete_map=MODEL_COLORS
        )
        fig_bar.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)", 
                         annotation_text="Random", annotation_position="top")
        fig_bar.update_layout(
            **DARK_LAYOUT,
            title="AUC-ROC by Model",
            showlegend=False,
            height=400,
            xaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─────────────────────────────────────────────────
    # SECTION 6: Feature Importance (if XGBoost saved it)
    # ─────────────────────────────────────────────────
    importance_path = os.path.join(PROJECT_ROOT, "models_saved", "feature_importance.csv")
    if os.path.exists(importance_path):
        st.markdown("---")
        st.markdown('<div class="section-header">🌟 XGBoost Feature Importance (Top 15)</div>', unsafe_allow_html=True)
        
        imp_df = pd.read_csv(importance_path).nlargest(15, 'Importance')
        
        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(**DARK_LAYOUT, title=None, coloraxis_showscale=False, height=500,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

    # ─────────────────────────────────────────────────
    # SECTION 7: Threshold Sensitivity Analysis
    # ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎚️ Threshold Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "Select model for threshold analysis:",
        list(model_scores.keys()),
        index=list(model_scores.keys()).index('Ensemble') if 'Ensemble' in model_scores else 0
    )
    
    if selected_model in model_scores:
        scores = model_scores[selected_model]['scores']
        
        thresholds_range = np.linspace(scores.min(), scores.max(), 100)
        f1_scores = []
        precisions = []
        recalls = []
        
        for t in thresholds_range:
            y_p = (scores >= t).astype(int)
            f1_scores.append(f1_score(y_test, y_p, zero_division=0))
            precisions.append(precision_score(y_test, y_p, zero_division=0))
            recalls.append(recall_score(y_test, y_p, zero_division=0))
        
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(x=thresholds_range, y=f1_scores, mode='lines', name='F1-Score', 
                                         line=dict(color='#6366f1', width=3)))
        fig_thresh.add_trace(go.Scatter(x=thresholds_range, y=precisions, mode='lines', name='Precision',
                                         line=dict(color='#34d399', width=2)))
        fig_thresh.add_trace(go.Scatter(x=thresholds_range, y=recalls, mode='lines', name='Recall',
                                         line=dict(color='#f472b6', width=2)))
        
        # Mark optimal F1 threshold
        best_idx = np.argmax(f1_scores)
        fig_thresh.add_vline(x=thresholds_range[best_idx], line_dash="dash", line_color="#fbbf24",
                            annotation_text=f"Best F1={f1_scores[best_idx]:.3f} @ {thresholds_range[best_idx]:.3f}")
        
        fig_thresh.update_layout(
            **DARK_LAYOUT,
            title=f"Threshold Analysis — {selected_model}",
            xaxis_title="Threshold",
            yaxis_title="Score",
            height=450,
            legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: #475569;">
            <p>📊 Model Evaluation Dashboard • All metrics computed on held-out test set</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
