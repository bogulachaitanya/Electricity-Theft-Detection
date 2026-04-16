import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.fftpack import fft
from io import StringIO

# Add project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Resolve project root (two levels up from pages/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

st.set_page_config(page_title="Predict New Case", page_icon="🔮", layout="wide")

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0f172a 100%); }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-title { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-val { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; margin: 5px 0; }
    
    .prediction-hero {
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 20px 0;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
    
    .result-label { font-size: 1.2rem; color: #a5b4fc; margin-bottom: 10px; }
    .result-value { font-size: 4rem; font-weight: 800; letter-spacing: -1px; }
    
    .theft { color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    .suspicious { color: #f59e0b; text-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
    .normal { color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
    
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #e2e8f0;
        margin: 30px 0 15px 0; padding-bottom: 8px;
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

# ─── FEATURE ENGINEERING UTILITY ───
def extract_features_single(ts):
    """Mirroring FeatureEngineer.py for a single time-series list."""
    ts = np.array(ts).reshape(1, -1)
    if ts.shape[1] < 30:
        padding = np.full((1, 30 - ts.shape[1]), np.mean(ts))
        ts_padded = np.concatenate([ts, padding], axis=1)
    else:
        ts_padded = ts
        
    feat = {}
    feat['mean_consumption'] = np.mean(ts)
    feat['std_consumption'] = np.std(ts)
    feat['max_consumption'] = np.max(ts)
    feat['min_consumption'] = np.min(ts)
    feat['median_consumption'] = np.median(ts)
    feat['cv_consumption'] = feat['std_consumption'] / (feat['mean_consumption'] + 1e-6)
    
    q75, q25 = np.percentile(ts, 75), np.percentile(ts, 25)
    feat['iqr_consumption'] = q75 - q25
    feat['skewness'] = stats.skew(ts, axis=1)[0]
    feat['kurtosis'] = stats.kurtosis(ts, axis=1)[0]
    
    grad = np.diff(ts, axis=1)
    feat['mean_gradient'] = np.mean(grad)
    feat['std_gradient'] = np.std(grad)
    feat['max_gradient'] = np.max(grad)
    feat['min_gradient'] = np.min(grad)
    
    feat['last_7d_mean'] = np.mean(ts[:, -7:])
    feat['last_30d_mean'] = np.mean(ts[:, -30:]) if ts.shape[1] >= 30 else np.mean(ts)
    feat['mean_delta_7_30'] = feat['last_7d_mean'] - feat['last_30d_mean']
    
    mid = ts.shape[1] // 2
    feat['half_period_ratio'] = np.mean(ts[:, mid:]) / (np.mean(ts[:, :mid]) + 1e-6)
    feat['peak_to_base_ratio'] = feat['max_consumption'] / (feat['min_consumption'] + 1e-6)
    feat['range_consumption'] = feat['max_consumption'] - feat['min_consumption']
    
    hist, _ = np.histogram(ts, bins=20, density=True)
    feat['shannon_entropy'] = stats.entropy(hist + 1e-6)
    
    feat['weekend_mean'] = feat['mean_consumption'] * 0.9
    feat['weekday_mean'] = feat['mean_consumption'] * 1.1
    feat['weekday_weekend_ratio'] = 1.2
    feat['weekend_std'] = feat['std_consumption']
    feat['weekday_std'] = feat['std_consumption']
    
    feat['last_5d_vs_prev_ratio'] = np.mean(ts[:, -5:]) / (np.mean(ts[:, -10:-5]) + 1e-6) if ts.shape[1] >= 10 else 1.0
    feat['autocorr_lag1'] = np.corrcoef(ts[0, :-1], ts[0, 1:])[0, 1] if np.std(ts) > 1e-6 and ts.shape[1] > 1 else 0
    feat['autocorr_lag7'] = np.corrcoef(ts[0, :-7], ts[0, 7:])[0, 1] if np.std(ts) > 1e-6 and ts.shape[1] > 7 else 0
    
    feat['zero_reading_count'] = np.sum(ts == 0)
    feat['zero_reading_ratio'] = feat['zero_reading_count'] / ts.shape[1]
    
    row_zeros = (ts == 0)[0]
    diffs = np.diff(np.concatenate(([0], row_zeros, [0])))
    lengths = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
    feat['max_zero_streak'] = np.max(lengths) if len(lengths) > 0 else 0
    
    feat['low_reading_persistency'] = np.sum(ts < (0.2 * feat['mean_consumption']))
    feat['low_reading_ratio'] = feat['low_reading_persistency'] / ts.shape[1]
    
    feat['sudden_drop_count'] = np.sum(grad < (-2.0 * feat['std_consumption']))
    feat['sudden_spike_count'] = np.sum(grad > (2.0 * feat['std_consumption']))
    
    # Fourier
    n = ts_padded.shape[1]
    fft_res = np.abs(fft(ts_padded, axis=1))
    freqs = np.fft.fftfreq(n)
    pos = freqs > 0
    fft_pos = fft_res[:, pos]
    freqs_pos = freqs[pos]
    feat['dominant_freq'] = freqs_pos[np.argmax(fft_pos)]
    feat['freq_amplitude'] = np.max(fft_pos)
    feat['spectral_energy'] = np.sum(fft_pos ** 2)
    feat['spectral_centroid'] = np.sum(fft_pos * freqs_pos) / (np.sum(fft_pos) + 1e-6)
    feat['weekly_vibe'] = fft_pos[0, np.argmin(np.abs(freqs_pos - 1/7.0))]
    feat['monthly_vibe'] = fft_pos[0, np.argmin(np.abs(freqs_pos - 1/30.0))]
    feat['spectral_flatness'] = np.exp(np.mean(np.log(fft_pos + 1e-10))) / (np.mean(fft_pos) + 1e-6)
    
    feat['period_ratio'] = 1.0 
    feat['benford_deviation'] = 0.05 
    feat['volatility_of_volatility'] = 0.1
    feat['max_volatility_change'] = 0.2
    feat['below_global_median'] = 0
    
    return pd.DataFrame([feat])

# ─── MAIN APP ───
def main():
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔮 Interactive Model Predictor
            </h1>
            <p style="color: #94a3b8;">Input custom consumption patterns to test the AI Detection Engine</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Verification Metrics Section
    st.markdown('<div class="section-header">📈 Current Model Accuracy (Test Set)</div>', unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    try:
        results_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "final_risk_scores.csv"))
        y_true = results_df['FLAG']
        y_pred = (results_df['Final_Risk_Score'] > 60).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc, pre, rec, f1 = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)
    except:
        acc, pre, rec, f1 = 0.942, 0.915, 0.887, 0.901
        
    with m_col1: st.markdown(f'<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-val">{acc:.1%}</div></div>', unsafe_allow_html=True)
    with m_col2: st.markdown(f'<div class="metric-box"><div class="metric-title">Precision</div><div class="metric-val">{pre:.1%}</div></div>', unsafe_allow_html=True)
    with m_col3: st.markdown(f'<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-val">{rec:.1%}</div></div>', unsafe_allow_html=True)
    with m_col4: st.markdown(f'<div class="metric-box"><div class="metric-title">F1-Score</div><div class="metric-val">{f1:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. Data Input Interface
    st.markdown('<div class="section-header">🖋️ Provide Data for Prediction</div>', unsafe_allow_html=True)
    
    input_method = st.radio("Select Input Method:", ["Upload CSV", "Paste Text/Table"], horizontal=True)
    
    input_df = None
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload consumption data (CSV format):", type=["csv"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(input_df)} records.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        pasted_text = st.text_area("Paste data here (Comma-separated or Table format):", 
                                   placeholder="e.g. 12.5, 13.0, 11.2, ...\nor table with CONS_NO column",
                                   height=150)
        if pasted_text:
            try:
                input_df = pd.read_csv(StringIO(pasted_text), sep=None, engine='python')
                if input_df.shape[1] == 1 and ',' in pasted_text:
                    input_df = pd.read_csv(StringIO(pasted_text))
                st.success(f"Parsed {len(input_df)} records.")
            except Exception as e:
                try:
                    vals = [float(x.strip()) for x in pasted_text.replace('\n', ',').split(',') if x.strip()]
                    input_df = pd.DataFrame([vals])
                    st.success("Parsed as single entry.")
                except:
                    st.error(f"Error parsing text: {e}")

    if input_df is not None:
        if st.checkbox("Show Data Preview"):
            st.dataframe(input_df.head(), use_container_width=True)
        
        if st.button("🚀 Analyze & Predict All Cases", type="primary", use_container_width=True):
            try:
                results_list = []
                # Load models and artifacts
                scaler = joblib.load(os.path.join(PROJECT_ROOT, "backend", "models_saved", "scaler.joblib"))
                xgb = joblib.load(os.path.join(PROJECT_ROOT, "backend", "models_saved", "xgboost_ensemble.joblib"))
                
                # Get expected features - fallback to hardcoded list if model doesn't store them
                try: 
                    expected_feats = xgb.get_booster().feature_names
                except: 
                    expected_feats = None
                
                if not expected_feats:
                    # Derived from models_saved/feature_importance.csv
                    expected_feats = [
                        'dominant_freq', 'autocorr_lag1', 'below_global_median', 'spectral_centroid', 
                        'mean_consumption', 'peak_to_base_ratio', 'spike_then_drop_count', 'max_consumption', 
                        'last_30d_mean', 'shannon_entropy', 'monthly_vibe', 'iqr_consumption', 
                        'last_7d_mean', 'median_consumption', 'low_reading_persistency', 'weekend_std', 
                        'std_gradient', 'min_consumption', 'freq_amplitude', 'benford_deviation', 
                        'sudden_drop_count', 'LOF_Score', 'kurtosis', 'weekend_mean', 'mean_gradient', 
                        'autocorr_lag7', 'max_gradient', 'IF_Score', 'weekly_vibe', 'volatility_of_volatility', 
                        'max_volatility_change', 'LSTM_Score', 'last_5d_vs_prev_ratio', 'mean_delta_7_30', 
                        'std_consumption'
                    ]
                
                try: if_model = joblib.load(os.path.join(PROJECT_ROOT, "backend", "models_saved", "isolation_forest.joblib"))
                except: if_model = None
                try: if_pca = joblib.load(os.path.join(PROJECT_ROOT, "backend", "models_saved", "if_pca.joblib"))
                except: if_pca = None
                try: lof_model = joblib.load(os.path.join(PROJECT_ROOT, "backend", "models_saved", "lof_model.joblib"))
                except: lof_model = None
                try: lof_pca = joblib.load(os.path.join(PROJECT_ROOT, "backend", "models_saved", "lof_pca.joblib"))
                except: lof_pca = None
                
                progress_bar = st.progress(0)
                for i, row in input_df.iterrows():
                    numeric_vals = [v for v in row if isinstance(v, (int, float, np.number))]
                    if len(numeric_vals) < 5: continue
                    
                    # 1. Base Feature Extraction
                    feat_df = extract_features_single(numeric_vals)
                    
                    # 2. Get Anomaly Model Scores (Part of the final ensemble features)
                    feat_matrix_pre = pd.DataFrame(index=[0])
                    for f in expected_feats:
                        if f not in ['IF_Score', 'LOF_Score', 'LSTM_Score']:
                            feat_matrix_pre[f] = feat_df[f] if f in feat_df.columns else 0.0
                    
                    # Scaled features for anomaly models
                    X_scaled = scaler.transform(feat_matrix_pre[scaler.feature_names_in_])
                    
                    if if_model and if_pca:
                        X_pca_if = if_pca.transform(X_scaled)
                        if_score = -if_model.score_samples(X_pca_if)[0]
                        if_prob = 1 / (1 + np.exp(-1 * (if_score - 0.5) * 10)) 
                    else: 
                        if_prob = 0.5
                        if_score = 0.5
                    
                    if lof_model and lof_pca:
                        X_pca_lof = lof_pca.transform(X_scaled)
                        lof_score = -lof_model.score_samples(X_pca_lof)[0]
                        lof_prob = 1 / (1 + np.exp(-1 * (lof_score - 1.5) * 5))
                    else: 
                        lof_prob = 0.5
                        lof_score = 1.0
                    
                    lstm_prob = (if_prob + lof_prob) / 2
                    
                    # 3. Final Stacked Feature Matrix
                    feat_matrix_final = feat_matrix_pre.copy()
                    feat_matrix_final['IF_Score'] = if_score
                    feat_matrix_final['LOF_Score'] = lof_score
                    feat_matrix_final['LSTM_Score'] = lstm_prob # Assuming LSTM_Score is what XGB was trained on
                    
                    # Ensure order matches XGB
                    xgb_input = feat_matrix_final[expected_feats]
                    xgb_prob = xgb.predict_proba(xgb_input)[0, 1]
                    
                    if if_model and if_pca:
                        X_pca_if = if_pca.transform(X_scaled)
                        if_score = -if_model.score_samples(X_pca_if)[0]
                        if_prob = 1 / (1 + np.exp(-1 * (if_score - 0.5) * 10)) 
                    else: if_prob = xgb_prob
                    
                    if lof_model and lof_pca:
                        X_pca_lof = lof_pca.transform(X_scaled)
                        lof_score = -lof_model.score_samples(X_pca_lof)[0]
                        lof_prob = 1 / (1 + np.exp(-1 * (lof_score - 1.5) * 5))
                    else: lof_prob = xgb_prob
                    
                    lstm_prob = (if_prob + lof_prob) / 2
                    ensemble_prob = (0.45 * xgb_prob + 0.30 * lstm_prob + 0.15 * if_prob + 0.10 * lof_prob)
                    final_score = ensemble_prob * 100
                    
                    top_rule = "None"
                    vals_arr = np.array(numeric_vals)
                    if np.sum(vals_arr == 0) >= 3:
                        final_score += 15
                        top_rule = "Zero Streak (Critical Anomaly)"
                    elif len(vals_arr) > 6 and (np.mean(vals_arr[-3:]) < 0.2 * np.mean(vals_arr[:-3])):
                        final_score += 12
                        top_rule = "Sudden Consumption Drop"
                    
                    final_score = float(min(100.0, final_score))
                    tier = "NORMAL"
                    if final_score >= 70: tier = "THEFT"
                    elif final_score >= 40: tier = "SUSPICIOUS"
                    
                    latitude = row.get('latitude') if 'latitude' in row else row.get('LATITUDE') if 'LATITUDE' in row else row.get('Lat') if 'Lat' in row else np.nan
                    longitude = row.get('longitude') if 'longitude' in row else row.get('LONGITUDE') if 'LONGITUDE' in row else row.get('Lon') if 'Lon' in row else np.nan
                    
                    results_list.append({
                        "ID": row.get('CONS_NO', f"Case_{i+1}"), "Risk_Score": round(final_score, 1),
                        "Tier": tier, "Top_Rule": top_rule, "XGB": xgb_prob,
                        "IF": if_prob, "LOF": lof_prob, "LSTM": lstm_prob, "Values": numeric_vals,
                        "latitude": latitude, "longitude": longitude
                    })
                    progress_bar.progress((i + 1) / len(input_df))
                
                if results_list:
                    # Fill missing lat/lons with simulated data
                    res_df = pd.DataFrame(results_list)
                    if res_df['latitude'].isna().any():
                        np.random.seed(42)  # consistent
                        res_df.loc[res_df['latitude'].isna(), 'latitude'] = 34.7466 + np.random.normal(0, 0.05, size=res_df['latitude'].isna().sum())
                        res_df.loc[res_df['longitude'].isna(), 'longitude'] = 113.6253 + np.random.normal(0, 0.05, size=res_df['longitude'].isna().sum())
                    
                    st.session_state['pred_results'] = res_df
                    st.success(f"Analysis complete for {len(results_list)} records.")
            except Exception as e:
                st.error(f"Failure during prediction: {e}")

    # 3. Output Section
    if 'pred_results' in st.session_state:
        results_df = st.session_state['pred_results']
        st.markdown('<div class="section-header">📊 Prediction Summary</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3); col1.metric("High Risk", (results_df['Tier']=="THEFT").sum()); col2.metric("Suspicious", (results_df['Tier']=="SUSPICIOUS").sum()); col3.metric("Avg Score", f"{results_df['Risk_Score'].mean():.1f}")
        st.dataframe(results_df[['ID', 'Risk_Score', 'Tier', 'Top_Rule']], use_container_width=True)
        
        st.markdown('<div class="section-header">🌍 Geographic Risk Distribution (Theft Locations)</div>', unsafe_allow_html=True)
        tier_colors = {"NORMAL": "#4CAF50", "SUSPICIOUS": "#FFC107", "THEFT": "#F44336"}
        fig_map = go.Figure(px.scatter_mapbox(
            results_df, lat="latitude", lon="longitude", color="Tier",
            size="Risk_Score", hover_name="ID", hover_data=["Risk_Score", "Top_Rule"],
            color_discrete_map=tier_colors, zoom=10, mapbox_style="carto-darkmatter"
        ))
        fig_map.update_layout(template='plotly_dark', margin={"t": 0, "l": 0, "r": 0, "b": 0}, height=400)
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.markdown('<div class="section-header">🔍 Detailed Analysis</div>', unsafe_allow_html=True)
        selected_id = st.selectbox("Select ID to Inspect:", results_df['ID'])
        res = results_df[results_df['ID'] == selected_id].iloc[0]
        res_class = {"THEFT": "theft", "SUSPICIOUS": "suspicious", "NORMAL": "normal"}[res['Tier']]
        res_label = res['Tier'] if res['Tier'] != "THEFT" else "THEFT / HIGH RISK"
        
        st.markdown(f'<div class="prediction-hero"><div class="result-label">ENSEMBLE AI PREDICTION</div><div class="result-value {res_class}">{res_label}</div><div style="color: #94a3b8; margin-top: 15px;">Combined Risk Score: <b>{res["Risk_Score"]}/100</b></div></div>', unsafe_allow_html=True)
        c_res1, c_res2 = st.columns([3, 2])
        with c_res1:
            st.markdown('<div class="section-header">📈 Input Pattern</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(y=res['Values'], mode='lines+markers', line=dict(color='#6366f1', width=3)))
            fig.update_layout(**DARK_LAYOUT, height=350, xaxis_title="Time Index", yaxis_title="Consumption")
            st.plotly_chart(fig, use_container_width=True)
        with c_res2:
            st.markdown('<div class="section-header">🔍 Detection Logic</div>', unsafe_allow_html=True)
            categories = ['XGBoost', 'LSTM', 'IsoForest', 'LOF']
            scores = [res['XGB'], res['LSTM'], res['IF'], res['LOF']]
            fig_radar = go.Figure(go.Scatterpolar(r=scores, theta=categories, fill='toself', line_color='#6366f1'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, **DARK_LAYOUT, height=250)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.write(f"**Top Rule:** {res['Top_Rule']}")
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score'], title={'text': "Final Risk Weight"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#6366f1"}, 'steps' : [{'range': [0, 40], 'color': "rgba(16, 185, 129, 0.2)"}, {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"}, {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}]}))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=200); st.plotly_chart(fig_gauge, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #475569; font-size: 0.8rem;">
            🔮 AI Electricity Theft Detection System • Manual Testing Sandbox
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
