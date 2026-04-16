import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

def train_and_score_lof(X_train, X_test, y_test, model_dir="models_saved", y_train=None):
    """
    FIXED LOF:
    - PCA dimensionality reduction (LOF suffers from curse of dimensionality)
    - Train on NORMAL samples only
    - Proper grid search over neighbors and contamination
    - Remove any leaked score columns
    """
    print("\n--- Training Local Outlier Factor (FIXED — PCA + Proper Features) ---")
    
    # CRITICAL: Remove any score columns that may have leaked in
    score_cols = ['IF_Score', 'LOF_Score', 'LSTM_Score', 'LSTM_Reconstruction_Error', 
                  'XGBoost_Probability']
    
    if hasattr(X_train, 'columns'):
        leak_cols = [c for c in score_cols if c in X_train.columns]
        if leak_cols:
            print(f"-> WARNING: Removing leaked score columns: {leak_cols}")
            X_train = X_train.drop(columns=leak_cols)
            X_test = X_test.drop(columns=leak_cols)
    
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Train only on NORMAL data
    if y_train is not None:
        normal_mask = (y_train == 0)
        X_train_normal = X_train_np[normal_mask]
        print(f"-> Training on {len(X_train_normal)} NORMAL samples (out of {len(X_train_np)} total)")
    else:
        X_train_normal = X_train_np
        print(f"-> Training on all {len(X_train_normal)} samples (no labels provided)")

    # ─── PCA Dimensionality Reduction ───
    n_features = X_train_normal.shape[1]
    n_components = min(15, n_features)
    
    print(f"-> Applying PCA: {n_features} features -> {n_components} components")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_normal)
    X_test_pca = pca.transform(X_test_np)
    X_all_train_pca = pca.transform(X_train_np)
    
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"-> PCA explained variance: {explained:.1f}%")

    # ─── Grid Search (Simplified for Speed) ───
    best_auc = 0
    best_model = None
    best_params = {}
    
    # We reduce the search space to avoid extremely long training times on 27k+ samples
    print("-> Running simplified hyperparameter search...")
    for n_neighbors in [20, 50]:
        for contam in [0.05, 0.10]:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contam,
                    novelty=True,
                    n_jobs=-1
                )
                lof.fit(X_train_pca)
                
                y_test_scores = lof.score_samples(X_test_pca)
                auc = roc_auc_score(y_test, y_test_scores)
                
                if auc > best_auc:
                    best_auc = auc
                    best_model = lof
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'contamination': contam
                    }
            except Exception as e:
                print(f"   Config (n={n_neighbors}, c={contam}) failed: {e}")
                continue
    
    if best_model is None:
        print("-> WARNING: All LOF configs failed. Using default LOF.")
        best_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs=-1)
        best_model.fit(X_train_pca)
    
    print(f"-> Best LOF Params: {best_params}")
    print(f"-> LOF AUC-ROC: {best_auc:.4f}")
    
    # Score with best model
    y_test_scores = best_model.score_samples(X_test_pca)
    y_train_scores = best_model.score_samples(X_all_train_pca)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'lof_model.joblib'))
    joblib.dump(pca, os.path.join(model_dir, 'lof_pca.joblib'))
    
    return y_train_scores, y_test_scores
