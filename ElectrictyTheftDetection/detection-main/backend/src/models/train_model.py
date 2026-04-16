import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from isolation_forest import train_and_score_if
from lof import train_and_score_lof
from xgboost_model import train_and_score_xgboost

class ModelTrainerTracker:
    def __init__(self, data_file, output_dir, model_dir="models"):
        self.data_file = data_file
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.scaler = StandardScaler()

    def prepare_data(self):
        print("\n" + "="*60)
        print("  [1/6] Loading Data & Feature Selection")
        print("="*60)
        
        print(f"-> Reading Feature Matrix from: {self.data_file}")
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Missing {self.data_file}")
            
        df = pd.read_csv(self.data_file)
        print(f"-> Original Shape: {df.shape}")
        
        df.dropna(subset=['FLAG'], inplace=True)
        df.fillna(0, inplace=True)

        meta_cols = ['CONS_NO', 'FLAG']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        print(f"-> Input Features: {len(feature_cols)}")
        
        # ─── Multicollinearity Removal ───
        print("\n  [2/6] Removing Highly Correlated Features (> 0.95)")
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        if to_drop:
            print(f"-> Dropping {len(to_drop)} redundant features: {to_drop}")
            df.drop(columns=to_drop, inplace=True)
            feature_cols = [c for c in df.columns if c not in meta_cols]

        # ─── Mutual Information Feature Selection ───
        print("\n  [2.5/6] Mutual Information Feature Selection")
        mi_scores = mutual_info_classif(df[feature_cols], df['FLAG'], random_state=42)
        mi_df = pd.DataFrame({'Feature': feature_cols, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
        
        print("-> Feature MI Scores:")
        for _, row in mi_df.iterrows():
            marker = "✓" if row['MI_Score'] > 0.001 else "✗ (LOW)"
            print(f"   {marker} {row['Feature']:30s} -> MI: {row['MI_Score']:.6f}")
        
        # Drop near-zero MI features
        low_mi = mi_df[mi_df['MI_Score'] < 0.001]['Feature'].tolist()
        if low_mi:
            print(f"-> Dropping {len(low_mi)} low-info features: {low_mi}")
            df.drop(columns=low_mi, inplace=True)
            feature_cols = [c for c in df.columns if c not in meta_cols]
            print(f"-> Final Feature Count: {len(feature_cols)}")

        X, y, cons_nos = df[feature_cols], df['FLAG'], df['CONS_NO']
        
        # ─── Split ───
        print("\n  [3/6] Data Splitting & SMOTE")
        X_train_raw, X_test, y_train_raw, y_test, cons_train, cons_test = train_test_split(
            X, y, cons_nos, test_size=0.30, stratify=y, random_state=42
        )
        print(f"-> Train: {X_train_raw.shape}, Test: {X_test.shape}")
        
        # Save original (pre-SMOTE) for unsupervised models
        self.X_train_original = X_train_raw.copy()
        self.y_train_original = y_train_raw.copy()
        
        # SMOTE — use 0.3 ratio (less aggressive, reduces noise)
        print("-> Applying SMOTE (ratio=0.3)...")
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_raw, y_train_raw)
        print(f"-> After SMOTE: {X_train_resampled.shape}")
        print(f"-> Label Counts:\n{pd.Series(y_train_resampled).value_counts()}")
        
        # ─── Scale ───
        print("-> Standard Scaling...")
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_resampled), columns=feature_cols
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=feature_cols
        )
        
        # Scale original (pre-SMOTE) for unsupervised models
        self.X_train_original_scaled = pd.DataFrame(
            self.scaler.transform(self.X_train_original), columns=feature_cols
        )
        
        # Raw copies for LSTM
        self.X_train_raw = pd.DataFrame(X_train_resampled, columns=feature_cols).values
        self.X_test_raw = pd.DataFrame(X_test, columns=feature_cols).values
        
        self.y_train = y_train_resampled.values
        self.y_test = y_test.values
        self.y_train_original_values = self.y_train_original.values
        self.feature_cols = feature_cols
        
        # Output tracking frames
        self.train_out_df = pd.DataFrame({
            'CONS_NO': [f"TRAIN_{i}" for i in range(len(self.y_train))], 
            'FLAG': self.y_train
        })
        self.test_out_df = pd.DataFrame({
            'CONS_NO': cons_test.values, 
            'FLAG': self.y_test
        })

    def run_pipeline(self):
        print("\n" + "="*60)
        print("  [4/6] Training All ML Models")
        print("="*60)

        # ─── 1. Isolation Forest ───
        # CRITICAL: Pass CLEAN feature columns only (no score columns)
        print("\n==> (1/4) Isolation Forest...")
        if_train, if_test = train_and_score_if(
            self.X_train_original_scaled.copy(),  # Copy to prevent mutation
            self.X_test_scaled.copy(), 
            self.y_test, 
            self.model_dir,
            y_train=self.y_train_original_values
        )
        
        # For SMOTE-augmented train set, re-score using IF on scaled features
        # But we need the PCA transform saved by IF
        if_pca_path = os.path.join(self.model_dir, 'if_pca.joblib')
        if os.path.exists(if_pca_path):
            if_model = joblib.load(os.path.join(self.model_dir, 'isolation_forest.joblib'))
            if_pca = joblib.load(if_pca_path)
            X_train_pca = if_pca.transform(self.X_train_scaled.values)
            if_train_full = if_model.score_samples(X_train_pca)
        else:
            if_model = joblib.load(os.path.join(self.model_dir, 'isolation_forest.joblib'))
            if_train_full = if_model.score_samples(self.X_train_scaled.values)
        
        self.train_out_df['IF_Score'] = if_train_full
        self.test_out_df['IF_Score'] = if_test

        # ─── 2. LOF ───
        print("\n==> (2/4) Local Outlier Factor...")
        lof_train, lof_test = train_and_score_lof(
            self.X_train_original_scaled.copy(),  # Copy to prevent mutation
            self.X_test_scaled.copy(), 
            self.y_test, 
            self.model_dir,
            y_train=self.y_train_original_values
        )
        
        # Re-score SMOTE train set
        lof_pca_path = os.path.join(self.model_dir, 'lof_pca.joblib')
        if os.path.exists(lof_pca_path):
            lof_model = joblib.load(os.path.join(self.model_dir, 'lof_model.joblib'))
            lof_pca = joblib.load(lof_pca_path)
            X_train_lof_pca = lof_pca.transform(self.X_train_scaled.values)
            lof_train_full = lof_model.score_samples(X_train_lof_pca)
        else:
            lof_model = joblib.load(os.path.join(self.model_dir, 'lof_model.joblib'))
            lof_train_full = lof_model.score_samples(self.X_train_scaled.values)
        
        self.train_out_df['LOF_Score'] = lof_train_full
        self.test_out_df['LOF_Score'] = lof_test

        # ─── 3. LSTM Autoencoder ───
        print("\n==> (3/4) LSTM Autoencoder (Raw Time-Series)...")
        try:
            from lstm_autoencoder import train_and_score_lstm
            lstm_train, lstm_test = train_and_score_lstm(
                self.X_train_scaled.values, self.X_test_scaled.values, 
                self.y_train, self.y_test, 
                self.model_dir
            )
        except (ImportError, Exception) as e:
            print(f"WARNING: LSTM Autoencoder failed or skipped ({e}). Using zero-scores fallback.")
            lstm_train = np.zeros(len(self.y_train))
            lstm_test = np.zeros(len(self.y_test))
            
        self.train_out_df['LSTM_Reconstruction_Error'] = lstm_train
        self.test_out_df['LSTM_Reconstruction_Error'] = lstm_test

        # ─── 4. XGBoost ───
        # Now add all model scores as meta-features for XGBoost
        print("\n==> (4/4) XGBoost Ensemble (with meta-features)...")
        
        X_train_xgb = self.X_train_scaled.copy()
        X_test_xgb = self.X_test_scaled.copy()
        
        X_train_xgb['IF_Score'] = if_train_full
        X_train_xgb['LOF_Score'] = lof_train_full
        X_train_xgb['LSTM_Score'] = lstm_train
        
        X_test_xgb['IF_Score'] = if_test
        X_test_xgb['LOF_Score'] = lof_test
        X_test_xgb['LSTM_Score'] = lstm_test
        
        print(f"-> XGBoost input shape: Train={X_train_xgb.shape}, Test={X_test_xgb.shape}")
        
        xgb_train_probs, xgb_test_probs = train_and_score_xgboost(
            X_train_xgb, X_test_xgb, 
            self.y_train, self.y_test, 
            self.model_dir
        )
        
        self.train_out_df['XGBoost_Probability'] = xgb_train_probs
        self.test_out_df['XGBoost_Probability'] = xgb_test_probs

    def export_results(self):
        print("\n" + "="*60)
        print("  [5/6] Exporting Results")
        print("="*60)
        os.makedirs(self.output_dir, exist_ok=True)
        train_path = os.path.join(self.output_dir, "train_ensemble_scores.csv")
        test_path = os.path.join(self.output_dir, "test_ensemble_scores.csv")
        
        self.train_out_df.to_csv(train_path, index=False)
        self.test_out_df.to_csv(test_path, index=False)
        
        # Save the scaler for real-time scoring use
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        
        print(f"-> Train scores: {train_path}")
        print(f"-> Test scores: {test_path}")
        print("Pipeline complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/features_sgcc.csv")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--model_dir", default="models_saved")
    args = parser.parse_args()
    
    tracker = ModelTrainerTracker(args.data, args.outdir, args.model_dir)
    tracker.prepare_data()
    tracker.run_pipeline()
    tracker.export_results()
