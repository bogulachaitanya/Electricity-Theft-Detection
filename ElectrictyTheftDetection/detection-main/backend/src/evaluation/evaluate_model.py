import os
import argparse
import pandas as pd
import numpy as np

class EvaluateModelTracker:
    def __init__(self, data_file, output_file):
        self.data_file = data_file
        self.output_file = output_file
        
    def load_model_scoring(self):
        print("Loading test dataset with all four model scores...")
        self.results_df = pd.read_csv(self.data_file)
        print(f"-> Shape: {self.results_df.shape}")
        print(f"-> Columns: {self.results_df.columns.tolist()}")
        
    def normalize_scores(self):
        print("Normalizing all model scores to 0-1 range...")
        for col in ['IF_Score', 'LOF_Score', 'LSTM_Reconstruction_Error']:
            if col in self.results_df.columns:
                col_min = self.results_df[col].min()
                col_max = self.results_df[col].max()
                norm_col = col.replace('_Score', '_Norm').replace('_Reconstruction_Error', '_Norm')
                self.results_df[norm_col] = (self.results_df[col] - col_min) / (col_max - col_min + 1e-8)
                print(f"   {col}: [{col_min:.6f}, {col_max:.6f}] -> {norm_col}")

    def apply_weights(self):
        """
        Updated weights per Implementation Plan Section 6:
        - 45% XGBoost (supervised, ground truth labels)
        - 30% LSTM (temporal patterns)
        - 15% IF (global anomaly)
        - 10% LOF (local anomaly)
        """
        print("Applying weighted fusion (45% XGB, 30% LSTM, 15% IF, 10% LOF)...")
        
        xgb_prob = self.results_df.get('XGBoost_Probability', pd.Series(np.zeros(len(self.results_df))))
        lstm_norm = self.results_df.get('LSTM_Norm', pd.Series(np.zeros(len(self.results_df))))
        if_norm = self.results_df.get('IF_Norm', pd.Series(np.zeros(len(self.results_df))))
        lof_norm = self.results_df.get('LOF_Norm', pd.Series(np.zeros(len(self.results_df))))
        
        base_score = (xgb_prob * 0.45) + (lstm_norm * 0.30) + (if_norm * 0.15) + (lof_norm * 0.10)
        self.results_df['Base_Score_100'] = base_score * 100
        
        print(f"-> Base score stats: mean={base_score.mean()*100:.1f}, "
              f"std={base_score.std()*100:.1f}, "
              f"min={base_score.min()*100:.1f}, max={base_score.max()*100:.1f}")
        
    def apply_domain_rules(self, original_features_file="data/processed/features_sgcc.csv"):
        print("Applying domain risk rules...")
        
        try:
            feats = pd.read_csv(original_features_file)
            needed_cols = ['CONS_NO', 'max_zero_streak', 'last_5d_vs_prev_ratio', 
                          'mean_delta_7_30', 'spike_then_drop_count', 'low_reading_persistency']
            available = [c for c in needed_cols if c in feats.columns]
            self.results_df = self.results_df.merge(feats[available], on='CONS_NO', how='left')
        except Exception as e:
            print(f"Warning: Could not load features for domain rules: {e}")
            self.results_df['Rule_Additions'] = 0
            self.results_df['Top_Flagged_Rule'] = "None"
            return

        self.results_df['Rule_Additions'] = 0
        self.results_df['Top_Flagged_Rule'] = "None"

        # Rule 1: Zero Streak (Critical)
        if 'max_zero_streak' in self.results_df.columns:
            zero_mask = self.results_df['max_zero_streak'] >= 7
            self.results_df.loc[zero_mask, 'Rule_Additions'] += 20
            self.results_df.loc[zero_mask, 'Top_Flagged_Rule'] = "Extended Zero Reading Streak"

        # Rule 2: End-of-month drop
        if 'last_5d_vs_prev_ratio' in self.results_df.columns:
            eom_mask = self.results_df['last_5d_vs_prev_ratio'] < 0.4
            self.results_df.loc[eom_mask, 'Rule_Additions'] += 10
            self.results_df.loc[eom_mask & (self.results_df['Top_Flagged_Rule'] == "None"), 
                               'Top_Flagged_Rule'] = "Suspicious End-of-Month Drop"

        # Rule 3: Sustained Drop
        if 'mean_delta_7_30' in self.results_df.columns and 'low_reading_persistency' in self.results_df.columns:
            sustained_mask = (self.results_df['mean_delta_7_30'] < -1.0) & \
                            (self.results_df['low_reading_persistency'] > 10)
            self.results_df.loc[sustained_mask, 'Rule_Additions'] += 15
            self.results_df.loc[sustained_mask & (self.results_df['Top_Flagged_Rule'] == "None"),
                               'Top_Flagged_Rule'] = "Sustained Baseline Reduction"

        # Rule 4: Zigzag
        if 'spike_then_drop_count' in self.results_df.columns:
            zigzag_mask = self.results_df['spike_then_drop_count'] >= 2
            self.results_df.loc[zigzag_mask, 'Rule_Additions'] += 10
            self.results_df.loc[zigzag_mask & (self.results_df['Top_Flagged_Rule'] == "None"),
                               'Top_Flagged_Rule'] = "Bypass Zigzag Pattern Detected"

        # Rule 5: Low Value Persistency
        if 'low_reading_persistency' in self.results_df.columns:
            persist_mask = self.results_df['low_reading_persistency'] > 25
            self.results_df.loc[persist_mask, 'Rule_Additions'] += 15
            self.results_df.loc[persist_mask & (self.results_df['Top_Flagged_Rule'] == "None"),
                               'Top_Flagged_Rule'] = "Persistent Low-Consumption Profile"
            
        final_score = self.results_df['Base_Score_100'] + self.results_df['Rule_Additions']
        self.results_df['Final_Risk_Score'] = np.clip(final_score, 0, 100)
        
        print(f"-> Final Score stats: mean={self.results_df['Final_Risk_Score'].mean():.1f}, "
              f"max={self.results_df['Final_Risk_Score'].max():.1f}")
        
    def classify_tier(self):
        print("Classifying into risk tiers...")
        def assign_tier(score):
            if score <= 30: return "Normal"
            elif score <= 60: return "Suspicious"
            elif score <= 80: return "High Risk"
            else: return "Theft"
            
        self.results_df['Risk_Tier'] = self.results_df['Final_Risk_Score'].apply(assign_tier)
        
        tier_counts = self.results_df['Risk_Tier'].value_counts()
        print(f"-> Tier distribution:\n{tier_counts}")
        
    def generate_final_output(self):
        print(f"\n-> Writing Final Risk Scores to {self.output_file}...")
        
        cols = ['CONS_NO', 'FLAG', 'Final_Risk_Score', 'Risk_Tier', 'Top_Flagged_Rule']
        if 'XGBoost_Probability' in self.results_df.columns: cols.append('XGBoost_Probability')
        if 'LSTM_Reconstruction_Error' in self.results_df.columns: cols.append('LSTM_Reconstruction_Error')

        final_export = self.results_df[cols]
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        final_export.to_csv(self.output_file, index=False)
        print("Pipeline Evaluated and Generated CSV output!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/test_ensemble_scores.csv")
    parser.add_argument("--output", default="data/processed/final_risk_scores.csv")
    args = parser.parse_args()
    
    engine = EvaluateModelTracker(args.data, args.output)
    engine.load_model_scoring()
    engine.normalize_scores()
    engine.apply_weights()
    engine.apply_domain_rules()
    engine.classify_tier()
    engine.generate_final_output()
