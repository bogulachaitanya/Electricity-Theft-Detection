import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

def check_model_metrics(data_path="data/processed/test_ensemble_scores.csv"):
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Please run pipeline first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Check if we have labels
    if 'FLAG' not in df.columns:
        print("Error: 'FLAG' column (ground truth) not found in the test dataset.")
        return

    print("="*50)
    print("        ENACTED MODEL EVALUATION METRICS        ")
    print("="*50)
    
    models = {
        'XGBoost': 'XGBoost_Probability',
        'Isolation Forest': 'IF_Score',
        'Local Outlier Factor (LOF)': 'LOF_Score',
        'Deep Dense Autoencoder': 'LSTM_Reconstruction_Error'
    }
    
    y_true = df['FLAG']
    test_size = len(y_true)
    anomalies = y_true.sum()
    print(f"Test Set Size: {test_size} | Theft Examples: {anomalies} | Normal: {test_size - anomalies}\n")
    
    for model_name, col_name in models.items():
        if col_name in df.columns:
            y_scores = df[col_name]
            auc_value = roc_auc_score(y_true, y_scores)
            
            # Formatted Output
            print(f"{model_name:30s} -> AUC-ROC: {auc_value:.4f}")
        else:
            print(f"{model_name:30s} -> [Column Not Found]")
            
    print("\n" + "="*50)

if __name__ == "__main__":
    check_model_metrics()
