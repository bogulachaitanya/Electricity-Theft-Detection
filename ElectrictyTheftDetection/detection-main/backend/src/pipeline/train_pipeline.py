import os
import subprocess
import sys

def run_step(step_name, command):
    print(f"\n{'='*70}")
    print(f"PIPELINE STEP: {step_name}")
    print(f"EXECUTING: {command}")
    print(f"{'='*70}")
    
    try:
        # We use shell=True for Windows compatibility with python paths
        subprocess.run(command, check=True, shell=True)
        print(f"SUCCESS: {step_name} finished.")
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed at: {step_name}")
        print(f"Details: {e}")
        sys.exit(1)

def main():
    print("Starting Electricity Theft Detection Pipeline...")
    
    # 1. Data Loading & Preprocessing
    run_step(
        "Phase 1: Data Ingestion & Cleaning",
        "python src/data/load_data.py --raw \"data/raw/sgcc/data set.csv\" --out data/processed/clean_sgcc.csv"
    )
    
    # 2. Feature Engineering
    run_step(
        "Phase 2: Feature Engineering (20+ Dimensions)",
        "python src/features/feature_engineer.py --input data/processed/clean_sgcc.csv --output data/processed/features_sgcc.csv"
    )
    
    # 3. Model Training (IF, LOF, LSTM, XGBoost)
    run_step(
        "Phase 3-5: Model Training & Cross-Validation",
        "python src/models/train_model.py --data data/processed/features_sgcc.csv --outdir data/processed --model_dir models_saved"
    )
    
    # 4. Evaluation & Risk Scoring
    run_step(
        "Phase 6: Final Risk Assessment & Metrics",
        "python src/evaluation/evaluate_model.py"
    )
    
    # 5. Database Initialization (for Dashboard)
    run_step(
        "Phase 7: Dashboard Database Setup",
        "python src/data/init_db.py"
    )
    
    print("\n" + "#"*70)
    print("PIPELINE EXECUTION COMPLETE!")
    print("Final Risk Scores: data/processed/final_risk_scores.csv")
    print("Models Saved In : models_saved/")
    print("#"*70)

if __name__ == "__main__":
    main()
