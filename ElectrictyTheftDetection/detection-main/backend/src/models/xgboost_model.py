import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

def train_and_score_xgboost(X_train, X_test, y_train, y_test, model_dir="models_saved"):
    """
    IMPROVED XGBoost:
    - More hyperparameter candidates with wider search
    - Proper scale_pos_weight calibration
    - Better early stopping
    - Saves feature importance
    - Stratified 5-Fold CV for robust evaluation
    """
    print("\n--- Training XGBoost Ensemble Model (IMPROVED) ---")
    
    # Calculate scale weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
    print(f"-> Class distribution: Normal={num_neg}, Theft={num_pos}")
    print(f"-> scale_pos_weight: {scale_pos_weight:.2f}")

    # Expanded hyperparameter search
    param_candidates = [
        {
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        },
        {
            'n_estimators': 2000,
            'learning_rate': 0.005,
            'max_depth': 8,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
            'reg_alpha': 0.05,
            'reg_lambda': 1.5,
        },
        {
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'max_depth': 5,
            'min_child_weight': 7,
            'subsample': 0.75,
            'colsample_bytree': 0.6,
            'gamma': 0.3,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
        },
        {
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'max_depth': 7,
            'min_child_weight': 4,
            'subsample': 0.9,
            'colsample_bytree': 0.75,
            'gamma': 0.05,
            'reg_alpha': 0.01,
            'reg_lambda': 0.8,
        },
        {
            'n_estimators': 2000,
            'learning_rate': 0.008,
            'max_depth': 9,
            'min_child_weight': 6,
            'subsample': 0.7,
            'colsample_bytree': 0.65,
            'gamma': 0.15,
            'reg_alpha': 0.2,
            'reg_lambda': 1.2,
        },
        {
            'n_estimators': 2000,
            'learning_rate': 0.015,
            'max_depth': 4,
            'min_child_weight': 10,
            'subsample': 0.85,
            'colsample_bytree': 0.9,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
        },
    ]

    # Cross-validation
    print("-> Running Stratified 5-Fold Cross-Validation...")
    best_cv_auc = 0
    best_params = None
    
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    for idx, params in enumerate(param_candidates):
        cv_aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train)):
            X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='auc',
                n_jobs=-1,
                random_state=42,
                tree_method='hist',
                early_stopping_rounds=100,
            )
            
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            y_val_probs = model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_val_probs)
            cv_aucs.append(fold_auc)
        
        mean_auc = np.mean(cv_aucs)
        std_auc = np.std(cv_aucs)
        print(f"   Config {idx+1}/{len(param_candidates)}: CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")
        
        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_params = params
    
    print(f"\n-> Best CV AUC: {best_cv_auc:.4f}")
    print(f"-> Best Params: lr={best_params['learning_rate']}, depth={best_params['max_depth']}, "
          f"subsample={best_params['subsample']}, colsample={best_params['colsample_bytree']}")
    
    # Train final model
    print("\n-> Training FINAL model on full training data...")
    
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=100,
    )
    
    final_model.fit(
        X_train_np, y_train,
        eval_set=[(X_test_np, y_test)],
        verbose=False
    )
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(model_dir, 'xgboost_ensemble.joblib'))

    # Evaluate
    y_test_probs = final_model.predict_proba(X_test_np)[:, 1]
    y_train_probs = final_model.predict_proba(X_train_np)[:, 1]
    y_test_pred = final_model.predict(X_test_np)
    
    auc_roc = roc_auc_score(y_test, y_test_probs)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"-> XGBoost Final AUC-ROC: {auc_roc:.4f}")
    print(f"-> XGBoost F1-Score: {f1:.4f}")
    
    # Feature importance
    if hasattr(X_train, 'columns'):
        importances = final_model.feature_importances_
        feat_names = X_train.columns.tolist()
        importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print("\n-> Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['Feature']:30s} -> {row['Importance']:.4f}")
        
        importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    return y_train_probs, y_test_probs
