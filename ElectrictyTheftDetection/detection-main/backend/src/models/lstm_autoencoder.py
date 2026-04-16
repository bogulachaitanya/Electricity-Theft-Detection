import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

def train_and_score_lstm(X_train, X_test, y_train, y_test, model_dir="models_saved"):
    """
    FIXED Deep Dense Autoencoder:
    - Uses the scaled 1034-dimensional engineered features.
    - Trains strictly on NORMAL samples to learn the baseline manifold.
    - Uses reconstruction error as the anomaly score.
    NOTE: Function name kept as `train_and_score_lstm` for compatibility.
    """
    print("\n--- Training Deep Dense Autoencoder (Formerly LSTM) ---")
    
    # ─── STEP 1: Train only on NORMAL points ───
    normal_mask = (y_train == 0)
    X_train_normal = X_train[normal_mask]
    print(f"-> Training Autoencoder on {len(X_train_normal)} NORMAL profiles "
          f"(out of {len(X_train)} total train profiles)")
          
    n_features = X_train.shape[1]
    
    # ─── STEP 2: Build Deep Dense Autoencoder ───
    model = Sequential([
        Input(shape=(n_features,)),
        
        # Encoder
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        BatchNormalization(),
        
        # Bottleneck
        Dense(64, activation='relu', name='bottleneck'),
        
        # Decoder
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(512, activation='relu'),
        BatchNormalization(),
        
        # Output layer
        Dense(n_features, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    
    # ─── STEP 3: Train Model ───
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, min_delta=1e-5
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    
    print("-> Fitting model...")
    history = model.fit(
        X_train_normal, X_train_normal,
        epochs=100,
        batch_size=128,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'lstm_autoencoder.keras'))
    
    # ─── STEP 4: Compute Reconstruction Errors ───
    print("-> Computing reconstruction errors...")
    
    train_preds = model.predict(X_train, batch_size=256, verbose=0)
    test_preds = model.predict(X_test, batch_size=256, verbose=0)
    
    # MSE for each row vector
    train_errors = np.mean(np.square(X_train - train_preds), axis=1)
    test_errors = np.mean(np.square(X_test - test_preds), axis=1)
    
    auc = roc_auc_score(y_test, test_errors)
    print(f"-> Deep Dense Autoencoder AUC-ROC: {auc:.4f}")
    
    return train_errors, test_errors
