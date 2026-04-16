# Electricity Theft Detection in Smart Meter Data
## Full Implementation Plan & Architecture

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset Strategy](#3-dataset-strategy)
4. [Feature Engineering Blueprint](#4-feature-engineering-blueprint)
5. [Model Architecture](#5-model-architecture)
6. [Risk Score Engine](#6-risk-score-engine)
7. [Dashboard Pages Plan](#7-dashboard-pages-plan)
8. [Phase-by-Phase Implementation Plan](#8-phase-by-phase-implementation-plan)
9. [File & Folder Structure](#9-file--folder-structure)
10. [Non-Functional Requirements Strategy](#10-non-functional-requirements-strategy)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Real-World Differentiators](#12-real-world-differentiators)
13. [Deployment Plan](#13-deployment-plan)
14. [Glossary](#14-glossary)

---

## 1. Project Overview

### Problem Statement

Power Distribution Companies (DISCOMs) in India — including TSNPDCL and APEPDCL — suffer thousands of crores in annual revenue losses due to electricity theft. Theft manifests as meter tampering, unauthorized direct tapping, meter bypass, and billing manipulation. Current detection depends entirely on reactive physical inspections — slow, expensive, and covering only a fraction of consumers.

### Proposed Solution

A proactive, ML-powered theft detection system that ingests smart meter time-series data, detects anomalous consumption patterns using multiple algorithms, assigns risk scores per household, and surfaces high-priority inspection queues through an interactive Streamlit dashboard.

### Target Users

- DISCOM vigilance engineers (investigation squad)
- DT (Distribution Transformer) level supervisors
- Revenue protection officers
- Data analytics teams at TSNPDCL / APEPDCL

### Core Objectives

- Detect abnormal electricity consumption patterns from hourly and daily smart meter data
- Classify consumption as Normal, Suspicious, or Theft
- Rank high-risk households for priority physical inspection
- Visualize consumption anomalies per household on a monitoring dashboard
- Auto-retrain the system when new monthly data is ingested

---

## 2. System Architecture

### High-Level Architecture Diagram (Text Representation)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                              │
│  SGCC Dataset  │  UCI Load Diagrams  │  Irish CER  │  Synthetic Gen │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                             │
│  CSV Parser  │  Schema Validator  │  Missing Value Logger  │ SQLite  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                             │
│  Outlier Clipping  │  Z-Score Normalization  │  Gap Imputation       │
│  Resampling to Daily  │  Household Alignment  │  Label Encoding      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING MODULE                          │
│  Rolling Mean/Std  │  Peak-to-Base Ratio  │  Entropy Score          │
│  Weekly Deviation  │  Monthly Billing Gap  │  Zero-Day Count        │
│  Gradient Features  │  Weekday/Weekend Ratio  │  Fourier Features   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌──────────────────────┐   ┌──────────────────────────────────────────┐
│  UNSUPERVISED PATH   │   │          SUPERVISED PATH                 │
│                      │   │                                          │
│  Isolation Forest    │   │  LSTM Autoencoder                        │
│  Local Outlier Factor│   │  (Reconstruction Error Score)            │
│  (Anomaly Scores)    │   │                                          │
│                      │   │  XGBoost Classifier                      │
│                      │   │  (Probability of Theft)                  │
└──────────┬───────────┘   └─────────────────┬────────────────────────┘
           │                                 │
           └──────────────┬──────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  ENSEMBLE RISK SCORE ENGINE                          │
│  Weighted Fusion  │  Domain Rules Layer  │  Threshold Calibration   │
│  0–100 Risk Score  │  Classification: Normal / Suspicious / Theft   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
               ┌───────────┴────────────┐
               ▼                        ▼
┌──────────────────────┐   ┌────────────────────────────────────────────┐
│  STREAMLIT DASHBOARD │   │         BACKEND SERVICES                   │
│  5 Pages (see §7)    │   │  MLflow Tracking  │  SQLite Results DB     │
│  Plotly Charts       │   │  Auto-Retrain Scheduler  │  PDF/CSV Export │
│  Folium Map          │   │  Model Versioning  │  Drift Detection      │
└──────────────────────┘   └────────────────────────────────────────────┘
```

### Component Responsibilities

**Data Ingestion Layer** — Reads raw CSV files, validates schema (column names, data types, date formats), logs missing meter IDs, and writes cleaned raw data to a local SQLite database.

**Preprocessing Pipeline** — Handles all data quality issues. Clips extreme outlier readings (e.g. negative values or readings 10x above household max), normalizes consumption values per household using Z-score, imputes gaps using forward-fill with a 3-day cap, and resamples irregular data to a consistent daily frequency.

**Feature Engineering Module** — Transforms cleaned time-series into a flat feature vector per household. This is the most critical module for model accuracy. Full details in Section 4.

**Unsupervised Path** — Runs without labels. Isolation Forest and LOF each produce an anomaly score (–1 or +1, plus a continuous score). These catch previously unseen theft patterns.

**Supervised Path** — Requires labeled data from SGCC. LSTM Autoencoder is trained on normal sequences only; high reconstruction error at inference = anomalous. XGBoost is trained on labeled feature vectors from SGCC.

**Ensemble Risk Score Engine** — Combines all model outputs into a single 0–100 risk score. Applies domain rule bonuses on top. Classifies into three tiers. Full details in Section 6.

**Streamlit Dashboard** — 5-page interactive application for DISCOM operators. Full details in Section 7.

**Backend Services** — MLflow tracks every experiment. SQLite stores results and history. Auto-retrain pipeline triggers on new CSV upload. Drift detector monitors score distributions monthly.

---

## 3. Dataset Strategy

### Primary Dataset — SGCC (State Grid Corporation of China)

**Source:** Kaggle — "State Grid Corporation of China Electricity Theft Detection Dataset"
**Size:** ~42,372 households, 1,009 days of daily readings
**Labels:** Binary — 0 = Normal, 1 = Theft
**Why it's the best:** Only publicly available large-scale labeled electricity theft dataset. The labeled nature allows supervised model training, making it the foundation of the entire system.

**Usage in this project:**
- 70% training split for XGBoost and LSTM Autoencoder
- 15% validation for threshold tuning
- 15% held-out test for final evaluation metrics
- The "normal" class alone trains the autoencoder

### Secondary Dataset — UCI Electricity Load Diagrams (2011–2014)

**Source:** UCI ML Repository
**Size:** 370 substations, 4-year hourly readings
**Labels:** None (all normal behavior)
**Why use it:** Provides rich normal consumption patterns for augmenting the normal class in SGCC. Helps the model generalize beyond Chinese consumption patterns.

**Usage in this project:**
- Augment normal class training data
- Validate feature engineering logic on diverse behavior
- Test seasonality handling

### Tertiary Dataset — Irish CER Smart Meter Dataset

**Source:** Irish Social Science Data Archive (ISSDA) — free academic access
**Size:** ~6,435 residential and SME consumers, 30-minute intervals over 1.5 years
**Labels:** None
**Why use it:** European consumption behavior adds geographical diversity and prevents overfitting.

**Usage in this project:**
- Unsupervised model validation
- Dashboard demo data for showcasing the system on fresh data

### Synthetic Theft Augmentation

Since SGCC's theft class is a small proportion (~4.5%), the dataset is severely imbalanced. Synthetic theft patterns address this by creating additional labeled theft examples.

**Theft Pattern Types to Synthesize:**

| Pattern Name | Simulation Method | Real-World Equivalent |
|---|---|---|
| Meter Tamper | Multiply actual by 0.3–0.6 for random 60-day windows | Physical meter manipulation |
| Sudden Drop | Set consecutive 30-day blocks to near-zero | Direct bypass installation |
| Billing Fraud | Set last 5 days of each month to 10% of rolling avg | Reading manipulation before billing |
| Zigzag Bypass | Alternate high/low daily values ±40% for 90 days | Bypass removed before meter reading |
| Gradual Reduction | Linearly decrease consumption 50% over 6 months | Slow tamper installation |

**Augmentation Ratio Target:** Bring theft class to 20–25% of total training samples using SMOTE and synthetic generation.

---

## 4. Feature Engineering Blueprint

Feature engineering is the single most impactful step. Models are only as good as the signals they receive.

### Time-Domain Statistical Features

| Feature Name | Definition | Why It Matters |
|---|---|---|
| Rolling Mean 7d | 7-day moving average of daily consumption | Captures short-term baseline |
| Rolling Mean 30d | 30-day moving average | Monthly behavioral baseline |
| Rolling Std 7d | 7-day rolling standard deviation | Detects variance anomalies |
| Rolling Std 30d | 30-day rolling standard deviation | Long-term volatility signal |
| Mean Delta | Difference between 7d mean and 30d mean | Flags sudden sustained drops |
| Z-Score Per Household | (value – household_mean) / household_std | Relative anomaly vs self |
| Daily Gradient | Day-over-day consumption change rate | Detects abrupt jumps/drops |

### Consumption Pattern Features

| Feature Name | Definition | Why It Matters |
|---|---|---|
| Peak-to-Base Ratio | Max consumption / min consumption in 30-day window | Tamper flattens this ratio unnaturally |
| Weekday-Weekend Ratio | Mean weekday usage / mean weekend usage | Bypass often disrupts this natural ratio |
| Peak Hour Ratio | Top 10% daily hours / bottom 10% hours | Usage profile shift signals tampering |
| Consumption Entropy | Shannon entropy of daily usage distribution | Theft creates unnaturally low entropy |
| Night-to-Day Ratio | Mean night hours (10pm–6am) / mean day hours | Tampering sometimes shifts usage profile |
| Seasonal Deviation | Actual minus expected based on same month prior year | Compares against historical seasonal norm |

### Billing-Level Features

| Feature Name | Definition | Why It Matters |
|---|---|---|
| Monthly Billing Gap | Predicted monthly bill (from rolling avg) minus actual billed | Persistent negative gap is core theft signal |
| Billing Cycle Variance | Std dev of monthly total consumption over 6 months | Sudden variance drop can signal billing fraud |
| End-of-Month Drop | Ratio of last 5 days vs first 25 days of month | Billing fraud pattern — drops before reading |
| Year-over-Year Change | % change in monthly total vs same month last year | Flags suspicious sustained reductions |

### Anomaly Indicator Features

| Feature Name | Definition | Why It Matters |
|---|---|---|
| Zero Reading Count | Number of zero-reading days in 30-day window | Direct meter tampering indicator |
| Consecutive Zero Streak | Longest streak of zero readings | Extended outage vs deliberate bypass |
| Negative Delta Days | Count of days with >30% drop from prior day | Abrupt drops suspicious |
| Spike-Then-Drop | Days with spike followed within 3 days by drop | Common bypass signature |
| Low Value Persistency | Count of days below 20% of household average | Sustained under-reading flag |

### Fourier Frequency Features

| Feature Name | Definition | Why It Matters |
|---|---|---|
| Weekly Periodicity Score | FFT power at 7-day frequency | Normal households show strong weekly cycles |
| Monthly Periodicity Score | FFT power at 30-day frequency | Billing-aligned periodic behavior |
| Dominant Frequency | Peak frequency in FFT spectrum | Tampered meters show disrupted frequency |

**Total Feature Count Target:** 30–35 features per household per 90-day window.

---

## 5. Model Architecture

### Model 1 — Isolation Forest

**Type:** Unsupervised anomaly detection
**Library:** scikit-learn

**How it works:** Builds an ensemble of random decision trees. Anomalous samples (theft) require fewer splits to isolate because they occupy sparse regions of feature space. The anomaly score is the inverse of average path length.

**Configuration Decisions:**
- Number of estimators: 200 (higher = more stable scores)
- Contamination parameter: 0.05 (assumes ~5% anomaly rate — calibrate on SGCC validation set)
- Max samples: 256 (default, sufficient for this dataset size)
- Input features: all 30–35 engineered features
- Output: anomaly score (continuous, –1 to +1) per household

**When it excels:** Novel theft patterns not seen during training. No labels required. Fast inference on 10,000 records.

**When it struggles:** Correlated high-dimensional features. Suboptimal without feature selection.

**Preprocessing required:** StandardScaler normalization of all features before fitting.

---

### Model 2 — Local Outlier Factor (LOF)

**Type:** Unsupervised density-based anomaly detection
**Library:** scikit-learn

**How it works:** Computes the local density of each household relative to its k nearest neighbors. Households in low-density regions (isolated from their neighbors) receive high anomaly scores.

**Configuration Decisions:**
- n_neighbors: 20
- Contamination: 0.05
- Algorithm: ball_tree (efficient for large datasets)
- Metric: Euclidean on normalized features

**When it excels:** Catches anomalies that are locally isolated — households in a feeder/zone that behave very differently from their geographic neighbors.

**Combination rationale with Isolation Forest:** Isolation Forest is global (tree-based), LOF is local (density-based). Together they catch different anomaly geometries in feature space.

---

### Model 3 — LSTM Autoencoder

**Type:** Unsupervised deep learning — sequence reconstruction
**Library:** Keras (TensorFlow backend)

**Architecture:**

```
Input:  (batch, sequence_length=90, features=1)
        ↓
Encoder
  LSTM Layer 1:  128 units, return_sequences=True, activation=tanh
  Dropout:       0.2
  LSTM Layer 2:  64 units, return_sequences=False, activation=tanh
  Dense:         32 units (bottleneck — compressed representation)
        ↓
Repeat Vector: (batch, 90, 32)  — expand bottleneck back to sequence length
        ↓
Decoder
  LSTM Layer 3:  64 units, return_sequences=True
  Dropout:       0.2
  LSTM Layer 4:  128 units, return_sequences=True
  TimeDistributed Dense: 1 output per timestep
        ↓
Output: (batch, 90, 1)  — reconstructed sequence

Loss: Mean Squared Error (MSE)
Optimizer: Adam, lr=0.001
```

**Training Strategy:**
- Train ONLY on normal-class households from SGCC
- 90-day sliding window per household as one training sequence
- Threshold = mean(reconstruction error on validation normals) + 3 × std(validation errors)
- At inference: compute reconstruction error → compare to threshold → flag if above

**Why LSTM specifically:** Consumption data is a temporal sequence with daily rhythms and weekly cycles. LSTM networks capture these temporal dependencies. An autoencoder trained on normal patterns cannot reconstruct tampered sequences — the high MSE becomes the theft signal.

**Anomaly score output:** MSE reconstruction error, normalized to 0–1 range.

---

### Model 4 — XGBoost Classifier

**Type:** Supervised gradient-boosted trees
**Library:** XGBoost

**Why XGBoost as the fourth model:** After running the three unsupervised models, their scores can be used as additional meta-features alongside the engineered features for a supervised classifier. XGBoost trained on SGCC labels integrates all signals into a single theft probability.

**Input features:**
- All 30–35 engineered features
- Isolation Forest anomaly score (from Model 1)
- LOF anomaly score (from Model 2)
- LSTM reconstruction error (from Model 3)

**Configuration Decisions:**
- n_estimators: 500
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: ratio of normal to theft class (handles class imbalance)
- Early stopping: 50 rounds on validation AUC

**Output:** Probability of theft (0.0 to 1.0) per household.

**Why this is the strongest single model:** It sees all engineered features AND the outputs of the other three models as inputs. It is the only model with direct access to ground truth labels for training.

---

### Model Training Order

1. Train Isolation Forest and LOF on full feature set (no labels needed)
2. Generate their anomaly scores on all households
3. Train LSTM Autoencoder on normal-class sequences only
4. Compute LSTM reconstruction errors on all households
5. Train XGBoost on features + IF score + LOF score + LSTM error + SGCC labels
6. Save all four models with MLflow versioning

---

## 6. Risk Score Engine

### Score Fusion Formula

Each model outputs a score normalized to the range [0, 1]:

```
IF_score      = normalize(Isolation Forest raw score)
LOF_score     = normalize(Local Outlier Factor raw score)
LSTM_score    = normalize(LSTM reconstruction error)
XGB_score     = XGBoost theft probability (already 0–1)

Base_score = (0.15 × IF_score)
           + (0.10 × LOF_score)
           + (0.30 × LSTM_score)
           + (0.45 × XGB_score)
```

Weights rationale: XGBoost has ground truth labels, highest trust. LSTM captures temporal patterns, high trust. IF and LOF are baseline checks.

### Domain Rules Layer

After computing Base_score, apply rule-based additions to reward known theft signatures:

| Rule | Condition | Score Addition |
|---|---|---|
| Sustained drop | Rolling 30d mean drops >50% vs prior 30d mean AND persists >21 days | +15 |
| End-of-month drop | Last 5 days of billing month <25% of month average | +10 |
| Zero streak | Consecutive zero readings ≥ 7 days (non-holiday) | +20 |
| Billing gap | Monthly actual billing < 40% of predicted for 3 consecutive months | +15 |
| Zigzag signature | Alternating high-low days (stddev of daily delta > 2.5σ) over 30 days | +10 |

Final score = min(100, Base_score × 100 + Rule_additions)

### Risk Tier Classification

| Score Range | Classification | Action |
|---|---|---|
| 0 – 30 | Normal | No action required |
| 31 – 60 | Suspicious | Flag for scheduled review |
| 61 – 80 | High Risk | Priority inspection within 30 days |
| 81 – 100 | Theft | Immediate field investigation |

### Threshold Calibration Strategy

Thresholds are not fixed at deployment. They are calibrated on the SGCC held-out validation set using a precision-recall curve. The goal is to find the threshold that simultaneously achieves ≥85% precision and ≤10% false positive rate. This threshold is stored in config.yaml and is recalibrated automatically after every retrain cycle.

---

## 7. Dashboard Pages Plan

### Page 1 — Overview Dashboard

**Purpose:** Morning briefing screen for DISCOM managers.

**Components:**
- KPI row: Total households monitored, Theft-flagged count, Suspicious count, Total estimated revenue loss this month
- Consumption anomaly trend chart: Daily count of new anomaly flags over the past 90 days (Plotly line chart)
- Risk distribution histogram: Distribution of all household risk scores (0–100) as a histogram — normal curve vs flagged spike visible at a glance
- Folium interactive map: Geo-pins for all households, color-coded by risk tier (green/orange/red), clustered by feeder zone
- Feeder-level heatmap: Which Distribution Transformer zones have highest aggregate risk

**Data source:** Latest scores from SQLite results database, updated on every run.

---

### Page 2 — Household Inspector

**Purpose:** Deep-dive per consumer for field team preparation.

**Components:**
- Meter ID search box (type or select)
- Full 365-day consumption time-series chart with anomaly markers (red dots on flagged days), rolling 30-day mean overlay in orange
- Feature breakdown panel: Bar chart showing the household's top contributing features to its risk score (SHAP values from XGBoost)
- Comparison vs zone average: Household daily consumption vs the average of its feeder neighbors (overlay chart)
- Anomaly event log: Table of all flagged dates with anomaly type, score, and which model flagged it
- Risk score history: Line chart of monthly risk scores over past 12 months (trend direction matters)
- Export button: Download full household report as PDF

---

### Page 3 — Risk Leaderboard

**Purpose:** Generate the inspection queue for field teams.

**Components:**
- Full ranked table of all consumers sorted by risk score descending
- Columns: Rank, Meter ID, Feeder Zone, Address, Risk Score (progress bar), Classification badge, Last Anomaly Date, Alert Count Last 90 Days, Inspection Status
- Filters: Zone, classification tier, date range, score threshold slider
- Bulk action: Select multiple consumers → Export selected as CSV with addresses for field inspection
- Summary: Estimated collective annual revenue loss for selected consumers (calculated from billing gap feature)

---

### Page 4 — Model Performance

**Purpose:** Technical validation screen — used in presentations and interviews.

**Components:**
- Model comparison table: Side-by-side Precision, Recall, F1, AUC-ROC for all four models on the held-out SGCC test set
- ROC curve chart: All four models on one Plotly chart for direct visual comparison
- Confusion matrix: Heatmap for each model (2×2 grid layout)
- Precision-Recall curve: Shows operating point selected vs full curve
- Anomaly score distribution: Histogram of risk scores for known normal vs known theft from SGCC test set — should show clear bimodal separation
- Feature importance chart: Top 20 XGBoost SHAP features ranked by mean absolute contribution
- LSTM training curve: Train vs validation MSE over epochs (verifies no overfitting)

---

### Page 5 — Retrain Control Panel

**Purpose:** Operational pipeline for monthly data refresh.

**Components:**
- CSV upload widget: Drag and drop new monthly smart meter data
- Schema validation preview: Auto-validates uploaded file against expected schema — shows pass/fail per column
- Retrain button: Triggers full pipeline (preprocessing → features → all 4 models → evaluation)
- Progress bar: Live progress through pipeline stages
- Before/After metrics comparison: Shows old model vs newly retrained model performance
- Drift detection panel: Plots risk score distribution for current month vs prior 3 months — flags if drift detected
- Model registry: Table of all past model versions (MLflow) with their evaluation metrics and deployment status
- Active model selector: Choose which saved model version is used for scoring

---

## 8. Phase-by-Phase Implementation Plan

### Phase 1 — Data Collection & EDA (Week 1)

**Deliverable:** Clean EDA notebook with statistical summary and anomaly preview.

**Tasks:**
1. Download SGCC dataset from Kaggle — verify file integrity and column schema
2. Download UCI Electricity Load Diagrams from UCI ML Repository
3. Register for and download Irish CER dataset from ISSDA
4. Load SGCC into Pandas — examine shape, null counts, data types, label distribution
5. Plot sample time series for 10 normal and 10 theft households — visually identify pattern differences
6. Compute per-household statistics: mean, std, min, max, zero count, total days
7. Analyze label imbalance — compute exact theft percentage
8. Plot consumption distribution by label — verify statistical separation exists
9. Identify and document all data quality issues: missing meters, negative values, gaps, duplicates
10. Write EDA summary notebook with all findings and charts

**Success criterion:** Can visually distinguish at least 3 theft pattern types from normal patterns in raw plots.

---

### Phase 2 — Preprocessing & Feature Engineering (Week 2)

**Deliverable:** Feature matrix CSV per dataset. Preprocessing pipeline saved as a reusable class.

**Tasks:**
1. Build PreprocessingPipeline class with methods: load_csv, validate_schema, clip_outliers, normalize_per_household, impute_gaps, resample_to_daily
2. Apply pipeline to SGCC — output clean_sgcc.csv
3. Apply pipeline to UCI — output clean_uci.csv
4. Apply pipeline to Irish CER — output clean_irish.csv
5. Build FeatureEngineer class with methods for each feature group (statistical, pattern, billing, anomaly indicator, Fourier)
6. Implement all 30–35 features defined in Section 4
7. Run feature engineering on all three cleaned datasets
8. Compute correlation matrix — identify and drop features with correlation >0.95 (keep one from each pair)
9. Run SHAP pre-analysis on a simple DecisionTree to rank feature importance before deep model training
10. Generate synthetic theft samples using the five augmentation strategies in Section 3
11. Apply SMOTE to bring theft class to 20–25% of training data
12. Split SGCC into train (70%), validation (15%), test (15%) — stratify by label
13. Save feature matrices as parquet files for fast loading

**Success criterion:** Feature matrix shape is (n_households × 30+), no nulls, class ratio balanced.

---

### Phase 3 — Isolation Forest & LOF (Week 3)

**Deliverable:** Trained IF and LOF models with anomaly scores per household. Evaluation report.

**Tasks:**
1. Load feature matrix training set
2. Apply StandardScaler — fit on train, transform train/val/test separately (never fit on test)
3. Train Isolation Forest with contamination=0.05, n_estimators=200
4. Generate anomaly scores for all households in all splits
5. Evaluate on SGCC test set using known labels — compute precision, recall, F1, AUC-ROC
6. Plot anomaly score distribution for normal vs theft households — verify separation
7. Tune contamination parameter using precision-recall curve on validation set
8. Train LOF with n_neighbors=20, contamination=0.05
9. Repeat evaluation for LOF
10. Save both models using joblib
11. Log all experiments to MLflow — parameters, metrics, model artifacts
12. Write comparison report: IF vs LOF on precision, recall, false positive rate

**Success criterion:** At least one model achieves AUC-ROC > 0.75 on SGCC test set.

---

### Phase 4 — LSTM Autoencoder (Week 4)

**Deliverable:** Trained LSTM Autoencoder with reconstruction error scores. Threshold determined.

**Tasks:**
1. Prepare sequence dataset: 90-day sliding windows from SGCC daily consumption
2. Create training set from normal-class households only
3. Build LSTM Autoencoder in Keras with architecture from Section 5
4. Train with Adam optimizer, MSE loss, batch_size=64, epochs=100, early stopping (patience=10)
5. Monitor train vs validation loss curves — save best epoch checkpoint
6. Compute reconstruction errors on all training normal sequences
7. Determine anomaly threshold = mean(normal errors) + 3 × std(normal errors)
8. Compute reconstruction errors on SGCC test set (normal + theft)
9. Apply threshold — classify above as anomalous
10. Evaluate: precision, recall, F1, AUC-ROC on test set
11. Plot: distribution of reconstruction errors for normal vs theft (should show clear separation)
12. Plot: 5 example theft sequences — original vs reconstructed — visually show poor reconstruction
13. Log model, threshold, and all metrics to MLflow
14. Save model in .keras format

**Success criterion:** AUC-ROC > 0.80. Reconstruction error distributions show visible bimodal separation.

---

### Phase 5 — XGBoost Ensemble Model (Week 5)

**Deliverable:** Final XGBoost model integrating all model signals. Full evaluation report.

**Tasks:**
1. Load SGCC train feature matrix
2. Add IF anomaly score, LOF score, and LSTM reconstruction error as three additional columns
3. Train XGBoost classifier with parameters from Section 5
4. Use early stopping on validation AUC — save best model
5. Generate SHAP values for all features — rank by mean absolute contribution
6. Plot SHAP summary plot (top 20 features)
7. Plot SHAP waterfall plot for 3 theft examples — explain why they scored high
8. Evaluate on test set: precision, recall, F1, AUC-ROC, confusion matrix
9. Compare all 4 models on a single evaluation table
10. Plot ROC curves for all 4 models on one chart
11. Tune scale_pos_weight for best precision/FPR tradeoff
12. Log all experiments to MLflow
13. Save final XGBoost model

**Success criterion:** XGBoost achieves ≥85% precision and ≤10% false positive rate on test set.

---

### Phase 6 — Risk Score Engine (Week 6)

**Deliverable:** RiskScoreEngine class. Scored CSV for all households. Risk tier assignments.

**Tasks:**
1. Build RiskScoreEngine class with: normalize_scores, apply_weights, apply_domain_rules, classify_tier methods
2. Implement score normalization (min-max per model output range)
3. Implement weighted fusion formula from Section 6
4. Implement all 5 domain rule checks from Section 6
5. Implement final score capping at 100
6. Implement tier classification logic
7. Run scoring on full SGCC dataset + Irish CER dataset (for demo diversity)
8. Generate risk_scores.csv: household ID, risk score, tier, top flagged rule, model contributing most
9. Validate false positive rate on known-normal test households — must be ≤10%
10. Validate precision on known-theft test households — must be ≥85%
11. Save threshold values and weights to config.yaml for dashboard use
12. Build threshold recalibration function using precision-recall optimization — add to retrain pipeline

**Success criterion:** Risk score distribution shows clear separation between classes. Tier thresholds produce no more than 10% FPR.

---

### Phase 7 — Streamlit Dashboard (Week 7)

**Deliverable:** Fully functional 5-page Streamlit application. All charts, tables, and controls operational.

**Tasks:**
1. Set up Streamlit project structure (see Section 9)
2. Build shared data_loader.py module: load_scores(), load_household_timeseries(), load_model_metrics()
3. Build shared components: risk_badge(), kpi_card(), anomaly_chart()
4. Build Page 1 (Overview): KPI cards, trend chart, Folium map, feeder heatmap
5. Build Page 2 (Household Inspector): search, time-series chart with anomaly markers, SHAP chart, comparison overlay, event log, export button
6. Build Page 3 (Risk Leaderboard): ranked table, filters, bulk CSV export
7. Build Page 4 (Model Performance): model comparison table, ROC curves, confusion matrices, SHAP importance
8. Build Page 5 (Retrain Control Panel): CSV upload, schema validator, retrain trigger, progress display, drift chart
9. Apply consistent styling using Streamlit's config.toml and custom CSS
10. Test all pages with SGCC and Irish CER demo data
11. Test Folium map with synthetic geo-coordinates for households
12. Test CSV and PDF export functionality
13. Add caching with @st.cache_data for all database reads (performance requirement)

**Success criterion:** All 5 pages load without error. Retrain pipeline completes end-to-end from upload to updated scores.

---

### Phase 8 — Testing, Deployment & Documentation (Week 8)

**Deliverable:** Deployed application. Full technical report. Presentation deck.

**Tasks:**
1. Write unit tests for PreprocessingPipeline, FeatureEngineer, RiskScoreEngine
2. Write integration test: end-to-end pipeline from raw CSV to risk scores
3. Load test: run pipeline on 10,000 synthetic household records — verify completion in <60 seconds
4. Fix any performance bottlenecks (vectorize loops, use Polars for large transforms if needed)
5. Dockerize the application: write Dockerfile and docker-compose.yml
6. Write requirements.txt with pinned versions
7. Create demo dataset: 500 households with mix of normal, suspicious, theft for quick demo
8. Record 3-minute Loom walkthrough video of the dashboard
9. Write technical report: architecture, datasets, feature rationale, model comparison, results
10. Build presentation deck: problem → solution → architecture → demo → results → DISCOM impact
11. Push complete codebase to GitHub with comprehensive README.md

**Success criterion:** Application runs from docker-compose up with zero manual setup. All test cases pass. Report is submission-ready.

---

## 9. File & Folder Structure

```
electricity-theft-detection/
│
├── data/
│   ├── raw/                          # Original downloaded datasets (gitignored)
│   │   ├── sgcc/
│   │   ├── uci/
│   │   └── irish_cer/
│   ├── processed/                    # Output of preprocessing pipeline
│   │   ├── clean_sgcc.parquet
│   │   ├── clean_uci.parquet
│   │   └── features_sgcc.parquet
│   ├── synthetic/                    # Augmented theft scenarios
│   └── demo/                         # 500-household demo dataset
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Phase 1 EDA
│   ├── 02_feature_engineering.ipynb  # Phase 2 feature analysis
│   ├── 03_model_experiments.ipynb    # Phase 3–5 experiments
│   └── 04_evaluation.ipynb           # Final model comparison
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestor.py               # CSV loading and schema validation
│   │   ├── preprocessor.py           # PreprocessingPipeline class
│   │   └── augmentor.py              # Synthetic theft generation + SMOTE
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py       # FeatureEngineer class (all 30+ features)
│   │   └── fourier_features.py       # FFT-based feature extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py       # IF training and scoring
│   │   ├── lof.py                    # LOF training and scoring
│   │   ├── lstm_autoencoder.py       # LSTM AE architecture and training
│   │   └── xgboost_model.py          # XGBoost training and SHAP
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── risk_score_engine.py      # Ensemble fusion + domain rules
│   │   └── threshold_calibrator.py   # Precision-recall threshold optimizer
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Precision, recall, F1, AUC-ROC
│   │   └── visualizations.py         # ROC curves, confusion matrices
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py         # Full training orchestration
│       └── retrain_pipeline.py       # Incremental retrain on new data
│
├── dashboard/
│   ├── app.py                        # Streamlit entry point — page router
│   ├── config.toml                   # Streamlit theme configuration
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Household_Inspector.py
│   │   ├── 3_Risk_Leaderboard.py
│   │   ├── 4_Model_Performance.py
│   │   └── 5_Retrain_Panel.py
│   └── components/
│       ├── data_loader.py            # Shared database reads
│       ├── charts.py                 # Reusable Plotly chart builders
│       ├── map.py                    # Folium map builder
│       └── export.py                 # PDF and CSV export utilities
│
├── database/
│   ├── schema.sql                    # SQLite schema definition
│   └── theft_detection.db            # Runtime database (gitignored)
│
├── models_saved/                     # Trained model artifacts (gitignored)
│   ├── isolation_forest_v1.joblib
│   ├── lof_v1.joblib
│   ├── lstm_autoencoder_v1.keras
│   └── xgboost_v1.json
│
├── mlruns/                           # MLflow experiment tracking (gitignored)
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_features.py
│   ├── test_risk_engine.py
│   └── test_integration.py
│
├── config/
│   └── config.yaml                   # All tunable parameters and thresholds
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 10. Non-Functional Requirements Strategy

### Performance: 10,000 households in 60 seconds

- Use Pandas vectorized operations exclusively — no Python-level row loops
- Store processed features in Parquet format (10–15x faster than CSV reads)
- Scikit-learn models (IF, LOF, XGBoost) run in parallel using n_jobs=-1
- Streamlit pages use @st.cache_data with TTL=3600 to avoid re-loading data on every interaction
- If vectorized Pandas is still too slow, replace the feature engineering loop with Polars (5–10x faster for time-series window operations)
- Benchmark: run time_pipeline.py script on 10,000 synthetic rows and verify <60s end-to-end

### Precision: ≥85%

- Use XGBoost as the primary model with ground truth labels
- Tune scale_pos_weight for class imbalance
- Calibrate threshold on precision-recall curve, not default 0.5
- Ensemble fusion combines weak signals into a stronger one
- SHAP-guided feature selection removes noise features before final training

### False Positive Rate: ≤10%

- Threshold calibration on held-out validation set specifically optimizes FPR
- Domain rules layer requires multiple signals — no single rule fires a high alert alone
- Final tier classification requires score ≥61 for High Risk (not just any anomaly)
- Monthly drift detection prevents accumulated FPR increase over time

### Auto-Retrain

- Retrain pipeline is triggered by CSV file upload in the dashboard
- Pipeline validates new data schema before accepting it
- After training, new model is registered in MLflow with all metrics
- Old model remains as fallback until new model passes minimum metric thresholds
- Config.yaml thresholds are recalibrated automatically after every retrain
- Drift detection compares new score distributions to historical baseline — alerts if KL divergence exceeds threshold

---

## 11. Evaluation Framework

### Primary Metrics

All models are evaluated on the SGCC held-out test set (15% stratified split, never touched until final evaluation).

| Metric | Definition | Target |
|---|---|---|
| Precision | TP / (TP + FP) — of flagged, how many are truly theft | ≥85% |
| Recall | TP / (TP + FN) — of all thefts, how many are caught | ≥75% |
| F1 Score | Harmonic mean of precision and recall | ≥80% |
| AUC-ROC | Area under receiver operating characteristic curve | ≥0.88 |
| False Positive Rate | FP / (FP + TN) | ≤10% |

### Secondary Metrics

- Average risk score for known-theft households vs known-normal households (should differ by ≥40 points)
- Ranking quality: NDCG@50 (top 50 households should be enriched with true theft cases)
- Processing time: full pipeline on 10,000 records ≤60 seconds
- Model size: saved XGBoost model ≤50MB for production feasibility

### Evaluation Comparison Table Format

| Model | Precision | Recall | F1 | AUC-ROC | FPR | Inference Time (10k) |
|---|---|---|---|---|---|---|
| Isolation Forest | — | — | — | — | — | — |
| Local Outlier Factor | — | — | — | — | — | — |
| LSTM Autoencoder | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |
| Ensemble (all) | — | — | — | — | — | — |

Fill in during Phase 3–6 evaluations.

---

## 12. Real-World Differentiators

### Differentiator 1 — Feeder / DT-Level Aggregation

Real DISCOM inspection teams are dispatched by Distribution Transformer (DT) zone, not by individual household. The risk score engine aggregates individual scores at the DT level to produce a feeder risk index. Field teams are dispatched to the highest-risk feeder, where they can then investigate all flagged consumers in one visit. This is the single feature that makes this directly usable by TSNPDCL / APEPDCL.

### Differentiator 2 — Temporal Pattern Fingerprinter

A rule-based pattern classifier identifies specific known theft signatures before the ML models even run. The five patterns (meter tamper, billing fraud, direct bypass, zigzag, gradual reduction) are implemented as deterministic rules based on consumption statistics. If a pattern matches, it is recorded in the household's anomaly log and feeds into the domain rules layer of the risk score engine. This hybrid ML + domain rules approach mirrors what operational fraud detection systems at actual utilities use.

### Differentiator 3 — Concept Drift Monitoring

Consumption patterns change seasonally. A model trained in winter performs poorly in summer because AC usage creates patterns that look anomalous to a winter-trained detector. A monthly KL divergence check compares the current month's risk score distribution to the trailing 3-month average. If divergence exceeds a configurable threshold, the dashboard alerts the operator and the retrain pipeline is triggered automatically. This prevents accuracy degradation in production.

### Differentiator 4 — SHAP Explainability for Field Teams

Every flagged household shows the top 5 SHAP features that contributed to its risk score. A field inspector looking at "Household 4821 — Risk Score 87" also sees "Top reasons: 1) Billing gap 43% below expected. 2) 12 consecutive zero-reading days. 3) End-of-month drop pattern detected." This transparency is required for legal action — a DISCOM cannot disconnect a consumer based on a black-box score; they need documented evidence.

### Differentiator 5 — Revenue Loss Quantification

The risk leaderboard includes an estimated annual revenue loss column for each flagged household, calculated from the billing gap feature: estimated_loss = billing_gap × 12 × tariff_rate. The Page 3 bulk export shows total estimated loss for the selected inspection batch. This converts the system from a technical tool into a business case — the operator can report "this batch of 50 inspections recovers approximately ₹28 lakhs per year."

---

## 13. Deployment Plan

### Local Development Setup

Install dependencies from requirements.txt, set PYTHONPATH to project root, initialize the SQLite database with schema.sql, run the training pipeline to generate model artifacts, and launch the Streamlit app.

### Docker Deployment

The Dockerfile installs all Python and system dependencies. The docker-compose.yml defines two services: the training pipeline service (run once to generate models) and the dashboard service (persistent web server). Mounting a local data volume allows new CSVs to be dropped without rebuilding the container.

### Configuration Management

All tunable parameters live in config/config.yaml. This includes model hyperparameters, risk score weights, tier thresholds, domain rule parameters, retrain schedule settings, and dataset paths. The dashboard reads thresholds from this file at startup. The retrain pipeline writes updated thresholds back to this file after calibration.

### MLflow Experiment Tracking

Every training run — including all intermediate experiments — is logged to MLflow with full parameter sets, evaluation metrics, and model artifacts. The Page 5 retrain panel reads from MLflow to display model version history. When deploying a retrained model, the operator selects the version from the MLflow registry in the dashboard.

---

## 14. Glossary

| Term | Definition |
|---|---|
| DISCOM | Distribution Company — entity responsible for electricity distribution to consumers in India |
| TSNPDCL | Telangana State Northern Power Distribution Company Limited |
| APEPDCL | Andhra Pradesh Eastern Power Distribution Company Limited |
| DT | Distribution Transformer — the transformer in a local neighborhood that feeds multiple households |
| SGCC | State Grid Corporation of China — source of the primary labeled theft dataset |
| Isolation Forest | Unsupervised anomaly detection algorithm using random tree partitioning |
| LOF | Local Outlier Factor — density-based anomaly detection algorithm |
| LSTM | Long Short-Term Memory — a type of recurrent neural network suited to sequential data |
| Autoencoder | Neural network trained to reconstruct its input; high reconstruction error signals anomaly |
| XGBoost | Extreme Gradient Boosting — ensemble tree method, state-of-the-art for tabular data |
| SHAP | SHapley Additive exPlanations — framework for explaining individual model predictions |
| MLflow | Open-source platform for tracking ML experiments and managing model versions |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve — model discrimination metric |
| SMOTE | Synthetic Minority Over-sampling Technique — creates synthetic examples of minority class |
| KL Divergence | Kullback-Leibler Divergence — statistical measure of distribution difference (used for drift detection) |
| Reconstruction Error | MSE between LSTM autoencoder input and its output — anomaly signal |
| Contamination | Isolation Forest / LOF parameter specifying expected fraction of outliers in dataset |
| Concept Drift | Change in underlying data distribution over time, degrading model accuracy |

---

*Document prepared for: Electricity Theft Detection System — Full Implementation Reference*
*Version: 1.0 | Status: Ready for Development*
