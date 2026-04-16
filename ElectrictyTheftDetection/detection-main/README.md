# ⚡ AI Electricity Theft Detection System

A cutting-edge, ML-powered revenue protection system designed to identify anomalies, predict risk, and visualize theft patterns in power distribution networks. 

---

## ✨ Features

- **4-Model ML Ensemble:** Combines supervised (XGBoost) and unsupervised (Isolation Forest, LOF, LSTM Autoencoder) models for robust detection.
- **Dynamic Risk Scoring:** Aggregated scoring system with weighted contributions from all models.
- **Interactive Dashboard:** Built with Streamlit for real-time visualization of network health.
- **Geographic Risk Heatmaps:** Pinpoint high-risk nodes (simulated) on interactive maps.
- **Household Deep-Dive:** Inspect individual consumer consumption patterns with time-series analysis.
- **Batch Processing:** Scalable pipeline for ingesting and processing large power datasets.

---

## 🏗️ Root Directory Structure

The project is split into two specialized services for ease of deployment and maintenance:

```
Electricity_Theft_Detection/
├── backend/              # ML Pipeline, Data Processing, Models & Docs
│   ├── Dockerfile           # Backend container build
│   ├── requirements.txt     # Backend Python dependencies
│   ├── config/              # Model & pipeline configuration
│   ├── database/            # SQLite schema definition
│   ├── docs/                # Project documentation & implementation plans
│   ├── models_saved/        # Trained model artifacts (.joblib, .keras)
│   ├── extract_sample.py    # Data sampling utility
│   └── src/                 # All data processing & model code
│
├── frontend/             # Streamlit Dashboard (User-facing)
│   ├── .streamlit/          # Streamlit theme & server config
│   ├── Dockerfile           # Frontend container build
│   ├── requirements.txt     # Frontend Python dependencies
│   ├── app.py               # Main dashboard entry point
│   ├── components/          # Reusable dashboard components
│   └── pages/               # Multi-page Streamlit pages
│
├── data/                 # Shared data directory
│   ├── processed/           # Processed CSV/Parquet files
│   └── raw/                 # Raw input data
│
├── railway.toml          # Railway deployment config
├── .dockerignore         # Docker build exclusions
└── .gitignore            # Git exclusions
```

---

## 🚀 Getting Started

### 1. Project Organization
The code is now fully organized into `frontend` and `backend` directories. This structure ensures that only the necessary dependencies and configurations are used for each part of the system.

### 2. Frontend (Dashboard)
To run the dashboard locally:
```bash
# Navigate to the frontend directory
cd frontend

# Install the necessary requirements
pip install -r requirements.txt

# Start the dashboard
streamlit run app.py --server.port=8080
```

### 3. Backend (ML Pipeline)
To train the models or process raw data:
```bash
# Navigate to the backend directory
cd backend

# Install the necessary requirements
pip install -r requirements.txt

# Run the training pipeline
python src/pipeline/train_pipeline.py
```

---

## 🌐 Deployment (Railway)

This project is configured for seamless deployment on **Railway.app**.

1. Connect your repository to Railway.
2. The `railway.toml` file will automatically instruct Railway to build the **frontend** service using the provided `frontend/Dockerfile`.
3. If you want to deploy the **backend** as a separate worker service, you can configure another service in Railway pointing to `backend/Dockerfile`.

---

## 🛠️ Tech Stack

- **Languages:** Python 3.9+
- **Frontend:** Streamlit, Plotly, HTML/CSS
- **ML Frameworks:** Scikit-learn, XGBoost, TensorFlow/Keras
- **Data:** Pandas, NumPy, Scipy, SQLite
- **DevOps:** Docker, Railway.toml

---

## 📄 License & Docs
Detailed implementation plans and domain rules can be found in the `docs/` folder.

> *Powered by Advanced Agentic Coding.*
