FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy frontend requirements and install
COPY frontend/requirements.txt .
RUN pip3 install -r requirements.txt

# Copy frontend code
COPY frontend/ ./frontend/

# Copy shared data and models
COPY data/ ./data/
COPY models_saved/ ./models_saved/

# Copy streamlit config
COPY frontend/.streamlit/ ./frontend/.streamlit/

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

WORKDIR /app/frontend
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
