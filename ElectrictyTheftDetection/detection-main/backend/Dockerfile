FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip3 install -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy shared data and models
COPY data/ ./data/
COPY models_saved/ ./models_saved/

WORKDIR /app/backend

# Default command: run the training pipeline
CMD ["python", "src/pipeline/train_pipeline.py"]
