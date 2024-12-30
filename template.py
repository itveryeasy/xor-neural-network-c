import os
import logging
from dask import dataframe as dd
from ray import tune
import tensorflow as tf
import json

# Create and configure logger
logging.basicConfig(
    filename="logs/app.log",
    format='%(asctime)s %(message)s',
    filemode='w'
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def create_file(path, content):
    try:
        with open(path, 'w') as file:
            file.write(content)
        print(f"Created file: {path}")
    except Exception as e:
        logger.error(f"Failed to create file: {path}. Error: {e}")

def setup_project_structure(root_folder):
    directories = [
        root_folder,
        os.path.join(root_folder, "data"),
        os.path.join(root_folder, "models"),
        os.path.join(root_folder, "src"),
        os.path.join(root_folder, "logs"),
        os.path.join(root_folder, "configs"),
        os.path.join(root_folder, "streamlit"),
        os.path.join(root_folder, "docker"),
        os.path.join(root_folder, "k8s"),
        os.path.join(root_folder, "terraform")
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory: {directory}. Error: {e}")

    # Create initial files
    create_file(os.path.join(root_folder, "README.md"), "# Project: Scalable Financial Data Prediction\n\nThis project implements an MLOps pipeline for financial data prediction using scalable and distributed frameworks.")

    create_file(os.path.join(root_folder, "requirements.txt"), """
# Python dependencies
numpy
pandas
dask
ray[default]
tensorflow
joblib
streamlit
scikit-learn
kubernetes
""")

    create_file(os.path.join(root_folder, "configs", "config.json"), '{\n    "dataset_path": "data/financial_data.csv",\n    "model_path": "models/random_forest_model.pkl",\n    "log_path": "logs/app.log",\n    "test_size": 0.2,\n    "random_state": 42,\n    "batch_size": 1024\n}')

    create_file(os.path.join(root_folder, "src", "data_preprocessing.py"), """
import dask.dataframe as dd

def load_data(file_path):
    try:
        return dd.read_csv(file_path)
    except Exception as e:
        print(f'Error loading data: {e}')
        return None

def preprocess_data(df):
    df = df.dropna()
    return df
""")

    create_file(os.path.join(root_folder, "src", "train_model.py"), """
import dask.dataframe as dd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import tensorflow as tf
import os

def train_random_forest(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    data = dd.read_csv(config['dataset_path'])
    data = data.dropna().compute()

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )

    model = RandomForestClassifier(random_state=config['random_state'])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    joblib.dump(model, config['model_path'])
    print(f'Model saved to {config['model_path']}')

if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.json')
    train_random_forest(config_path)
""")

    create_file(os.path.join(root_folder, "src", "main.py"), """
import os
from data_preprocessing import load_data, preprocess_data
from train_model import train_random_forest

def main():
    config_path = os.path.join('configs', 'config.json')
    train_random_forest(config_path)

if __name__ == '__main__':
    main()
""")

    create_file(os.path.join(root_folder, "streamlit", "app.py"), """
import streamlit as st
import pandas as pd
import joblib
import os

st.title('Scalable Financial Data Prediction')

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.write(data.head())

    model_path = os.path.join('models', 'random_forest_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        predictions = model.predict(data)
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.error("Model file not found. Please train the model first.")
""")

    create_file(os.path.join(root_folder, "docker", "Dockerfile"), """
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.enableCORS=false"]
""")

    create_file(os.path.join(root_folder, "docker", "docker-compose.yml"), """
version: '3.8'
services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
""")

    create_file(os.path.join(root_folder, "k8s", "deployment.yaml"), """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-prediction-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financial-prediction-app
  template:
    metadata:
      labels:
        app: financial-prediction-app
    spec:
      containers:
      - name: app
        image: financial-prediction:latest
        ports:
        - containerPort: 8501
""")

    create_file(os.path.join(root_folder, "terraform", "main.tf"), """
provider "aws" {
  region = "us-west-2"
}

resource "aws_ecs_cluster" "financial_prediction_cluster" {}

resource "aws_ecs_service" "financial_prediction_service" {
  cluster        = aws_ecs_cluster.financial_prediction_cluster.id
  desired_count  = 3
}
""")

    create_file(os.path.join(root_folder, "data", ".gitkeep"), "")
    create_file(os.path.join(root_folder, "models", ".gitkeep"), "")
    create_file(os.path.join(root_folder, "logs", ".gitkeep"), "")

if __name__ == "__main__":
    project_name = input("Enter the project name: ").strip()

    if not project_name:
        print("Project name cannot be empty!")
    else:
        setup_project_structure(project_name)
        print(f"Scalable MLOps project setup complete in folder: {project_name}")
