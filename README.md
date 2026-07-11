# 💳 AI-Powered Fraud Detection System

An end-to-end Machine Learning application for detecting fraudulent financial transactions using the Credit Card Fraud Detection dataset. The project includes data preprocessing, model training, REST API integration, an interactive Streamlit dashboard, and prediction history storage.

---

## 📌 Project Overview

Financial fraud has become a major concern with the rapid growth of digital payment systems. This project uses Machine Learning to automatically identify potentially fraudulent credit card transactions.

The application allows users to upload transaction datasets, predicts whether each transaction is fraudulent, assigns a fraud probability and risk level, visualizes insights through an interactive dashboard, and stores prediction history in a SQLite database.

---

## 🚀 Features

- 📂 Upload transaction CSV files
- 🤖 Predict fraudulent transactions using Random Forest
- 📊 Interactive dashboard with Plotly visualizations
- 📈 Fraud probability scoring
- 🚦 Risk level classification (Low / Medium / High)
- 📜 Prediction history stored in SQLite
- 📥 Download prediction results as CSV
- ⚡ FastAPI backend for inference
- 📱 Clean Streamlit user interface

---

# Dataset

**Credit Card Fraud Detection Dataset**

- Source: Kaggle
- Total Transactions: **284,807**
- Fraud Cases: **492**
- Fraud Percentage: **0.172%**

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

---

# Machine Learning Pipeline

### Data Cleaning

- Removed duplicate transactions
- Checked for missing values
- Feature scaling using StandardScaler

### Handling Class Imbalance

- SMOTE (Synthetic Minority Oversampling Technique)

### Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Final Selected Model

✅ Random Forest Classifier

Reason:
- Highest ROC-AUC
- Strong precision-recall balance
- Better generalization compared to Decision Tree

---

# Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 97.37% | 5.30% | 87.37% | 10.00% | 96.26% |
| Decision Tree | 99.75% | 35.67% | 64.21% | 45.86% | 82.00% |
| **Random Forest** | **99.92%** | **73.53%** | **78.95%** | **76.14%** | **97.93%** |
| XGBoost | 99.92% | 74.26% | 78.95% | 76.53% | 97.02% |

Cross Validation (Random Forest)

Average F1 Score:
```
0.8335
```

---

# Project Structure

```
Fraud-Detection/
│
├── app.py                 # Streamlit Dashboard
├── api.py                 # FastAPI Backend
├── database.py            # SQLite operations
├── fraud_model.pkl
├── scaler.pkl
├── fraud.db
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt
└── README.md
```

---

# Dashboard Features

### Dashboard

- Upload CSV
- Fraud Detection
- Fraud Probability
- Risk Level Assignment
- Interactive Charts

### Visualizations

- Fraud Distribution Pie Chart
- Prediction Count Bar Chart
- Transaction Amount Histogram
- Fraud vs Transaction Amount Box Plot
- Highest Fraud Transactions

### Prediction History

- SQLite Storage
- Timestamp
- Risk Level
- Fraud Probability

---

# Tech Stack

### Machine Learning

- Scikit-learn
- XGBoost
- SMOTE

### Backend

- FastAPI
- Joblib

### Frontend

- Streamlit
- Plotly

### Database

- SQLite

### Language

- Python

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/Fraud-Detection.git
```

Move inside the project

```bash
cd Fraud-Detection
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit

```bash
streamlit run app.py
```

---

# Run FastAPI

```bash
uvicorn api:app --reload
```

---

# API Endpoint

POST

```
/predict
```

Input

```json
{
    "features": [
        ...
    ]
}
```

Output

```json
{
    "is_fraud": 1,
    "fraud_probability": 96.8,
    "status": "Success"
}
```

---

# Future Improvements

- Hyperparameter tuning using GridSearchCV
- Ensemble learning
- Real-time transaction monitoring
- Deep Learning models
- Explainable AI using SHAP
- Docker deployment
- Cloud deployment on AWS

---

# Authors

Developed as part of an internship project on Machine Learning based Financial Fraud Detection.

---

# License

This project is intended for educational and internship purposes.
