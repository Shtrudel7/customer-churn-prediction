# 🚀 Customer Churn Prediction Service

## 📌 Business Problem
Customer churn is a critical metric for subscription-based businesses. Retaining an existing customer is significantly cheaper than acquiring a new one. The goal of this project is to build an end-to-end Machine Learning pipeline that predicts the likelihood of a customer leaving the telecom company and explains the key drivers behind each prediction.

## 📊 Dataset
The project uses the **Telco Customer Churn** dataset (IBM Sample Data Set).
- **Source:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features (demographics, services, billing).
- **Target:** `Churn` (Binary: Yes/No).
- **Class Imbalance:** ~26.5% churn rate.

## 🛠️ Tech Stack & Tools
- **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
- **Hyperparameter Tuning:** Optuna
- **Explainability (XAI):** SHAP
- **Backend API:** FastAPI, Uvicorn
- **Frontend UI:** Streamlit
- **Deployment:** Docker, Supervisor

## 🧠 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Handled missing values in `TotalCharges` (converted from object to numeric, filled NaNs with 0).
- Identified key churn drivers: `Contract` type (Month-to-month), `tenure` (new customers), and `InternetService` (Fiber optic).

### 2. Preprocessing & Feature Engineering
- Encoded binary features (0/1) and ordinal features (`Contract`).
- Applied `OneHotEncoder` for nominal features (`InternetService`, `PaymentMethod`).
- Used `ColumnTransformer` and `Pipeline` to prevent data leakage and ensure reproducibility.
- Applied `StandardScaler` for numerical features.

### 3. Modeling & Hyperparameter Tuning
- Tested multiple models: Logistic Regression (L1/L2), Random Forest, and XGBoost.
- Used **Stratified K-Fold Cross-Validation** (5 folds) to evaluate model stability.
- Applied **Optuna** for hyperparameter optimization.
- **Final Model:** Tuned XGBoost (ROC-AUC: ~0.85, Recall: ~0.82).

### 4. Explainability (SHAP)
- Integrated SHAP to explain both global feature importance and local (per-customer) predictions.
- Visualized findings using Bar, Beeswarm, and Waterfall plots to translate model outputs into actionable business insights.

### 5. Deployment (API & UI)
- **FastAPI:** REST API that accepts JSON payload, predicts churn probability, and returns top-3 SHAP reasons for the prediction.
- **Streamlit:** Interactive web interface for business users to input data and view results.
- Both services run simultaneously inside a single Docker container managed by `supervisord`.

## 📁 Project Structure
```text
.
├── app.py                      # FastAPI backend
├── ui.py                       # Streamlit frontend
├── churn_model.joblib          # Trained XGBoost pipeline
├── encodings.json              # Categorical mapping rules
├── Dockerfile                  # Docker build instructions
├── supervisord.conf            # Process manager config
├── requirements.txt            # Python dependencies
├── Customer_Churn.csv          # Raw dataset
├── First Project...ipynb       # Notebook with EDA, Modeling, SHAP
└── README.md                   # Project documentation
