from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import shap
import numpy as np

app = FastAPI(title="Churn Prediction API")

# Загружаем модель и маппинги при старте
model = joblib.load("churn_model.joblib")
with open("encodings.json") as f:
    encodings = json.load(f)

# Достаем препроцессор и классификатор для SHAP
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
explainer = shap.TreeExplainer(classifier)
feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]

# Схема входных данных (упрощенная для примера)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerData):
    # 1. Преобразуем в dict и кодируем бинарные признаки
    input_dict = data.dict()
    for col, mapping in encodings["binary_map"].items():
        if col in input_dict:
            input_dict[col] = mapping[input_dict[col]]
    
    # Кодируем остальные
    input_dict["Contract"] = encodings["contract_map"][input_dict["Contract"]]
    
    # Примечание: InternetService и PaymentMethod мы оставили для OneHotEncoder,
    # поэтому их передаем как строки в DataFrame, препроцессор сам их закодирует.
    
    df = pd.DataFrame([input_dict])
    
    # 2. Предсказание
    proba = model.predict_proba(df)[0, 1]
    
    # 3. SHAP объяснение
    df_transformed = preprocessor.transform(df)
    shap_values = explainer.shap_values(df_transformed)[0]
    
    # Находим топ-3 фактора
    top_indices = np.argsort(np.abs(shap_values))[::-1][:3]
    top_factors = []
    for i in top_indices:
        top_factors.append({
            "feature": feature_names[i],
            "impact": float(shap_values[i]),
            "direction": "increases risk" if shap_values[i] > 0 else "decreases risk"
        })
    
    return {
        "churn_probability": round(float(proba), 4),
        "risk_level": "High" if proba > 0.6 else "Medium" if proba > 0.3 else "Low",
        "top_factors": top_factors
    }