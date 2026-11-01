import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "churn_model.joblib")
with (ROOT / "encodings.json").open(encoding="utf-8") as f:
    encodings = json.load(f)

preprocessor = model.named_steps["preprocessor"]
explainer = shap.TreeExplainer(model.named_steps["classifier"])
feature_names = [
    name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
]

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]
PhoneOption = Literal["Yes", "No", "No phone service"]


class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: PhoneOption
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


def prepare_customer(customer: CustomerData) -> pd.DataFrame:
    values = customer.model_dump()
    for column, mapping in encodings["binary_map"].items():
        if column in values:
            label = values[column]
            if label in ("No internet service", "No phone service"):
                label = "No"
            values[column] = mapping[label]
    values["Contract"] = encodings["contract_map"][values["Contract"]]
    return pd.DataFrame([values])


def explain_customer(features: pd.DataFrame) -> list[dict]:
    transformed = preprocessor.transform(features)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    effects = values[0]
    indices = np.argsort(np.abs(effects))[-3:][::-1]
    return [
        {
            "feature": feature_names[i],
            "impact": round(float(effects[i]), 4),
            "direction": "increases risk" if effects[i] > 0 else "decreases risk",
        }
        for i in indices
    ]


app = FastAPI(title="Customer Churn API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    features = prepare_customer(customer)
    probability = float(model.predict_proba(features)[0, 1])
    risk = "High" if probability >= 0.6 else "Medium" if probability >= 0.3 else "Low"
    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "top_factors": explain_customer(features),
    }
