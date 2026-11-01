# Customer churn prediction

A small ML project built around the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). I compare a few classifiers, tune the tree-based models, and use SHAP to inspect the predictions. There is also a FastAPI endpoint and a Streamlit form for trying the saved XGBoost model.

## Dataset

`Customer_Churn.csv` contains 7,043 customer records. The task is binary classification: predict whether a customer leaves the service (`Churn = Yes`). Features cover account tenure, contract terms, subscribed services and billing.

`TotalCharges` has a few blank entries. The notebook converts it to numeric and handles missing values in the preprocessing pipeline.

## Models and evaluation

The notebook includes:

- Logistic Regression (L1, L2 and Elastic Net)
- Random Forest
- XGBoost
- LightGBM

The data is split into stratified train and test sets (80/20). I compare models using 5-fold cross-validation on the training split, with ROC-AUC as the main metric. Optuna tunes Random Forest and XGBoost. The saved model is the tuned XGBoost, following the original experiment.

The notebook writes CV scores to `reports/baseline_cv_results.csv`, tuned scores to `reports/tuned_cv_results.csv`, test metrics to `reports/test_metrics.json`, and figures to `reports/figures/`. These are generated when the notebook runs; I don't include a new score here until the updated experiment has been rerun on the full dataset.

SHAP plots show which features contribute most to the model's raw score. The contributions are in log-odds, not percentage points of churn probability.

## Run the notebook

Requires Python 3.11.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `First Project. Customer churn prediction.ipynb` and run the cells in order. The CSV is already in the repository root. The Optuna trial count is set near the top of the notebook.

The notebook exports `churn_model.joblib`, `xgb_churn_model.joblib` and `encodings.json` at the repository root. Both `.joblib` files contain the same fitted pipeline; the API uses `churn_model.joblib`.

## Try the app

The repository includes a pre-trained model, so the notebook does not need to be rerun to try the app.

Run the API and web interface in separate terminals:

```bash
uvicorn app:app --reload
streamlit run ui.py
```

The API is available at `http://localhost:8000/docs`; Streamlit runs at `http://localhost:8501`. The interface uses `CHURN_API_URL` to find the API (default: `http://localhost:8000`).

Or run both services with Docker:

```bash
docker build -t churn-prediction .
docker run --rm -p 8000:8000 -p 8501:8501 churn-prediction
```

The `/predict` endpoint accepts the same customer fields as the dataset, excluding `customerID` and `Churn`. It returns the predicted probability, a simple risk band and three SHAP factors. Fixed yes/no and contract mappings are applied before passing the row to the saved sklearn pipeline.

## Files

```text
Customer_Churn.csv                       Dataset
First Project. Customer churn prediction.ipynb
app.py                                   FastAPI inference
ui.py                                    Streamlit frontend
churn_model.joblib                       Model used by the API
xgb_churn_model.joblib                   Second export of the same model
encodings.json                          Feature mappings
requirements.txt
Dockerfile
supervisord.conf
README.md
```

## Notes

This is a comparison exercise, not a production churn system. The probability threshold and risk bands are illustrative; they have not been calibrated against retention costs. The SHAP factors describe model behavior, not causal effects. The pre-trained model files are kept for the demo, while a new notebook run will overwrite them with a newly fitted model.