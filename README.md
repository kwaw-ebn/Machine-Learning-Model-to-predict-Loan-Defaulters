# Loan Portfolio Lab

A full stack, **educational model explorer** built from the repository's saved RandomForestClassifier. The former Streamlit script is retained for provenance, while `frontend/` and `backend/` are the current application.

## What changed

- The static frontend collects every required numerical input and a loan purpose; it provides a fictional example, responsive result view, and plain language model limitations.
- The FastAPI service loads only the committed `loan_default_model.joblib`, checks its 19 feature names and class labels, validates inputs, and returns the saved model's score for `not.fully.paid`.
- Annual income is transformed with the natural logarithm and credit history years with 365.25 days. Interest percentage is divided by 100. The seven purpose indicators are one hot encoded. No missing feature is silently set to zero.
- Scenarios are sent to the API for one request and are not persisted. No applicant identity or account data is collected.

## Important model limitations

The label in the training notebook is `not.fully.paid`. It must not be presented as a verified default event or as a loan approval decision. The notebook contains illustrative hardcoded cross validation values and has no dataset in this repository. Consequently there is no reproducible claim for current discrimination, calibration, subgroup behavior, or generalization. The displayed value is an uncalibrated random forest score. It is not suitable for lending decisions, pricing, or adverse action reasons. Use fictional scenarios only. Before any institutional use, obtain the training data and documented consent or rights, reproduce a leakage free temporal evaluation, assess calibration and subgroup performance, monitor drift, and independently review security and applicable law.

The joblib file is a pickle based artifact and should be loaded only from this trusted repository. It was created with scikit-learn 1.5.1; the backend pins that version and Python 3.12.

## Run locally

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
FRONTEND_ORIGIN=http://localhost:5500 uvicorn backend.main:app --reload --port 8000
```

In another terminal: `python -m http.server 5500 --directory frontend`. Open `http://localhost:5500`. `frontend/config.js` points to the Render API by default; set `window.LOAN_API_URL='http://localhost:8000'` there during local testing.

## Deploy to Render

`render.yaml` defines a free Python web service and a static frontend in Frankfurt. Import the repository as a Blueprint, or create both services using the same commands and names. Python version is pinned in `.python-version`. The API permits only the frontend origin set by `FRONTEND_ORIGIN`. Verify `/health`, `/api/model-info`, one fictional score, browser CORS, and responsive layout after deployment.

The original Streamlit app and notebook are left intact for reference. They are not part of the deployed runtime.
