"""Read-only educational scoring API for the repository's historical model."""
from __future__ import annotations
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / 'loan_default_model.joblib'
PURPOSES = ('all_other', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'major_purchase', 'small_business')
EXPECTED = ('credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', *(f'purpose_{p}' for p in PURPOSES))

class Scenario(BaseModel):
    model_config = ConfigDict(extra='forbid')
    credit_policy_met: bool
    interest_rate_percent: float = Field(ge=0, le=100)
    monthly_installment: float = Field(ge=0, le=1000000)
    annual_income: float = Field(gt=0, le=1000000000)
    debt_to_income_percent: float = Field(ge=0, le=100)
    fico_score: int = Field(ge=300, le=850)
    credit_history_years: float = Field(ge=0, le=80)
    revolving_balance: float = Field(ge=0, le=1000000000)
    revolving_utilization_percent: float = Field(ge=0, le=150)
    inquiries_last_six_months: int = Field(ge=0, le=100)
    delinquencies_last_two_years: int = Field(ge=0, le=100)
    public_records: int = Field(ge=0, le=100)
    purpose: Literal['all_other', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'major_purchase', 'small_business']

@lru_cache(maxsize=1)
def load_model():
    # Joblib is pickle based. Load only the version-controlled local artifact, never uploads.
    model, names = joblib.load(MODEL_PATH)
    if tuple(names) != EXPECTED or getattr(model, 'n_features_in_', None) != len(EXPECTED) or list(model.classes_) != [0, 1]:
        raise RuntimeError('The bundled model schema does not match the scoring API')
    return model, names

app = FastAPI(title='Loan Portfolio Lab API', version='1.0.0', docs_url='/docs')
origins = [x.strip() for x in os.getenv('FRONTEND_ORIGIN', 'http://localhost:5500').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

@app.get('/health')
def health():
    load_model()
    return {'status': 'ok', 'model': 'loaded'}

@app.get('/api/model-info')
def model_info():
    model, names = load_model()
    return {'model': type(model).__name__, 'target': 'not fully paid in the historical dataset', 'feature_count': len(names), 'training_data_available': False, 'validated_for_lending': False, 'stores_scenarios': False}

@app.post('/api/score')
def score(body: Scenario):
    model, names = load_model()
    values = {'credit.policy': int(body.credit_policy_met), 'int.rate': body.interest_rate_percent / 100,
              'installment': body.monthly_installment, 'log.annual.inc': math.log(body.annual_income),
              'dti': body.debt_to_income_percent, 'fico': body.fico_score,
              'days.with.cr.line': body.credit_history_years * 365.25,
              'revol.bal': body.revolving_balance, 'revol.util': body.revolving_utilization_percent,
              'inq.last.6mths': body.inquiries_last_six_months,
              'delinq.2yrs': body.delinquencies_last_two_years, 'pub.rec': body.public_records}
    values.update({f'purpose_{purpose}': int(body.purpose == purpose) for purpose in PURPOSES})
    frame = pd.DataFrame([[values[name] for name in names]], columns=names)
    try:
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(503, 'Model scoring is temporarily unavailable') from exc
    if not math.isfinite(probability):
        raise HTTPException(503, 'Model produced an invalid score')
    return {'not_fully_paid_score': round(probability, 4), 'target': 'historical not fully paid label',
            'interpretation': 'Exploratory model output, not a calibrated default probability or lending decision.',
            'inputs_used': len(names), 'stored': False}
