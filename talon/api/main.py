"""
Parity API
POST /synthesize  — upload CSV, get synthetic CSV + fidelity scores
GET  /health      — confirm API is running
GET  /sample      — download a sample input CSV to test with
"""

import io
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse # type: ignore

# Make core package importable when running inside the repo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.synthesizer import synthesize, validate_dataframe

app = FastAPI(
    title="Parity",
    description="High-fidelity synthetic financial data. Zero PII.",
    version="0.1.0"
)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# ── Sample CSV download ───────────────────────────────────────────────────────
@app.get("/sample")
def sample():
    """Download a sample input CSV to test the /synthesize endpoint."""

    np.random.seed(99)
    n = 200

    def gen_hours(n):
        hours = []
        for _ in range(n):
            c = np.random.choice(
                ['midnight', 'lunch', 'evening', 'other'],
                p=[0.25, 0.20, 0.30, 0.25]
            )
            if c == 'midnight':
                h = int(np.clip(np.random.normal(1, 1), 0, 3))
            elif c == 'lunch':
                h = int(np.clip(np.random.normal(12, 1), 10, 14))
            elif c == 'evening':
                h = int(np.clip(np.random.normal(19, 1.5), 16, 23))
            else:
                h = np.random.randint(0, 24)
            hours.append(h)
        return hours

    sample_df = pd.DataFrame({
        'transaction_id':    range(1, n+1),
        'amount':            np.round(np.random.lognormal(3.5, 1.2, n), 2),
        'merchant_category': np.random.choice(
            ['groceries','transport','dining','entertainment','utilities'],
            size=n, p=[0.30,0.25,0.20,0.15,0.10]
        ),
        'transaction_hour':  gen_hours(n),
        'is_fraud':          np.random.choice([0,1], size=n, p=[0.97,0.03]),
        'customer_age':      np.random.randint(18, 75, n),
        'account_balance':   np.round(np.random.lognormal(7, 1, n), 2)
    })

    stream = io.StringIO()
    sample_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"}
    )

# ── Main synthesis endpoint ───────────────────────────────────────────────────
@app.post("/synthesize")
async def synthesize_endpoint(
    file: UploadFile = File(...),
    n_rows: Optional[int] = None
):
    """
    Upload a CSV of real transaction data.
    Returns synthetic CSV + fidelity scores.
    
    Required columns:
    transaction_id, amount, merchant_category,
    transaction_hour, is_fraud, customer_age, account_balance
    """

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV. Got: " + file.filename
        )

    # Read uploaded CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )

    # Validate dataframe
    validation = validate_dataframe(df)
    if not validation['valid']:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Data validation failed",
                "errors":  validation['errors']
            }
        )

    # Run synthesis pipeline
    result = synthesize(df, n_rows=n_rows)

    if not result['success']:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Synthesis failed",
                "errors":  result['errors']
            }
        )

    # Convert synthetic dataframe to CSV stream
    stream = io.StringIO()
    result['synthetic'].to_csv(stream, index=False)
    stream.seek(0)

    # Return synthetic CSV as download
    # Fidelity scores go in response headers so client can read them
    headers = {
        "Content-Disposition": "attachment; filename=synthetic_transactions.csv",
        "X-Parity-Score":          str(result['fidelity']['overall_score']),
        "X-Fraud-Rate-Real":       str(result['fidelity']['fraud_rate_real']),
        "X-Fraud-Rate-Synthetic":  str(result['fidelity']['fraud_rate_synthetic']),
        "X-Fraud-Error-Pct":       str(result['fidelity']['fraud_rate_error_pct']),
        "X-Amount-KS":             str(result['fidelity']['amount_ks']),
        "X-Hour-KS":               str(result['fidelity']['hour_ks']),
        "X-Privacy-Safe":          str(result['fidelity']['privacy_safe']),
        "X-Privacy-Leaks":         str(result['fidelity']['privacy_leaks']),
        "Access-Control-Expose-Headers": "*"
    }

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers=headers
    )

# ── Synthesize and return JSON (alternative endpoint) ────────────────────────
@app.post("/synthesize/json")
async def synthesize_json_endpoint(
    file: UploadFile = File(...),
    n_rows: Optional[int] = None
):
    """Same as `/synthesize` but returns JSON with fidelity and synthetic rows."""
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    # Read uploaded CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    # Validate dataframe
    validation = validate_dataframe(df)
    if not validation['valid']:
        return JSONResponse(status_code=422, content={
            "success": False,
            "errors": validation['errors']
        })

    # Run synthesis pipeline
    result = synthesize(df, n_rows=n_rows)
    if not result['success']:
        return JSONResponse(status_code=500, content={
            "success": False,
            "errors": result.get('errors', [])
        })

    synthetic_records = result['synthetic'].to_dict(orient='records')

    return JSONResponse(status_code=200, content={
        "success": True,
        "synthetic": synthetic_records,
        "fidelity": result['fidelity'],
        "errors": []
    })