"""
Talon API — Async version
POST /synthesize     → submit job, get job_id immediately
GET  /status/{id}   → poll job status
GET  /result/{id}   → download result when ready
GET  /health        → status check
GET  /sample        → sample CSV download
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import sys
import os
import uuid
import threading
import sqlite3
import json

def keep_alive():
    """Ping own health endpoint every 5 mins to prevent Render spin-down."""
    import time
    import urllib.request
    while True:
        time.sleep(270)
        try:
            urllib.request.urlopen(
                "https://talon-api-uvs9.onrender.com/health",
                timeout=10
            )
        except Exception:
            pass

threading.Thread(target=keep_alive, daemon=True).start()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Talon",
    description="High-fidelity synthetic financial data. Zero PII.",
    version="0.2.0"
)

# ── Persistent job store (SQLite) ─────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "jobs.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                csv TEXT,
                fidelity TEXT,
                preview TEXT,
                error TEXT
            )
        """)

init_db()

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    with get_db() as conn:
        active_count = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'").fetchone()[0]
    return {
        "status": "ok",
        "version": "0.2.0",
        "active_jobs": active_count
    }

# ── Sample CSV ────────────────────────────────────────────────────────────────
@app.get("/sample")
def sample():
    import numpy as np
    np.random.seed(99)
    n = 200

    def gen_hours(n):
        hours = []
        for _ in range(n):
            c = np.random.choice(
                ['midnight','lunch','evening','other'],
                p=[0.25,0.20,0.30,0.25]
            )
            if c == 'midnight': h = int(np.clip(np.random.normal(1,1),0,3))  # noqa: E701
            elif c == 'lunch':  h = int(np.clip(np.random.normal(12,1),10,14))  # noqa: E701
            elif c == 'evening':h = int(np.clip(np.random.normal(19,1.5),16,23))  # noqa: E701
            else:               h = np.random.randint(0,24)  # noqa: E701
            hours.append(h)
        return hours

    df = pd.DataFrame({
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
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"}
    )

# ── Background synthesis worker ───────────────────────────────────────────────
def run_synthesis(job_id: str, df: pd.DataFrame, n_rows: int):
    """Runs in background thread. Updates jobs dict when done."""
    try:
        from core.synthesizer import synthesize
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
            conn.commit()

        result = synthesize(df, n_rows=n_rows)

        if result['success']:
            # Store CSV as string
            stream = io.StringIO()
            result['synthetic'].to_csv(stream, index=False)

            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'done', csv = ?, fidelity = ?, preview = ? WHERE job_id = ?",
                    (stream.getvalue(), json.dumps(result['fidelity']),
                     json.dumps(result['synthetic'].head(10).to_dict(orient='records')), job_id)
                )
                conn.commit()
        else:
            with get_db() as conn:
                conn.execute("UPDATE jobs SET status = 'failed', error = ? WHERE job_id = ?",
                             (json.dumps(result['errors']), job_id))
                conn.commit()
    except Exception as e:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status = 'failed', error = ? WHERE job_id = ?",
                         (json.dumps([str(e)]), job_id))
            conn.commit()

# ── Submit job ────────────────────────────────────────────────────────────────
@app.post("/synthesize")
async def submit_job(
    file: UploadFile = File(...),
    n_rows: int = None
):
    """
    Submit synthesis job. Returns job_id immediately.
    Poll /status/{job_id} to check progress.
    Download from /result/{job_id} when done.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    from core.synthesizer import validate_dataframe
    validation = validate_dataframe(df)
    if not validation['valid']:
        raise HTTPException(status_code=422, detail={
            "message": "Validation failed",
            "errors":  validation['errors']
        })

    # Create job
    job_id = str(uuid.uuid4())[:8]
    with get_db() as conn:
        conn.execute("INSERT INTO jobs (job_id, status) VALUES (?, ?)", (job_id, 'queued'))
        conn.commit()

    # Run in background thread so request returns immediately
    n = n_rows or len(df)
    thread = threading.Thread(
        target=run_synthesis,
        args=(job_id, df, n),
        daemon=True
    )
    thread.start()

    return {
        "job_id":     job_id,
        "status":     "queued",
        "rows":       len(df),
        "message":    "Job submitted. Poll /status/{job_id} for updates."
    }

# ── Poll status ───────────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
def job_status(job_id: str):
    with get_db() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job_id,
        "status": job['status']
    }

    if job['status'] == 'done':
        response['fidelity'] = json.loads(job['fidelity'])
        response['preview']  = json.loads(job['preview'])

    if job['status'] == 'failed':
        response['error'] = json.loads(job['error'])

    return response

# ── Download result ───────────────────────────────────────────────────────────
@app.get("/result/{job_id}")
def job_result(job_id: str):
    with get_db() as conn:
        job = conn.execute("SELECT status, csv FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job['status'] != 'done':
        raise HTTPException(
            status_code=425,
            detail=f"Job not ready. Status: {job['status']}"
        )

    return StreamingResponse(
        iter([job['csv']]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=synthetic_transactions.csv"
        }
    )