"""
Talon API — Async version
POST /synthesize     → submit job, get job_id immediately
GET  /status/{id}   → poll job status
GET  /result/{id}   → download result when ready
GET  /health        → status check
GET  /sample        → sample CSV download
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.oauth2 import id_token
from google.auth.transport import requests as google_auth_requests
import urllib.request
import pandas as pd
import numpy as np
import io
import sys
import os
import uuid
import threading
import sqlite3
# In a production-ready version, we'd replace the local thread/sqlite
# with Celery + Redis and PostgreSQL to handle horizontal scaling.
import json
import torch
import tempfile
import time
import resource
import gc
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.synthesizer import synthesize, validate_dataframe

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

def keep_alive():
    """Ping own health endpoint to prevent Render spin-down."""
    # Use the environment URL if available, fallback to hardcoded only as last resort
    url = os.environ.get("TALON_API_URL", "https://talon-api-uvs9.onrender.com") + "/health"
    while True:
        time.sleep(270)
        try:
            print(f"[*] Keep-alive: Pinging {url}", file=sys.stderr)
            urllib.request.urlopen(url, timeout=10)
        except Exception:
            pass

# Only start keep_alive if we're on Render Free Tier.
# Paid plans (Starter/Standard) don't need this as they stay awake.
# You can set RENDER_INSTANCE_TYPE in your Render env vars.
if os.environ.get("RENDER") and os.environ.get("RENDER_INSTANCE_TYPE", "free") == "free":
    threading.Thread(target=keep_alive, daemon=True).start()

app = FastAPI(
    title="Talon",
    description="High-fidelity synthetic financial data. Zero PII.",
    version="0.2.0"
)

# ── Observability Middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def log_request_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log time per request and failure status
    log_msg = f"[*] {request.method} {request.url.path} | Status: {response.status_code} | Time: {process_time:.4f}s"
    print(log_msg, file=sys.stderr)
    
    return response

# ── Persistent job store (SQLite) ─────────────────────────────────────────────
DB_PATH = os.path.join(tempfile.gettempdir(), "jobs.db")

def get_db():
    # Add 20s timeout to wait for locks before failing
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        with get_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
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
            # Migration: Add columns for queuing if they don't exist
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN input_csv TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN n_rows INTEGER")
            except sqlite3.OperationalError:
                pass
                
            # Robust User table setup
            conn.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, picture TEXT, last_login DATETIME DEFAULT CURRENT_TIMESTAMP)")
            
            # Ensure last_login index for analytics performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_login ON users(last_login)")

            # Analytics and Feedback tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    email TEXT,
                    message TEXT
                )
            """)

        print(f"Database initialized at {DB_PATH}", file=sys.stderr)
    except Exception as e:
        print(f"FATAL: Database initialization failed: {e}", file=sys.stderr)
        raise e

init_db()

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Welcome to the Talon API",
        "docs": "/docs",
        "health": "/health"
    }

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

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/auth/google")
async def auth_google(request: Request):
    data = await request.json()
    token = data.get("token")

    if not GOOGLE_CLIENT_ID:
        print("[!] Auth Error: GOOGLE_CLIENT_ID environment variable is not set", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Server configuration error")

    try:
        # Verify the Google Token
        idinfo = id_token.verify_oauth2_token(token, google_auth_requests.Request(), GOOGLE_CLIENT_ID, clock_skew_in_seconds=10)

        email = idinfo['email']
        name = idinfo.get('name')
        picture = idinfo.get('picture')

        with get_db() as conn:
            conn.execute("""
                INSERT INTO users (email, name, picture, last_login) 
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(email) DO UPDATE SET last_login=CURRENT_TIMESTAMP, name=?, picture=?
            """, (email, name, picture, name, picture))
            conn.commit()

        return {"status": "success", "user": {"email": email, "name": name, "picture": picture}}
    except Exception as e:
        print(f"[!] Auth Error: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=401, detail="Invalid Google Token")

# ── Analytics & Feedback ──────────────────────────────────────────────────────
@app.post("/visit")
def record_visit(request: Request):
    user_agent = request.headers.get("user-agent")
    with get_db() as conn:
        conn.execute("INSERT INTO visits (user_agent) VALUES (?)", (user_agent,))
        conn.commit()
    return {"status": "recorded"}

@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    email = data.get("email")
    message = data.get("message")
    with get_db() as conn:
        conn.execute("INSERT INTO feedback (email, message) VALUES (?, ?)", (email, message))
        conn.commit()
    return {"status": "thank you"}

@app.get("/analytics")
def get_analytics():
    """Internal endpoint to check usage."""
    with get_db() as conn:
        visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        users_list = conn.execute("SELECT email, name, last_login FROM users ORDER BY last_login DESC").fetchall()
        feedback = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
        
    return {
        "total_visits": visits,
        "total_jobs_submitted": jobs,
        "registered_users": [dict(u) for u in users_list],
        "feedback_entries": [dict(f) for f in feedback]
    }

# ── Sample CSV ────────────────────────────────────────────────────────────────
@app.get("/sample")
def sample():
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
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
            conn.commit()

        result = synthesize(df, n_rows=n_rows)
        gc.collect()

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
        print(f"[!] Synthesis Failure at point: {traceback.format_exc()}", file=sys.stderr)
    finally:
        # Ensure input CSV is cleared even on failure to save space
        with get_db() as conn:
            conn.execute("UPDATE jobs SET input_csv = NULL WHERE job_id = ?", (job_id,))
            conn.commit()

# ── Queue Worker ──────────────────────────────────────────────────────────────
def worker_loop():
    """Continuously checks DB for queued jobs and runs them sequentially."""
    while True:
        try:
            job = None
            with get_db() as conn:
                job = conn.execute(
                    "SELECT job_id, input_csv, n_rows FROM jobs WHERE status = 'queued' ORDER BY rowid ASC LIMIT 1"
                ).fetchone()

            # Global torch throttle
            torch.set_num_threads(1)

            if job:
                # Found a job, process it
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"[*] Worker: Processing job {job['job_id']} (System RAM: {mem_usage:.2f}MB)", file=sys.stderr)
                job_id = job['job_id']
                n_rows = job['n_rows']
                csv_str = job['input_csv']

                df = pd.read_csv(io.StringIO(csv_str))
                
                # Aggressively free memory before training starts
                del csv_str
                del job
                gc.collect()

                run_synthesis(job_id, df, n_rows)
                
            else:
                time.sleep(2) # No jobs, wait before polling again
        except Exception as e:
            print(f"Worker error: {e}", file=sys.stderr)
            time.sleep(5)

threading.Thread(target=worker_loop, daemon=True, name="queue_worker").start()

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

    validation = validate_dataframe(df)
    if not validation['valid']:
        raise HTTPException(status_code=422, detail={
            "message": "Validation failed",
            "errors":  validation['errors']
        })

    # Create job
    job_id = str(uuid.uuid4())[:8]
    
    # Convert DF back to CSV string for storage
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    input_csv_str = stream.getvalue()

    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, input_csv, n_rows) VALUES (?, ?, ?, ?)", 
            (job_id, 'queued', input_csv_str, n_rows or len(df))
        )
        conn.commit()

    # Worker loop will pick it up automatically

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
        # Select only necessary columns to avoid reading heavy CSV blob
        job = conn.execute(
            "SELECT job_id, status, fidelity, preview, error FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()

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