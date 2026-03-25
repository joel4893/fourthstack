"""
Talon — Frontend
Upload real transaction CSV → get synthetic data + fidelity report
"""

import streamlit as st
import pandas as pd
import requests
import time
import os

# Resolve API URL from environment (set TALON_API_URL in deployment/secrets)
API_URL = os.environ.get("TALON_API_URL", "https://talon-api-uvs9.onrender.com")

st.set_page_config(
    page_title="Talon",
    page_icon="⬡",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⬡ Talon")
st.caption("High-fidelity synthetic financial data. Zero PII.")
st.divider()

if st.button("Test API connectivity"):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        st.success(f"/health returned {resp.status_code}")
        try:
            st.json(resp.json())
        except Exception:
            st.text(resp.text)
    except Exception as e:
        st.error(f"Connectivity test failed: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Upload your real transaction CSV
    2. Talon trains on your data
    3. Generates statistically identical synthetic data
    4. Zero real records in the output — guaranteed

    **Required columns:**
    - `transaction_id`
    - `amount`
    - `merchant_category`
    - `transaction_hour`
    - `is_fraud`
    - `customer_age`
    - `account_balance`
    """)
    st.divider()
    st.caption("Talon — Built with SDV + CTGAN")

# ── Sample download ───────────────────────────────────────────────────────────
st.subheader("Don't have a CSV?")
if st.button("Download sample transaction data"):
    try:
        r = requests.get(f"{API_URL}/sample", timeout=30)
        st.download_button(
            label="Save sample_transactions.csv",
            data=r.content,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )
    except Exception:
        st.error("API not reachable. Try again in 120 seconds.")

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
st.subheader("Upload your data")
uploaded = st.file_uploader(
    "Drop your transaction CSV here",
    type=["csv"]
)

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df):,} rows")

    with st.expander("Preview"):
        st.dataframe(df.head(5), width="stretch")

    col1, col2 = st.columns(2)
    col1.metric("Total rows", f"{len(df):,}")
    fraud_pct = df['is_fraud'].mean() * 100 if 'is_fraud' in df.columns else 0
    col2.metric("Fraud rate", f"{fraud_pct:.2f}%")

    st.divider()
    st.subheader("Generate synthetic data")

    n_rows = st.slider(
        "How many synthetic rows?",
        min_value=100,
        max_value=max(1000, len(df)),
        value=len(df),
        step=50
    )

    if st.button("Generate"):

        # ── Wake up API first ─────────────────────────────────────────────────
        with st.spinner("Waking up API... (first request takes ~30s on free tier)"):
            try:
                wake = requests.get(f"{API_URL}/health", timeout=60)
                if wake.status_code != 200:
                    st.error("API not responding. Try again.")
                    st.stop()
            except Exception:
                st.error("API unreachable. Try again in 30 seconds.")
                st.stop()

        # ── Submit job ────────────────────────────────────────────────────────
        try:
            uploaded.seek(0)
            submit = requests.post(
                f"{API_URL}/synthesize",
                files={"file": ("data.csv", uploaded, "text/csv")},
                params={"n_rows": n_rows},
                timeout=180
            )

            if submit.status_code != 200:
                st.error(f"Submission failed: {submit.json()}")
                st.stop()

            job_id = submit.json()['job_id']
            st.info(f"Job submitted — ID: `{job_id}`")

        except Exception as e:
            st.error(f"Could not submit job: {str(e)}")
            st.stop()

        # ── Poll for completion ───────────────────────────────────────────────
        progress_bar = st.progress(0)
        status_text  = st.empty()
        elapsed_text = st.empty()

        start     = time.time()
        max_wait  = 600
        poll_interval = 5
        fake_progress = 0

        # Robust polling with retries + exponential backoff for transient errors
        retry_count = 0
        max_retries = 6
        backoff_base = 2
        max_backoff = 30

        while True:
            elapsed = int(time.time() - start)

            try:
                poll = requests.get(
                    f"{API_URL}/status/{job_id}",
                    timeout=60
                )

                if poll.status_code != 200:
                    # Non-OK responses should be visible to the user
                    if poll.status_code == 502:
                        status_text.warning("Server busy or restarting (HTTP 502). Retrying...")
                    elif poll.status_code == 404:
                        # Job lost (likely due to server restart clearing non-persistent DB)
                        st.error("Job not found. The server may have restarted. Please resubmit.")
                        st.stop()
                    else:
                        status_text.warning(f"Polling: HTTP {poll.status_code} — {poll.text[:200]}")

                    retry_count += 1
                    sleep_time = min(backoff_base ** retry_count, max_backoff)
                    time.sleep(sleep_time)
                    if retry_count > max_retries:
                        st.error("Polling failed repeatedly — try again later.")
                        st.stop()
                    continue

                poll_data = poll.json()
                status    = poll_data.get('status')
                retry_count = 0

            except Exception as e:
                # Network error / timeout — exponential backoff
                retry_count += 1
                sleep_time = min(backoff_base ** retry_count, max_backoff)
                status_text.warning(f"Polling error ({e.__class__.__name__}): {str(e)[:200]} — retrying in {sleep_time}s")
                if retry_count > max_retries:
                    st.error("Unable to reach API after several attempts. Try again later.")
                    st.stop()
                time.sleep(sleep_time)
                continue

            # Update UI
            fake_progress = min(fake_progress + 3, 90)
            progress_bar.progress(fake_progress)
            status_text.markdown(f"**Status:** `{status}` — training model...")
            elapsed_text.caption(f"Elapsed: {elapsed}s")

            if status == 'done':
                progress_bar.progress(100)
                status_text.success("Done!")
                elapsed_text.caption(f"Completed in {elapsed}s")

                fidelity = poll_data['fidelity']
                preview  = poll_data['preview']
                break

            elif status == 'failed':
                st.error(f"Synthesis failed: {poll_data.get('error')}")
                st.stop()

            elif elapsed > max_wait:
                st.error("Timed out after 10 minutes. Try a smaller dataset.")
                st.stop()

            time.sleep(poll_interval)

        # ── Show results ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Fidelity report")

        score = fidelity['overall_score']
        color = "green" if score >= 80 else "orange" if score >= 60 else "red"
        st.markdown(f"### Overall score: :{color}[{score}/100]")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud rate (real)",      f"{fidelity['fraud_rate_real']}%")
        c2.metric("Fraud rate (synthetic)", f"{fidelity['fraud_rate_synthetic']}%",
                  delta=f"{fidelity['fraud_rate_error_pct']}% error")
        c3.metric("Privacy leaks",          fidelity['privacy_leaks'],
                  delta="safe" if fidelity['privacy_safe'] else "risk",
                  delta_color="inverse")
        c4.metric("Synthetic rows",         f"{fidelity['synthetic_rows']:,}")

        st.divider()
        st.subheader("Distribution fidelity")
        st.caption("KS score: lower = more faithful to real data")

        for label, key in [
            ("Transaction amount", "amount_ks"),
            ("Transaction hour",   "hour_ks"),
            ("Customer age",       "age_ks"),
            ("Account balance",    "balance_ks"),
        ]:
            ks  = fidelity[key]
            pct = max(0, 1 - ks)
            st.markdown(f"**{label}** — KS: {ks}")
            st.progress(pct)

        st.divider()
        st.subheader("Synthetic data preview")
        st.caption("First 10 rows — zero real records present")
        st.dataframe(pd.DataFrame(preview), width="stretch")

        st.divider()
        st.subheader("Download")

        try:
            r_csv = requests.get(
                f"{API_URL}/result/{job_id}",
                timeout=30
            )
            st.download_button(
                label="Download synthetic_transactions.csv",
                data=r_csv.content,
                file_name="synthetic_transactions.csv",
                mime="text/csv",
                width="stretch"
            )
        except Exception:
            st.warning("Refresh to download — result is ready on the server.")