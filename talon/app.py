"""
Talon — Frontend
Upload real transaction CSV → get synthetic data + fidelity report
"""

import streamlit as st
import pandas as pd
import requests
import time

API_URL = "https://talon-api-uvs9.onrender.com"

st.set_page_config(
    page_title="Talon",
    page_icon="⬡",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⬡ Talon")
st.caption("High-fidelity synthetic financial data. Zero PII.")
st.divider()

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

    if st.button("Generate", type="primary", width="stretch"):

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
                timeout=120
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

        while True:
            elapsed = int(time.time() - start)

            try:
                poll = requests.get(
                    f"{API_URL}/status/{job_id}",
                    timeout=10
                )
                poll_data = poll.json()
                status    = poll_data['status']

            except Exception:
                status_text.warning("Polling... API temporarily unreachable")
                time.sleep(poll_interval)
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