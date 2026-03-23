"""
Talon — Frontend
Upload real transaction CSV → get synthetic data + fidelity report
"""

import streamlit as st
import pandas as pd
import requests
import time
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talon",
    page_icon="⬡",
    layout="centered"
)

# ── API URL — switch to Render URL once deployed ──────────────────────────────
# Prefer an environment variable so deployments can override the default URL.
API_URL = os.environ.get(
    "TALON_API_URL",
    "https://talon-api-uvs9.onrender.com"
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
    2. Talon trains on your data locally
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
    st.caption("Built with SDV + CTGAN")

# ── Sample download ───────────────────────────────────────────────────────────
st.subheader("Don't have a CSV?")
if st.button("Download sample transaction data"):
    try:
        r = requests.get(f"{API_URL}/sample")
        st.download_button(
            label="Save sample_transactions.csv",
            data=r.content,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )
    except Exception:
        st.error("API not reachable. Is it running?")

st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
st.subheader("Upload your data")
uploaded = st.file_uploader(
    "Drop your transaction CSV here",
    type=["csv"],
    help="Must contain the required columns listed in the sidebar"
)

if uploaded:
    # Show preview of uploaded data
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    with st.expander("Preview your data"):
        st.dataframe(df.head(5), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total rows", f"{len(df):,}")
    with col2:
        fraud_pct = df['is_fraud'].mean() * 100 if 'is_fraud' in df.columns else 0
        st.metric("Fraud rate", f"{fraud_pct:.2f}%")

    st.divider()

    # ── Synthesize button ─────────────────────────────────────────────────────
    st.subheader("Generate synthetic data")
    n_rows = st.slider(
        "How many synthetic rows?",
        min_value=100,
        max_value=max(1000, len(df)),
        value=len(df),
        step=50,
        help="Defaults to same size as your input"
    )

    if st.button("Generate", use_container_width=True):
        with st.spinner("Training synthesis model... (~4 minutes)"):
            start = time.time()

            try:
                # Reset file pointer
                uploaded.seek(0)

                # Hit the JSON endpoint for fidelity scores
                r = requests.post(
                    f"{API_URL}/synthesize/json",
                    files={"file": ("data.csv", uploaded, "text/csv")},
                    params={"n_rows": n_rows},
                    timeout=600
                )

                elapsed = round(time.time() - start, 1)

                if r.status_code == 200:
                    result = r.json()
                    fidelity = result['fidelity']
                    synthetic_df = pd.DataFrame(result['preview_rows'])

                    st.success(f"Done in {elapsed}s")
                    st.divider()

                    # ── Fidelity report ───────────────────────────────────────
                    st.subheader("Fidelity report")

                    # Overall score — big and prominent
                    score = fidelity['overall_score']
                    color = (
                        "green" if score >= 80
                        else "orange" if score >= 60
                        else "red"
                    )
                    st.markdown(
                        f"### Overall score: "
                        f":{color}[{score}/100]"
                    )

                    # Key metrics in columns
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Fraud rate (real)",
                        f"{fidelity['fraud_rate_real']}%"
                    )
                    c2.metric(
                        "Fraud rate (synthetic)",
                        f"{fidelity['fraud_rate_synthetic']}%",
                        delta=f"{fidelity['fraud_rate_error_pct']}% error"
                    )
                    c3.metric(
                        "Privacy leaks",
                        fidelity['privacy_leaks'],
                        delta="safe" if fidelity['privacy_safe'] else "risk",
                        delta_color="inverse"
                    )
                    c4.metric(
                        "Synthetic rows",
                        f"{fidelity['synthetic_rows']:,}"
                    )

                    st.divider()

                    # KS scores as progress bars
                    st.subheader("Distribution fidelity")
                    st.caption(
                        "KS score measures how similar real vs synthetic "
                        "distributions are. Lower = more faithful."
                    )

                    ks_metrics = {
                        "Transaction amount": fidelity['amount_ks'],
                        "Transaction hour":   fidelity['hour_ks'],
                        "Customer age":       fidelity['age_ks'],
                        "Account balance":    fidelity['balance_ks'],
                    }

                    for label, ks in ks_metrics.items():
                        fidelity_pct = max(0, 1 - ks)
                        st.markdown(f"**{label}** — KS: {ks}")
                        st.progress(fidelity_pct)

                    st.divider()

                    # ── Synthetic data preview ────────────────────────────────
                    st.subheader("Synthetic data preview")
                    st.caption("First 10 rows — no real records present")
                    st.dataframe(synthetic_df, use_container_width=True)

                    st.divider()

                    # ── Download full synthetic CSV ────────────────────────────
                    st.subheader("Download full dataset")

                    # Hit the CSV endpoint for the full file
                    uploaded.seek(0)
                    r_csv = requests.post(
                        f"{API_URL}/synthesize",
                        files={"file": ("data.csv", uploaded, "text/csv")},
                        params={"n_rows": n_rows},
                        timeout=600
                    )

                    if r_csv.status_code == 200:
                        st.download_button(
                            label="Download synthetic_transactions.csv",
                            data=r_csv.content,
                            file_name="synthetic_transactions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "Preview generated. "
                            "Refresh to download full CSV."
                        )

                else:
                    detail = r.json().get('detail', 'Unknown error')
                    st.error(f"API error {r.status_code}: {detail}")

            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out. "
                    "Try a smaller dataset or increase timeout."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach API. "
                    "Make sure it's running on port 8000."
                )
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")