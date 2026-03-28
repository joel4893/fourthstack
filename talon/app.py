"""
Talon — Frontend
Upload real transaction CSV → get synthetic data + fidelity report
"""

import streamlit as st
import pandas as pd
import requests
import streamlit.components.v1 as components
import time
import os

# Resolve API URL from environment (set TALON_API_URL in deployment/secrets)
API_URL = os.environ.get("TALON_API_URL", "https://talon-api-uvs9.onrender.com")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

st.set_page_config(
    page_title="Talon",
    page_icon="⬡",
    layout="centered"
)

# ── Track Visit ───────────────────────────────────────────────────────────────
if 'visited' not in st.session_state:
    try:
        requests.post(f"{API_URL}/visit", timeout=5)
        st.session_state.visited = True
    except:
        pass

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⬡ Talon")
st.caption("High-fidelity synthetic financial data. Zero PII.")
st.divider()

# ── Login Logic ───────────────────────────────────────────────────────────────
def login_sidebar():
    if not GOOGLE_CLIENT_ID:
        st.sidebar.error("Google Client ID missing. Set GOOGLE_CLIENT_ID in environment variables.")
        return

    if 'user' not in st.session_state:
        st.subheader("🔑 Access Talon")
        # Google Sign-In HTML/JS snippet
        html_code = f"""
            <div id="g_id_onload"
                 data-client_id="{GOOGLE_CLIENT_ID}"
                 data-context="signin"
                 data-ux_mode="popup"
                 data-callback="handleCredentialResponse"
                 data-auto_prompt="false"
                 data-auto_select="false"
                 data-itp_support="true"
                 data-use_fedcm_for_prompt="true"
                 data-allowed_parent_origin="https://fourthstack-y2zay9tdvuqldbbrqkx3hm.streamlit.app">
            </div>
            <div class="g_id_signin" data-type="standard"></div>
            <script src="https://accounts.google.com/gsi/client" async defer></script>
            <script>
                function handleCredentialResponse(response) {{
                    try {{
                        // Safely detect the parent URL for redirection
                        const targetUrl = (window.location.ancestorOrigins && window.location.ancestorOrigins.length > 0) 
                                          ? window.location.ancestorOrigins[0] 
                                          : window.parent.location.href;
                        const url = new URL(targetUrl);
                        url.searchParams.set('token', response.credential);
                        
                        // Force the parent window to redirect with the token
                        window.open(url.toString(), "_top");
                    }} catch (e) {{
                        console.error("Redirection failed:", e);
                        // Fallback: Try a normal reload if parent access is strictly blocked
                        const fallbackUrl = new URL(window.location.href);
                        fallbackUrl.searchParams.set('token', response.credential);
                        window.location.href = fallbackUrl.toString();
                    }}
                }}
            </script>
        """
        components.html(html_code, height=70)

        # Logic to handle the token if passed via URL (simplified callback)
        token = st.query_params.get("token")
        if token:
            resp = requests.post(f"{API_URL}/auth/google", json={"token": token}, timeout=15)
            if resp.status_code == 200:
                st.session_state.user = resp.json()['user']
                st.query_params.clear()
                st.rerun()
    else:
        user = st.session_state.user
        st.sidebar.success(f"Logged in as {user['name']}")
        if st.sidebar.button("Logout"):
            del st.session_state.user
            st.rerun()

if st.button("Test API connectivity"):
    try:
        # Increased timeout to 60s to handle Render free-tier cold starts
        resp = requests.get(f"{API_URL}/health", timeout=60)
        st.success(f"/health returned {resp.status_code}")
        try:
            st.json(resp.json())
        except Exception:
            st.text(resp.text)
    except Exception as e:
        st.error(f"Connectivity test failed.")
        st.info(f"Make sure the backend is running at {API_URL}")
        with st.expander("Error details"):
            st.code(str(e))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    login_sidebar()
    st.divider()
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
    st.subheader("💬 Talk to the Founder")
    with st.form("feedback_form", clear_on_submit=True):
        default_email = st.session_state.user['email'] if 'user' in st.session_state else ""
        user_email = st.text_input("Email", value=default_email, placeholder="How can I reach you?")
        user_msg = st.text_area("Feedback", placeholder="Would you pay for this? What's missing?")
        submitted = st.form_submit_button("Send to Talon Team")
        
        if submitted:
            if user_msg:
                try:
                    requests.post(f"{API_URL}/feedback", 
                                  json={"email": user_email, "message": user_msg}, 
                                  timeout=5)
                    st.success("Thanks! I'll read this today.")
                except:
                    st.error("Couldn't send feedback. API down?")
            else:
                st.warning("Please enter a message.")

    st.divider()
    st.design_attr = st.toggle("Debug Mode", value=False)
    if st.design_attr:
        st.write(f"**Target API:** `{API_URL}`")
    st.divider()
    st.caption("Talon — Built with SDV + CTGAN")

# ── Sample download ───────────────────────────────────────────────────────────
st.subheader("Don't have a CSV?")
if st.button("Download sample transaction data"):
    try:
        r = requests.get(f"{API_URL}/sample", timeout=60)
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

    # Cap rows to 1000 to prevent OOM/Timeouts on free tier
    n_rows = st.slider(
        "How many synthetic rows?",
        min_value=100,
        max_value=1000,
        value=min(len(df), 1000),
        step=50
    )

    if st.button("Generate"):

        # ── Wake up API first ─────────────────────────────────────────────────
        with st.spinner("Waking up Talon Inference Engine..."):
            api_ready = False
            for _ in range(40):  # Retry for 200 seconds (40 * 5s)
                try:
                    resp = requests.get(f"{API_URL}/health", timeout=5)
                    if resp.status_code == 200:
                        api_ready = True
                        break
                except:
                    pass
                time.sleep(5)
            
            if not api_ready:
                st.error("API failed to wake up. Please check if the backend is running.")
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
        progress_bar = st.progress(10)
        status_text  = st.empty()
        elapsed_text = st.empty()

        start     = time.time()
        max_wait  = 900  # 15 minutes
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
                    timeout=20
                )

                if poll.status_code != 200:
                    if poll.status_code == 504:
                        status_text.warning("Gateway Timeout — Server is processing heavy model. Retrying...")
                        time.sleep(poll_interval)
                        continue
                    
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

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
                # Specific handling for timeouts during heavy synthesis
                status_text.warning("Server is under heavy load (Synthesis in progress)... retrying.")
                time.sleep(poll_interval)
                continue
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
            fake_progress = min(fake_progress + 20, 95)
            if status == 'queued':
                status_text.markdown(f"**Status:** `queued` — Preparing dataset...")
            else:
                progress_bar.progress(fake_progress)
                status_text.markdown(f"**Status:** `running` — Running Talon Inference...")
            
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
                st.error("Timed out after 15 minutes. The server is under extreme load.")
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
                timeout=60
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
                st.error("Result found but could not be downloaded. Please try again.")
        except Exception:
            st.warning("Refresh to download — result is ready on the server.")