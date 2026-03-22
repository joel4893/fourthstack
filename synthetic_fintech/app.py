import streamlit as st
import pandas as pd
from sdv.metadata import SingleTableMetadata
import matplotlib
matplotlib.use('Agg')

from synthesizer import (
    FintechSynthesizer,
    generate_demo_data,
    compare_data,
    plot_comparison
)

st.set_page_config(layout="wide")

st.title("YC-Synth: A Demonstrator for High-Fidelity Synthetic Financial Data")
st.write(
    "This app demonstrates a sophisticated method for generating synthetic data. "
    "It's designed to handle the nuances of real-world financial data, like "
    "heavy-tailed value distributions and rare event modeling (e.g., fraud)."
)

st.header("Step 1: The Core Engine")
st.info(
    "Our advantage lies in a two-part strategy:
"
    "1.  **Quantile Transformation**: We preprocess heavy-tailed distributions (like transaction amounts) to a normal distribution, which is much easier for a GAN to learn.
"
    "2.  **Stratified Training**: We train separate models on fraudulent and legitimate transactions to ensure the rare fraud cases are learned correctly, then re-blend them at the real-world rate."
)


if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
    st.session_state.real_data = None
    st.session_state.metrics = None


def run_synthesis():
    """The main function to run the data synthesis process."""
    with st.spinner('Step 1/5: Generating real-world demo data...'):
        real_data = generate_demo_data(n=2000)
        st.session_state.real_data = real_data

    with st.spinner('Step 2/5: Defining data metadata...'):
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(data=real_data)
        meta.update_column(column_name='transaction_id', sdtype='id')

    with st.spinner('Step 3/5: Fitting the synthesizer... (This is the slow part)'):
        # In a real app, you'd run this async or on a backend
        synth = FintechSynthesizer(metadata=meta, epochs=150)
        synth.fit(real_data)

    with st.spinner('Step 4/5: Sampling new synthetic data...'):
        synthetic_data = synth.sample(num_rows=2000)
        st.session_state.synthetic_data = synthetic_data
    
    with st.spinner('Step 5/5: Evaluating data quality...'):
        metrics = compare_data(real_data, synthetic_data)
        st.session_state.metrics = metrics


st.header("Step 2: Run the Demo")

if st.button("▶️ Generate Synthetic Data (takes ~2-3 mins)", type="primary"):
    run_synthesis()
    st.success("Synthesis complete! See the results below.")


if st.session_state.synthetic_data is not None:
    st.header("Step 3: Evaluate the Results")

    st.subheader("Visual Comparison")
    
    # Generate and save the plot
    plot_path = 'comparison.png'
    plot_comparison(
        st.session_state.real_data,
        st.session_state.synthetic_data,
        st.session_state.metrics,
        filepath=plot_path
    )
    st.image(plot_path, caption="Distribution Comparison: Real vs. Synthetic")
    st.markdown(
        "**Key takeaway**: The distributions of the synthetic data (purple) closely "
        "match the shape of the real data (blue). The KS (Kolmogorov-Smirnov) score "
        "quantifies this similarity, with a lower score being better."
    )

    st.subheader("Data Preview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(st.session_state.real_data.head(10), use_container_width=True)
        st.caption("Real Data (first 10 rows)")
    with col2:
        st.dataframe(st.session_state.synthetic_data.head(10), use_container_width=True)
        st.caption("Synthetic Data (first 10 rows)")

    st.subheader("Next Steps")
    st.markdown(
        "This demo shows the core capability. The next steps on our roadmap are:
"
        "- Allow users to **upload their own datasets**.
"
        "- Provide a simple **API endpoint** for developers.
"
        "- Add more advanced privacy and utility controls."
    )
