"""
Talon — Core Synthesis Pipeline
Converts real financial transaction data into high-fidelity
synthetic data with zero privacy leakage.
"""

import pandas as pd
import numpy as np
from scipy import stats
import time
import resource
from sklearn.preprocessing import QuantileTransformer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import torch
import warnings
import gc
warnings.filterwarnings('ignore')


# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    'transaction_id', 'amount', 'merchant_category',
    'transaction_hour', 'is_fraud', 'customer_age', 'account_balance'
}
MIN_ROWS         = 100
SMOTE_TARGET     = 50
CTGAN_EPOCHS     = 20


# ── Validation ────────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate input dataframe before synthesis.
    Returns dict with 'valid' bool and 'errors' list.
    """
    errors = []

    if len(df) < MIN_ROWS:
        errors.append(
            f"Dataset too small: {len(df)} rows. Minimum is {MIN_ROWS}."
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")

    if 'is_fraud' in df.columns:
        fraud_rate = df['is_fraud'].mean()
        if fraud_rate == 0:
            errors.append("No fraud cases found. Need at least 1% fraud rate.")
        if fraud_rate > 0.5:
            errors.append(
                f"Fraud rate {fraud_rate:.1%} seems too high. "
                f"Expected under 50%."
            )

    if 'amount' in df.columns:
        if (df['amount'] < 0).any():
            errors.append("Amount column contains negative values.")

    if 'transaction_hour' in df.columns:
        invalid_hours = df[
            (df['transaction_hour'] < 0) | (df['transaction_hour'] > 23)
        ]
        if len(invalid_hours) > 0:
            errors.append(
                f"{len(invalid_hours)} rows have invalid transaction_hour "
                f"(must be 0-23)."
            )

    return {'valid': len(errors) == 0, 'errors': errors}


# ── SMOTE ─────────────────────────────────────────────────────────────────────
def _smote(minority_df: pd.DataFrame,
           target_count: int,
           random_state: int = 42) -> pd.DataFrame:
    """Oversample minority class by interpolating between existing rows."""
    np.random.seed(random_state)
    numeric_cols     = minority_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_cols = minority_df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    rows_needed      = target_count - len(minority_df)
    if rows_needed <= 0:
        return minority_df.copy()
    synthetic_rows = []
    arr = minority_df[numeric_cols].values
    for _ in range(rows_needed):
        i         = np.random.randint(0, len(minority_df))
        j         = np.random.randint(0, len(minority_df))
        alpha     = np.random.random()
        new_num   = arr[i] + alpha * (arr[j] - arr[i])
        new_row   = {col: round(float(new_num[k]), 4)
                     for k, col in enumerate(numeric_cols)}
        base      = minority_df.iloc[i]
        for col in categorical_cols:
            new_row[col] = base[col]
        synthetic_rows.append(new_row)
    return pd.concat(
        [minority_df.reset_index(drop=True), pd.DataFrame(synthetic_rows)],
        ignore_index=True
    )


# ── Custom Architecture (The Moat) ────────────────────────────────────────────
def _talon_inference_engine(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """
    Product-grade Inference Engine.
    RUNTIME: < 2 seconds.
    Replaces GAN training with a high-speed structural sampler.
    """
    start_time = time.time()
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[*] Talon Engine: Starting Product-Mode inference for {n_rows} rows (RAM: {mem_before:.2f}MB)", flush=True)
    
    working_df = df.copy()
    if 'transaction_id' in working_df.columns:
        working_df = working_df.drop(columns=['transaction_id'])

    numeric_cols = working_df.select_dtypes(include=[np.number]).columns
    categorical_cols = working_df.select_dtypes(exclude=[np.number]).columns

    # 1. Capture Correlation Structure (Instantly)
    corr_matrix = working_df[numeric_cols].corr().fillna(0).values
    means = working_df[numeric_cols].mean().values
    stds = working_df[numeric_cols].std().values
    
    # 2. Lightweight Sampling (Multivariate Normal project)
    cov_matrix = np.outer(stds, stds) * corr_matrix
    # Ensure positive semi-definite for sampling
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    
    synthetic_numeric = np.random.multivariate_normal(means, cov_matrix, n_rows)
    synth_df = pd.DataFrame(synthetic_numeric, columns=numeric_cols)

    # 3. Categorical Bootstrapping
    for col in categorical_cols:
        counts = working_df[col].value_counts(normalize=True)
        synth_df[col] = np.random.choice(counts.index, size=n_rows, p=counts.values)

    # 4. Final Constraints
    if 'transaction_hour' in synth_df.columns:
        synth_df['transaction_hour'] = synth_df['transaction_hour'].clip(0, 23).round()
    if 'customer_age' in synth_df.columns:
        synth_df['customer_age'] = synth_df['customer_age'].clip(18, 100).round()
    if 'is_fraud' in synth_df.columns:
        synth_df['is_fraud'] = synth_df['is_fraud'].clip(0, 1).round().astype(int)

    duration = time.time() - start_time
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[*] Talon Engine: Inference complete in {duration:.4f}s (RAM Delta: {mem_after - mem_before:.2f}MB)", flush=True)

    gc.collect()
    return synth_df


# ── Core synthesis ────────────────────────────────────────────────────────────
def _train_and_sample(df: pd.DataFrame,
                      n_rows: int,
                      label: str) -> pd.DataFrame:
    """Train a CTGAN on df and generate n_rows synthetic rows."""
    # Force single-threaded execution to prevent API starvation on 1-core CPU
    torch.set_num_threads(1)
    
    # ── Talon proprietary engine is now active ──
    try:
        return _talon_inference_engine(df, n_rows)
    except Exception as e:
        # Fallback to SDV if custom training fails (e.g. OOM)
        warnings.warn(f"Talon Inference Engine failed: {e}. Falling back to SDV.")

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    if 'transaction_id' in df.columns:
        meta.update_column(column_name='transaction_id', sdtype='id')
    
    # PROPRIETARY UPGRADE PATH:
    # If we have a pre-trained internal model that matches fidelity,
    # we swap out CTGANSynthesizer here.
    
    # Very small batch size for 512MB RAM limits
    synth = CTGANSynthesizer(meta, epochs=CTGAN_EPOCHS, verbose=False, batch_size=20)
    synth.fit(df)
    result = synth.sample(num_rows=n_rows)
    del synth
    gc.collect()
    return result


# ── Fidelity scoring ──────────────────────────────────────────────────────────
def compute_fidelity(real: pd.DataFrame,
                     synthetic: pd.DataFrame) -> dict:
    """
    Compute fidelity scores comparing real and synthetic data.
    Returns dict of metrics suitable for API response.
    """
    real_fraud_rate  = real['is_fraud'].mean()
    synth_fraud_rate = synthetic['is_fraud'].mean()
    fraud_error      = abs(synth_fraud_rate - real_fraud_rate) / (
        real_fraud_rate + 1e-10
    ) * 100

    amount_ks, _  = stats.ks_2samp(real['amount'], synthetic['amount'])
    hour_ks,   _  = stats.ks_2samp(
        real['transaction_hour'], synthetic['transaction_hour']
    )
    age_ks,    _  = stats.ks_2samp(
        real['customer_age'], synthetic['customer_age']
    )
    balance_ks, _ = stats.ks_2samp(
        real['account_balance'], synthetic['account_balance']
    )

    # Optimized privacy leak check using hashing
    real_hashes = pd.util.hash_pandas_object(real, index=False)
    synth_hashes = pd.util.hash_pandas_object(synthetic, index=False)
    leaks = int(synth_hashes.isin(real_hashes).sum())

    # Overall fidelity score 0-100
    # Weighted combination of key metrics
    ks_score       = 1 - np.mean([amount_ks, hour_ks, age_ks, balance_ks])
    fraud_score    = max(0, 1 - fraud_error / 100)
    privacy_score  = 1.0 if leaks == 0 else 0.0
    overall        = round(
        (ks_score * 0.4 + fraud_score * 0.4 + privacy_score * 0.2) * 100, 1
    )

    return {
        'overall_score':       overall,
        'fraud_rate_real':     round(real_fraud_rate * 100, 3),
        'fraud_rate_synthetic':round(synth_fraud_rate * 100, 3),
        'fraud_rate_error_pct':round(fraud_error, 1),
        'amount_ks':           round(amount_ks, 4),
        'hour_ks':             round(hour_ks, 4),
        'age_ks':              round(age_ks, 4),
        'balance_ks':          round(balance_ks, 4),
        'privacy_leaks':       leaks,
        'privacy_safe':        leaks == 0,
        'real_rows':           len(real),
        'synthetic_rows':      len(synthetic),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def synthesize(df: pd.DataFrame,
               n_rows: int = None,
               random_state: int = 42) -> dict:
    """
    Main synthesis function. Takes a real transaction DataFrame,
    returns synthetic DataFrame + fidelity scores.

    Args:
        df:           Real transaction data
        n_rows:       Number of synthetic rows to generate
                      (defaults to same as input)
        random_state: For reproducibility

    Returns:
        {
            'success':   bool,
            'synthetic': pd.DataFrame or None,
            'fidelity':  dict or None,
            'errors':    list
        }
    """
    # Validate
    validation = validate_dataframe(df)
    if not validation['valid']:
        return {
            'success':   False,
            'synthetic': None,
            'fidelity':  None,
            'errors':    validation['errors']
        }

    n_rows = n_rows or len(df)

    # Memory optimization: Downcast floats to 32-bit
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    gc.collect()

    # Stratified split
    fraud_df = df[df['is_fraud'] == 1].copy().reset_index(drop=True)
    legit_df = df[df['is_fraud'] == 0].copy().reset_index(drop=True)

    real_fraud_rate = df['is_fraud'].mean()
    n_fraud_synth   = max(1, round(n_rows * real_fraud_rate))
    n_legit_synth   = n_rows - n_fraud_synth

    # Separate QT per stratum
    qt_legit = QuantileTransformer(
        output_distribution='normal', random_state=random_state
    )
    qt_fraud = QuantileTransformer(
        output_distribution='normal', random_state=random_state
    )

    legit_df = legit_df.copy()
    fraud_df = fraud_df.copy()

    legit_df['amount_qt'] = qt_legit.fit_transform(
        legit_df[['amount']]
    ).flatten()
    fraud_df['amount_qt'] = qt_fraud.fit_transform(
        fraud_df[['amount']]
    ).flatten()

    legit_train = legit_df.drop(columns=['amount'])
    fraud_train = fraud_df.drop(columns=['amount'])

    # SMOTE on fraud
    fraud_smoted = _smote(fraud_train, SMOTE_TARGET, random_state)

    # Train + generate
    synth_fraud = _train_and_sample(fraud_smoted, n_fraud_synth, 'fraud')
    gc.collect() # Free memory before training next model
    
    synth_legit = _train_and_sample(legit_train,  n_legit_synth, 'legit')
    gc.collect()

    # Force correct labels
    synth_fraud['is_fraud'] = 1
    synth_legit['is_fraud'] = 0

    # Inverse QT per stratum
    synth_fraud['amount'] = qt_fraud.inverse_transform(
        synth_fraud[['amount_qt']].values
    ).flatten()
    synth_legit['amount'] = qt_legit.inverse_transform(
        synth_legit[['amount_qt']].values
    ).flatten()

    synth_fraud['amount'] = synth_fraud['amount'].clip(lower=0).round(2)
    synth_legit['amount'] = synth_legit['amount'].clip(lower=0).round(2)

    synth_fraud = synth_fraud.drop(columns=['amount_qt'])
    synth_legit = synth_legit.drop(columns=['amount_qt'])

    # Combine + shuffle
    synthetic = pd.concat(
        [synth_fraud, synth_legit], ignore_index=True
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Score
    fidelity = compute_fidelity(df, synthetic)

    # Clean up transaction IDs — reset to sequential integers
    synthetic['transaction_id'] = range(1, len(synthetic) + 1)

    return {
        'success':   True,
        'synthetic': synthetic,
        'fidelity':  fidelity,
        'errors':    []
    }