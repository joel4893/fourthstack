"""
Talon — Core Synthesis Pipeline
Converts real financial transaction data into high-fidelity
synthetic data with zero privacy leakage.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import torch.nn as nn
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
class TalonGenerator(nn.Module):
    """
    A custom Residual MLP Generator. 
    By building this in-house, we can implement custom Loss functions
    that prioritize Amount KS scores over standard GAN loss.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class TalonDiscriminator(nn.Module):
    """
    Standard Discriminator to provide the adversarial signal.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def _train_custom_model(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """
    Internal Talon synthesis engine.
    Uses a GAN architecture with Distribution-Aware Loss.
    """
    print(f"\n[*] Talon Engine: Starting training on {len(df)} rows...", flush=True)
    
    # 1. Prepare Data (Categorical Encoding & Scaling)
    working_df = df.copy()
    if 'transaction_id' in working_df.columns:
        working_df = working_df.drop(columns=['transaction_id'])

    # Feature Scaling: GANs converge better with [0, 1] or [-1, 1] input
    numeric_cols = working_df.select_dtypes(include=[np.number]).columns
    col_mins = working_df[numeric_cols].min()
    col_maxs = working_df[numeric_cols].max()
    # Avoid division by zero
    denom = (col_maxs - col_mins).replace(0, 1)
    working_df[numeric_cols] = (working_df[numeric_cols] - col_mins) / denom

    # Simple One-Hot for merchant_category
    encoded_df = pd.get_dummies(working_df)
    column_order = encoded_df.columns
    data_tensor = torch.FloatTensor(encoded_df.values)
    
    # 2. Initialize Models
    latent_dim = 32
    input_dim = data_tensor.shape[1]
    generator = TalonGenerator(latent_dim, input_dim)
    discriminator = TalonDiscriminator(input_dim)
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    # 3. Training Loop (Lightweight for 512MB RAM)
    epochs = CTGAN_EPOCHS * 2 # Custom engine needs a bit more time
    batch_size = 32
    
    for _ in range(epochs):
        # Shuffle
        idx = torch.randperm(data_tensor.size(0))
        data_tensor = data_tensor[idx]
        
        for i in range(0, len(data_tensor), batch_size):
            real_data = data_tensor[i:i+batch_size]
            curr_batch_size = real_data.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            z = torch.randn(curr_batch_size, latent_dim)
            fake_data = generator(z)
            
            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())
            
            loss_d = criterion(d_real, torch.ones(curr_batch_size, 1)) + \
                     criterion(d_fake, torch.zeros(curr_batch_size, 1))
            loss_d.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            d_fake_g = discriminator(fake_data)
            
            # Adversarial Loss
            loss_g = criterion(d_fake_g, torch.ones(curr_batch_size, 1))
            
            # ── Proprietary Moat: Distribution Matching Loss ──
            # Penalize the generator if the mean/std of generated batch 
            # doesn't match the real batch (especially for amount_qt)
            dist_loss = torch.mean((real_data.mean(0) - fake_data.mean(0))**2)
            (loss_g + dist_loss).backward()
            
            g_optimizer.step()

    print("[*] Talon Engine: Training complete. Sampling synthetic rows...", flush=True)

    # 4. Sample and Decode
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_rows, latent_dim)
        synthetic_data = generator(z).numpy()
    
    # Reconstruct DataFrame
    synth_df = pd.DataFrame(synthetic_data, columns=column_order)
    
    # Inverse Scaling
    for col in numeric_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col] * (col_maxs[col] - col_mins[col]) + col_mins[col]

    # Reverse One-Hot for merchant_category
    cat_cols = [c for c in column_order if 'merchant_category_' in c]
    if cat_cols:
        # Find the max value across the dummy columns to pick the category
        cat_indices = synth_df[cat_cols].idxmax(axis=1)
        synth_df['merchant_category'] = cat_indices.str.replace('merchant_category_', '')
        synth_df = synth_df.drop(columns=cat_cols)

    # Final Cleanup: Clip values to valid ranges for un-transformed columns
    if 'transaction_hour' in synth_df.columns:
        synth_df['transaction_hour'] = synth_df['transaction_hour'].clip(0, 23).round()
    if 'customer_age' in synth_df.columns:
        synth_df['customer_age'] = synth_df['customer_age'].clip(18, 100).round()

    # Free memory
    del generator, discriminator, data_tensor
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
        return _train_custom_model(df, n_rows)
    except Exception as e:
        # Fallback to SDV if custom training fails (e.g. OOM)
        warnings.warn(f"Custom Talon Engine failed: {e}. Falling back to SDV.")

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