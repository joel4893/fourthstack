import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# ── 1. Same transaction data ──────────────────────────────────────────────────
np.random.seed(42)
n = 500

def generate_transaction_hours(n):
    hours = []
    for _ in range(n):
        cluster = np.random.choice(
            ['midnight', 'lunch', 'evening', 'other'],
            p=[0.25, 0.20, 0.30, 0.25]
        )
        if cluster == 'midnight':
            h = int(np.clip(np.random.normal(1, 1), 0, 3))
        elif cluster == 'lunch':
            h = int(np.clip(np.random.normal(12, 1), 10, 14))
        elif cluster == 'evening':
            h = int(np.clip(np.random.normal(19, 1.5), 16, 23))
        else:
            h = np.random.randint(0, 24)
        hours.append(h)
    return hours

data = pd.DataFrame({
    'transaction_id': range(1, n+1),
    'amount': np.round(np.random.lognormal(mean=3.5, sigma=1.2, size=n), 2),
    'merchant_category': np.random.choice(
        ['groceries', 'transport', 'dining', 'entertainment', 'utilities'],
        size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    ),
    'transaction_hour': generate_transaction_hours(n),
    'is_fraud': np.random.choice([0, 1], size=n, p=[0.97, 0.03]),
    'customer_age': np.random.randint(18, 75, size=n),
    'account_balance': np.round(
        np.random.lognormal(mean=7, sigma=1, size=n), 2
    )
})

print("── Real data ready ───────────────────────────────")
print(f"Shape: {data.shape}")

# ── 2. THE FIX: Cyclical encoding + per-category synthesis ───────────────────
print("\n── Applying fixes ────────────────────────────────")
print("Fix 1: Encoding transaction_hour as cyclical sin/cos features")
print("Fix 2: Training separate CTGAN per merchant category")
print()

# FIX 1: Cyclical encoding
# Instead of hour=23 and hour=0 looking far apart to the model,
# we encode them as points on a circle using sin and cos.
# Hour 23 and hour 0 become adjacent points — which they actually are.
def add_cyclical_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
    return df

def recover_hour_from_cyclical(df):
    """Convert sin/cos back to hour after synthesis"""
    df = df.copy()
    angle = np.arctan2(df['hour_sin'], df['hour_cos'])
    df['transaction_hour'] = np.round(
        (angle / (2 * np.pi) * 24) % 24
    ).astype(int)
    df = df.drop(columns=['hour_sin', 'hour_cos'])
    return df

# FIX 2: Per-category synthesis
# Dining transactions peak at lunch. Transport peaks at morning commute.
# Training one model on everything averages these patterns out and loses them.
# Training per category preserves each category's unique timing signature.

categories = data['merchant_category'].unique()
synthetic_parts = []

for category in categories:
    subset = data[data['merchant_category'] == category].copy()
    subset = subset.reset_index(drop=True)
    
    # Apply cyclical encoding to this subset
    subset_encoded = add_cyclical_features(subset)
    
    # Drop original hour column — model only sees sin/cos
    subset_encoded = subset_encoded.drop(columns=['transaction_hour'])
    
    print(f"Training CTGAN for '{category}' ({len(subset)} rows)...")
    
    # Build metadata for this subset
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(subset_encoded)
    meta.update_column(column_name='transaction_id', sdtype='id')
    
    # Train CTGAN on this category's encoded data
    synthesizer = CTGANSynthesizer(meta, epochs=150, verbose=False)
    synthesizer.fit(subset_encoded)
    
    # Generate same number of rows as original category
    synth_subset = synthesizer.sample(num_rows=len(subset))
    
    # Recover hour from sin/cos
    synth_subset = recover_hour_from_cyclical(synth_subset)
    
    # Restore category column (CTGAN might drift it slightly)
    synth_subset['merchant_category'] = category
    
    synthetic_parts.append(synth_subset)
    print(f"  Done. Generated {len(synth_subset)} synthetic rows.")

# Combine all categories back together
improved_synthetic = pd.concat(synthetic_parts, ignore_index=True)
print(f"\nTotal synthetic rows: {len(improved_synthetic)}")

# ── 3. Baseline CTGAN (no fixes) for comparison ──────────────────────────────
print("\n── Training baseline CTGAN (no fixes) ───────────")
meta_base = SingleTableMetadata()
meta_base.detect_from_dataframe(data)
meta_base.update_column(column_name='transaction_id', sdtype='id')
baseline_ctgan = CTGANSynthesizer(meta_base, epochs=150, verbose=False)
baseline_ctgan.fit(data)
baseline_synthetic = baseline_ctgan.sample(num_rows=500)
print("Baseline done.")

# ── 4. Score all three ────────────────────────────────────────────────────────
baseline_ks, _ = stats.ks_2samp(
    data['transaction_hour'],
    baseline_synthetic['transaction_hour']
)
improved_ks, _ = stats.ks_2samp(
    data['transaction_hour'],
    improved_synthetic['transaction_hour']
)

print("\n── Results ───────────────────────────────────────")
print(f"{'Model':<35} {'KS Score':>10} {'Better?':>10}")
print("─" * 58)
print(f"{'Baseline CTGAN (last experiment)':<35} {0.2360:>10.4f} {'baseline':>10}")
print(f"{'CTGAN no fixes (this run)':<35} {baseline_ks:>10.4f}")
print(f"{'CTGAN + cyclical + per-category':<35} {improved_ks:>10.4f}", end="")

if improved_ks < baseline_ks:
    improvement = ((baseline_ks - improved_ks) / baseline_ks) * 100
    print(f"  ← {improvement:.1f}% better")
else:
    print("  ← interesting, let's investigate")

# ── 5. Fraud rate preservation ────────────────────────────────────────────────
print("\n── Fraud rate preservation ───────────────────────")
print(f"Real:                    {data['is_fraud'].mean():.3%}")
print(f"Baseline CTGAN:          {baseline_synthetic['is_fraud'].mean():.3%}")
print(f"Improved (per-category): {improved_synthetic['is_fraud'].mean():.3%}")

# ── 6. Per-category hour analysis — the real test ─────────────────────────────
print("\n── Per-category temporal fidelity ────────────────")
print(f"{'Category':<15} {'Real peak hr':>13} {'Improved peak hr':>17} {'Match?':>8}")
print("─" * 56)

for cat in categories:
    real_peak = data[data['merchant_category'] == cat]['transaction_hour'].mode()[0]
    synth_peak = improved_synthetic[
        improved_synthetic['merchant_category'] == cat
    ]['transaction_hour'].mode()[0]
    match = "✓" if abs(real_peak - synth_peak) <= 2 else "✗"
    print(f"{cat:<15} {real_peak:>13} {synth_peak:>17} {match:>8}")

# ── 7. The three-way chart ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Fixing Temporal Patterns: Three Generations of the Model', fontsize=12)

bins = range(0, 25)

axes[0].hist(data['transaction_hour'], bins=bins,
             color='steelblue', alpha=0.85, edgecolor='white')
axes[0].set_title('Real Data\n(ground truth)')
axes[0].set_xlabel('Hour of day')
axes[0].set_ylabel('Count')
for peak, color, label in [(1,'red','midnight'), (12,'orange','lunch'), (19,'green','evening')]:
    axes[0].axvline(x=peak, color=color, linestyle='--', alpha=0.6, label=label)
axes[0].legend(fontsize=8)

axes[1].hist(baseline_synthetic['transaction_hour'], bins=bins,
             color='coral', alpha=0.85, edgecolor='white')
axes[1].set_title(f'Baseline CTGAN\n(KS: {baseline_ks:.4f})')
axes[1].set_xlabel('Hour of day')
for peak, color in [(1,'red'), (12,'orange'), (19,'green')]:
    axes[1].axvline(x=peak, color=color, linestyle='--', alpha=0.6)

axes[2].hist(improved_synthetic['transaction_hour'], bins=bins,
             color='mediumseagreen', alpha=0.85, edgecolor='white')
axes[2].set_title(f'Improved CTGAN\n(cyclical + per-category) KS: {improved_ks:.4f}')
axes[2].set_xlabel('Hour of day')
for peak, color in [(1,'red'), (12,'orange'), (19,'green')]:
    axes[2].axvline(x=peak, color=color, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('fixed_temporal.png', dpi=150, bbox_inches='tight')
print("\n── Saved: fixed_temporal.png ─────────────────────")
print("Open fixed_temporal.png to see if the lunch peak is back.")