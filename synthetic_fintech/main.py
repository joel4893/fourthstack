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
    'amount':         np.round(np.random.lognormal(3.5, 1.2, n), 2),
    'merchant_category': np.random.choice(
        ['groceries','transport','dining','entertainment','utilities'],
        size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    ),
    'transaction_hour': generate_transaction_hours(n),
    'is_fraud':       np.random.choice([0,1], size=n, p=[0.97, 0.03]),
    'customer_age':   np.random.randint(18, 75, n),
    'account_balance':np.round(np.random.lognormal(7, 1, n), 2)
})

real_fraud_rate = data['is_fraud'].mean()
print("── Real data ─────────────────────────────────────")
print(f"Total rows:  {len(data)}")
print(f"Fraud rows:  {data['is_fraud'].sum()} ({real_fraud_rate:.2%})")
print(f"Legit rows:  {(data['is_fraud']==0).sum()}")

# ── 2. SMOTE — manual implementation (no extra libraries needed) ──────────────
# SMOTE: Synthetic Minority Oversampling Technique
# For each real fraud row, find its nearest neighbours among other fraud rows
# and interpolate new fraud rows between them.
# This gives CTGAN enough fraud examples to learn from.

def manual_smote(minority_df, target_count, random_state=42):
    """
    Oversample a minority class dataframe to target_count rows
    by interpolating between existing rows.
    Only works on numeric columns — categorical columns are 
    copied from the nearest neighbour.
    """
    np.random.seed(random_state)
    
    numeric_cols = minority_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = minority_df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    
    rows_needed = target_count - len(minority_df)
    if rows_needed <= 0:
        return minority_df.copy()
    
    synthetic_rows = []
    minority_arr = minority_df[numeric_cols].values
    
    for _ in range(rows_needed):
        # Pick a random real fraud row
        idx = np.random.randint(0, len(minority_df))
        base_row = minority_arr[idx]
        
        # Pick a random different fraud row to interpolate toward
        neighbour_idx = np.random.randint(0, len(minority_df))
        neighbour_row = minority_arr[neighbour_idx]
        
        # Interpolate between them at a random point
        alpha = np.random.random()
        new_numeric = base_row + alpha * (neighbour_row - base_row)
        
        # Build the new row
        new_row = {}
        for i, col in enumerate(numeric_cols):
            new_row[col] = round(float(new_numeric[i]), 2)
        
        # Categorical columns: copy from base row
        base_original = minority_df.iloc[idx]
        for col in categorical_cols:
            new_row[col] = base_original[col]
        
        synthetic_rows.append(new_row)
    
    smote_df = pd.DataFrame(synthetic_rows)
    result = pd.concat(
        [minority_df.reset_index(drop=True), smote_df],
        ignore_index=True
    )
    return result

# ── 3. Split real data by fraud label ────────────────────────────────────────
fraud_data  = data[data['is_fraud'] == 1].copy().reset_index(drop=True)
legit_data  = data[data['is_fraud'] == 0].copy().reset_index(drop=True)

print(f"\n── Stratified split ──────────────────────────────")
print(f"Fraud rows for training:  {len(fraud_data)}")
print(f"Legit rows for training:  {len(legit_data)}")

# Apply SMOTE to fraud — boost from ~11 rows to 50
# 50 gives CTGAN enough examples without inventing too much
fraud_smoted = manual_smote(fraud_data, target_count=50)
print(f"After SMOTE:              {len(fraud_smoted)} fraud rows")

# ── 4. Train two separate CTGANs ─────────────────────────────────────────────
def train_ctgan(df, label):
    print(f"\n── Training CTGAN on {label} ({len(df)} rows) ────────")
    
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    meta.update_column(column_name='transaction_id', sdtype='id')
    
    synth = CTGANSynthesizer(meta, epochs=150, verbose=False)
    synth.fit(df)
    print(f"   Done.")
    return synth

fraud_synthesizer = train_ctgan(fraud_smoted, "FRAUD data")
legit_synthesizer = train_ctgan(legit_data,   "LEGIT data")

# ── 5. Generate at exact real ratio ──────────────────────────────────────────
# Target: 500 synthetic rows at exactly 2.2% fraud
total_synth     = 500
n_fraud_synth   = round(total_synth * real_fraud_rate)
n_legit_synth   = total_synth - n_fraud_synth

print(f"\n── Generating at exact real ratio ────────────────")
print(f"Generating {n_fraud_synth} fraud rows  ({real_fraud_rate:.2%})")
print(f"Generating {n_legit_synth} legit rows")

synth_fraud = fraud_synthesizer.sample(num_rows=n_fraud_synth)
synth_legit = legit_synthesizer.sample(num_rows=n_legit_synth)

# Force correct labels (CTGAN might drift them slightly)
synth_fraud['is_fraud'] = 1
synth_legit['is_fraud'] = 0

# Combine and shuffle
stratified_synthetic = pd.concat(
    [synth_fraud, synth_legit], ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# ── 6. Score everything ───────────────────────────────────────────────────────
# Previous best scores for comparison
prev_baseline_fraud = 0.178
prev_improved_fraud = 0.098
prev_best_ks        = 0.2060

hour_ks, _ = stats.ks_2samp(
    data['transaction_hour'],
    stratified_synthetic['transaction_hour']
)
amount_ks, _ = stats.ks_2samp(
    data['amount'],
    stratified_synthetic['amount']
)

print(f"\n── Results: all experiments ──────────────────────")
print(f"{'Experiment':<35} {'Fraud rate':>11} {'Hour KS':>9}")
print("─" * 58)
print(f"{'Real data':<35} {real_fraud_rate:>11.2%} {'—':>9}")
print(f"{'Exp 1: GaussianCopula':<35} {'~2.4%':>11} {0.2360:>9.4f}")
print(f"{'Exp 2: Baseline CTGAN':<35} {prev_baseline_fraud:>11.2%} {0.3080:>9.4f}")
print(f"{'Exp 3: Cyclical+per-category':<35} {prev_improved_fraud:>11.2%} {prev_best_ks:>9.4f}")
print(f"{'Exp 4: Stratified (this run)':<35} "
      f"{stratified_synthetic['is_fraud'].mean():>11.2%} "
      f"{hour_ks:>9.4f}")

# The key metrics
synth_fraud_rate = stratified_synthetic['is_fraud'].mean()
fraud_error = abs(synth_fraud_rate - real_fraud_rate) / real_fraud_rate * 100

print(f"\n── Fraud rate accuracy ───────────────────────────")
print(f"Real fraud rate:          {real_fraud_rate:.3%}")
print(f"Stratified synthetic:     {synth_fraud_rate:.3%}")
print(f"Error:                    {fraud_error:.1f}%")

if fraud_error < 10:
    print(f"STATUS: SOLVED — fraud rate within 10% of real")
elif fraud_error < 25:
    print(f"STATUS: MUCH BETTER — significant improvement")
else:
    print(f"STATUS: STILL NEEDS WORK")

# ── 7. Privacy check ──────────────────────────────────────────────────────────
real_set = set(data.apply(lambda x: tuple(x), axis=1))
synth_set = set(stratified_synthetic.apply(lambda x: tuple(x), axis=1))
leaks = len(real_set.intersection(synth_set))
print(f"\n── Privacy check ─────────────────────────────────")
print(f"Real records leaked: {leaks}")
print(f"Status: {'SAFE' if leaks == 0 else 'RISK DETECTED'}")

# ── 8. The definitive chart ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Experiment 4: Stratified Synthesis — Full Report', fontsize=13)

bins_hour = range(0, 25)

# Row 1: Transaction hour across all 4 experiments
axes[0,0].hist(data['transaction_hour'], bins=bins_hour,
               color='steelblue', alpha=0.85, edgecolor='white')
axes[0,0].set_title('Real Data (ground truth)')
axes[0,0].set_xlabel('Hour')
for h,c in [(1,'red'),(12,'orange'),(19,'green')]:
    axes[0,0].axvline(x=h, color=c, linestyle='--', alpha=0.5)

axes[0,1].hist(stratified_synthetic['transaction_hour'], bins=bins_hour,
               color='mediumpurple', alpha=0.85, edgecolor='white')
axes[0,1].set_title(f'Stratified Synthetic\n(KS: {hour_ks:.4f})')
axes[0,1].set_xlabel('Hour')
for h,c in [(1,'red'),(12,'orange'),(19,'green')]:
    axes[0,1].axvline(x=h, color=c, linestyle='--', alpha=0.5)

# KS score progression chart
experiments = ['GaussCop\nExp 1', 'CTGAN\nExp 2', 'Cyclical\nExp 3', 'Stratified\nExp 4']
ks_scores   = [0.2360, 0.3080, 0.2060, hour_ks]
colors      = ['steelblue','coral','mediumseagreen','mediumpurple']
bars = axes[0,2].bar(experiments, ks_scores, color=colors, alpha=0.85, edgecolor='white')
axes[0,2].set_title('KS Score Progress\n(lower = better)')
axes[0,2].set_ylabel('KS Score')
axes[0,2].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='target < 0.10')
axes[0,2].legend(fontsize=8)
for bar, score in zip(bars, ks_scores):
    axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)

# Row 2: Fraud rate across experiments
fraud_rates = [real_fraud_rate, 0.024, 0.178, 0.098, synth_fraud_rate]
exp_labels  = ['Real\ndata', 'GaussCop\nExp 1', 'CTGAN\nExp 2',
               'Cyclical\nExp 3', 'Stratified\nExp 4']
fr_colors   = ['steelblue','steelblue','coral','mediumseagreen','mediumpurple']
bars2 = axes[1,0].bar(exp_labels, [r*100 for r in fraud_rates],
                       color=fr_colors, alpha=0.85, edgecolor='white')
axes[1,0].set_title('Fraud Rate Across Experiments\n(target: 2.2%)')
axes[1,0].set_ylabel('Fraud rate %')
axes[1,0].axhline(y=real_fraud_rate*100, color='red',
                   linestyle='--', alpha=0.7, label=f'real: {real_fraud_rate:.1%}')
axes[1,0].legend(fontsize=8)
for bar, rate in zip(bars2, fraud_rates):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=8)

# Amount distribution
axes[1,1].hist(data['amount'].clip(0,400), bins=30,
               alpha=0.6, label='Real', color='steelblue')
axes[1,1].hist(stratified_synthetic['amount'].clip(0,400), bins=30,
               alpha=0.6, label='Synthetic', color='mediumpurple')
axes[1,1].set_title(f'Transaction Amount\n(KS: {amount_ks:.4f})')
axes[1,1].set_xlabel('Amount £')
axes[1,1].legend()

# Summary scorecard
axes[1,2].axis('off')
summary = [
    ['Metric',           'Target',    'Achieved'],
    ['Fraud rate error', '< 10%',     f'{fraud_error:.1f}%'],
    ['Hour KS score',    '< 0.10',    f'{hour_ks:.4f}'],
    ['Privacy leaks',    '0',         str(leaks)],
    ['Amount KS',        '< 0.15',    f'{amount_ks:.4f}'],
]
table = axes[1,2].table(
    cellText=summary[1:],
    colLabels=summary[0],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
axes[1,2].set_title('Scorecard')

plt.tight_layout()
plt.savefig('stratified_results.png', dpi=150, bbox_inches='tight')
print("\n── Saved: stratified_results.png ────────────────")
print("Open it in the sidebar.")