import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# ── 1. Transaction data ───────────────────────────────────────────────────────
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
    'transaction_id':    range(1, n+1),
    'amount':            np.round(np.random.lognormal(3.5, 1.2, n), 2),
    'merchant_category': np.random.choice(
        ['groceries','transport','dining','entertainment','utilities'],
        size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    ),
    'transaction_hour':  generate_transaction_hours(n),
    'is_fraud':          np.random.choice([0,1], size=n, p=[0.97, 0.03]),
    'customer_age':      np.random.randint(18, 75, n),
    'account_balance':   np.round(np.random.lognormal(7, 1, n), 2)
})

real_fraud_rate = data['is_fraud'].mean()
print("── Real data ─────────────────────────────────────")
print(f"Rows:        {len(data)}")
print(f"Fraud rate:  {real_fraud_rate:.2%}")
print(f"Avg amount:  £{data['amount'].mean():.2f}")

# ── 2. Quantile transform amount ──────────────────────────────────────────────
# Maps your exact amount distribution → perfect normal bell curve
# Much more powerful than log transform for heavy-tailed data
print("\n── Quantile transforming amount ──────────────────")
qt = QuantileTransformer(output_distribution='normal', random_state=42)
data['amount_qt'] = qt.fit_transform(data[['amount']]).flatten()

print(f"Original amount: skew={data['amount'].skew():.2f} "
      f"(high = heavy tail)")
print(f"After QT:        skew={data['amount_qt'].skew():.2f} "
      f"(close to 0 = bell curve)")
print("CTGAN can now model this distribution cleanly")

# Drop original amount — model trains on amount_qt only
data_transformed = data.drop(columns=['amount'])

# ── 3. SMOTE ──────────────────────────────────────────────────────────────────
def manual_smote(minority_df, target_count, random_state=42):
    np.random.seed(random_state)
    numeric_cols     = minority_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = minority_df.select_dtypes(exclude=[np.number]).columns.tolist()
    rows_needed      = target_count - len(minority_df)
    if rows_needed <= 0:
        return minority_df.copy()
    synthetic_rows = []
    minority_arr   = minority_df[numeric_cols].values
    for _ in range(rows_needed):
        idx       = np.random.randint(0, len(minority_df))
        neighbour = np.random.randint(0, len(minority_df))
        alpha     = np.random.random()
        new_num   = minority_arr[idx] + alpha * (
            minority_arr[neighbour] - minority_arr[idx]
        )
        new_row = {col: round(float(new_num[i]), 4)
                   for i, col in enumerate(numeric_cols)}
        base = minority_df.iloc[idx]
        for col in categorical_cols:
            new_row[col] = base[col]
        synthetic_rows.append(new_row)
    return pd.concat(
        [minority_df.reset_index(drop=True), pd.DataFrame(synthetic_rows)],
        ignore_index=True
    )

# ── 4. Stratified split ───────────────────────────────────────────────────────
fraud_data   = data_transformed[data_transformed['is_fraud']==1].copy().reset_index(drop=True)
legit_data   = data_transformed[data_transformed['is_fraud']==0].copy().reset_index(drop=True)
fraud_smoted = manual_smote(fraud_data, target_count=50)

print(f"\n── Stratified split ──────────────────────────────")
print(f"Fraud (after SMOTE): {len(fraud_smoted)} rows")
print(f"Legit:               {len(legit_data)} rows")

# ── 5. Train two CTGANs ───────────────────────────────────────────────────────
def train_ctgan(df, label):
    print(f"\n── Training CTGAN: {label} ({len(df)} rows) ────────")
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    meta.update_column(column_name='transaction_id', sdtype='id')
    synth = CTGANSynthesizer(meta, epochs=150, verbose=False)
    synth.fit(df)
    print("   Done.")
    return synth

fraud_synthesizer = train_ctgan(fraud_smoted, "FRAUD")
legit_synthesizer = train_ctgan(legit_data,   "LEGIT")

# ── 6. Generate at exact real ratio ──────────────────────────────────────────
total_synth   = 500
n_fraud_synth = round(total_synth * real_fraud_rate)
n_legit_synth = total_synth - n_fraud_synth

print(f"\n── Generating {total_synth} rows at {real_fraud_rate:.2%} fraud ──────")
synth_fraud = fraud_synthesizer.sample(num_rows=n_fraud_synth)
synth_legit = legit_synthesizer.sample(num_rows=n_legit_synth)

synth_fraud['is_fraud'] = 1
synth_legit['is_fraud'] = 0

synthetic_raw = pd.concat(
    [synth_fraud, synth_legit], ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# ── 7. Inverse quantile transform → recover real amount scale ─────────────────
print("\n── Inverse quantile transform → recovering amounts ─")
synthetic_raw['amount'] = qt.inverse_transform(
    synthetic_raw[['amount_qt']].rename(columns={'amount_qt': 'amount'})
).flatten()
synthetic_raw['amount'] = synthetic_raw['amount'].clip(lower=0).round(2)
synthetic_final = synthetic_raw.drop(columns=['amount_qt'])

print(f"Synthetic avg amount:  £{synthetic_final['amount'].mean():.2f}")
print(f"Real avg amount:       £{data['amount'].mean():.2f}")

# ── 8. Score everything ───────────────────────────────────────────────────────
amount_ks, _ = stats.ks_2samp(data['amount'],            synthetic_final['amount'])
hour_ks,   _ = stats.ks_2samp(data['transaction_hour'],  synthetic_final['transaction_hour'])
age_ks,    _ = stats.ks_2samp(data['customer_age'],      synthetic_final['customer_age'])

synth_fraud_rate = synthetic_final['is_fraud'].mean()
fraud_error      = abs(synth_fraud_rate - real_fraud_rate) / real_fraud_rate * 100

real_set  = set(data.apply(lambda x: tuple(x), axis=1))
synth_set = set(synthetic_final.apply(lambda x: tuple(x), axis=1))
leaks     = len(real_set.intersection(synth_set))

print(f"\n── Full scorecard ────────────────────────────────")
print(f"{'Metric':<25} {'Target':>8} {'Exp 5':>8} {'Exp 6':>8} {'Status':>12}")
print("─" * 65)

results = [
    ('Fraud rate error', '< 10%',  '0.0%',  f'{fraud_error:.1f}%',
     'SOLVED' if fraud_error < 10 else 'NEEDS WORK'),
    ('Amount KS',        '< 0.15', '0.450', f'{amount_ks:.4f}',
     'SOLVED' if amount_ks < 0.15 else 'NEEDS WORK'),
    ('Hour KS',          '< 0.10', '0.250', f'{hour_ks:.4f}',
     'SOLVED' if hour_ks < 0.10 else 'NEEDS WORK'),
    ('Privacy leaks',    '0',      '0',     str(leaks),
     'SOLVED' if leaks == 0 else 'RISK'),
]

for metric, target, exp5, exp6, status in results:
    print(f"{metric:<25} {target:>8} {exp5:>8} {exp6:>8} {status:>12}")

solved = sum(1 for *_, s in results if s == 'SOLVED')
print(f"\n{solved}/4 metrics solved")

# ── 9. Charts ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Experiment 6: Quantile Transform — Final Push', fontsize=13)

# Amount distribution
axes[0,0].hist(data['amount'].clip(0, 500), bins=40,
               alpha=0.6, label='Real', color='steelblue', edgecolor='white')
axes[0,0].hist(synthetic_final['amount'].clip(0, 500), bins=40,
               alpha=0.6, label='Synthetic', color='mediumpurple', edgecolor='white')
axes[0,0].set_title(f'Transaction Amount\nKS: {amount_ks:.4f} (target < 0.15)')
axes[0,0].set_xlabel('Amount £')
axes[0,0].legend()

# Quantile transformed view
axes[0,1].hist(data['amount_qt'], bins=40,
               alpha=0.6, label='Real (QT)', color='steelblue', edgecolor='white')
axes[0,1].hist(synthetic_raw['amount_qt'], bins=40,
               alpha=0.6, label='Synthetic (QT)', color='mediumpurple', edgecolor='white')
axes[0,1].set_title('Amount After Quantile Transform\n(what CTGAN learned)')
axes[0,1].set_xlabel('Transformed amount')
axes[0,1].legend()

# Transaction hour
axes[1,0].hist(data['transaction_hour'], bins=range(0, 25),
               alpha=0.6, label='Real', color='steelblue', edgecolor='white')
axes[1,0].hist(synthetic_final['transaction_hour'], bins=range(0, 25),
               alpha=0.6, label='Synthetic', color='mediumpurple', edgecolor='white')
axes[1,0].set_title(f'Transaction Hour\nKS: {hour_ks:.4f}')
axes[1,0].set_xlabel('Hour of day')
axes[1,0].legend()

# Progress across all 6 experiments
exp_labels  = ['Exp1\nGauss', 'Exp2\nCTGAN', 'Exp3\nCyclical',
               'Exp4\nStratified', 'Exp5\nLog', 'Exp6\nQuantile']
amount_hist = [0.570, 0.570, 0.570, 0.570, 0.450, amount_ks]
fraud_hist  = [0.09,  1.78,  0.98,  0.00,  0.00,  fraud_error/100]
x = np.arange(len(exp_labels))
w = 0.35
bars1 = axes[1,1].bar(x - w/2, amount_hist, w,
                       label='Amount KS', color='steelblue', alpha=0.8)
bars2 = axes[1,1].bar(x + w/2, fraud_hist, w,
                       label='Fraud error', color='coral', alpha=0.8)
axes[1,1].axhline(y=0.15, color='steelblue', linestyle='--',
                   alpha=0.5, label='Amount target')
axes[1,1].axhline(y=0.10, color='red', linestyle='--',
                   alpha=0.5, label='Fraud target')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(exp_labels, fontsize=8)
axes[1,1].set_title('Progress Across All Experiments')
axes[1,1].legend(fontsize=7)

plt.tight_layout()
plt.savefig('exp6_quantile.png', dpi=150, bbox_inches='tight')
print("\n── Saved: exp6_quantile.png ──────────────────────")
print("Open it in the sidebar.")