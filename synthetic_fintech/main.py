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
print(f"Rows: {len(data)} | Fraud: {real_fraud_rate:.2%}")

# ── 2. Stratified split FIRST ─────────────────────────────────────────────────
fraud_data = data[data['is_fraud']==1].copy().reset_index(drop=True)
legit_data = data[data['is_fraud']==0].copy().reset_index(drop=True)

print("\n── Stratified split ──────────────────────────────")
print(f"Fraud: {len(fraud_data)} rows | Legit: {len(legit_data)} rows")

# ── 3. Separate QT per stratum — THE FIX ─────────────────────────────────────
# Each stratum gets its own quantile transformer
# fitted only on that stratum's amount distribution
# so inverse transform is always correctly calibrated
print("\n── Fitting separate QT per stratum ───────────────")

qt_legit = QuantileTransformer(output_distribution='normal', random_state=42)
qt_fraud  = QuantileTransformer(output_distribution='normal', random_state=42)

legit_data = legit_data.copy()
fraud_data = fraud_data.copy()

legit_data['amount_qt'] = qt_legit.fit_transform(
    legit_data[['amount']]
).flatten()

fraud_data['amount_qt'] = qt_fraud.fit_transform(
    fraud_data[['amount']]
).flatten()

print(f"Legit QT fitted on {len(legit_data)} rows")
print(f"Fraud QT fitted on {len(fraud_data)} rows")

# Drop original amount — model trains on amount_qt only
legit_train = legit_data.drop(columns=['amount'])
fraud_train = fraud_data.drop(columns=['amount'])

# ── 4. SMOTE on fraud ─────────────────────────────────────────────────────────
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

fraud_smoted = manual_smote(fraud_train, target_count=50)
print(f"\n── SMOTE: fraud rows boosted to {len(fraud_smoted)} ──────────")

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
legit_synthesizer = train_ctgan(legit_train,  "LEGIT")

# ── 6. Generate at exact real ratio ──────────────────────────────────────────
total_synth   = 500
n_fraud_synth = round(total_synth * real_fraud_rate)
n_legit_synth = total_synth - n_fraud_synth

print(f"\n── Generating {total_synth} rows ─────────────────────────")
synth_fraud = fraud_synthesizer.sample(num_rows=n_fraud_synth)
synth_legit = legit_synthesizer.sample(num_rows=n_legit_synth)

synth_fraud['is_fraud'] = 1
synth_legit['is_fraud'] = 0

# ── 7. Inverse transform using MATCHING QT per stratum ───────────────────────
print("\n── Inverse transform (matched QT per stratum) ────")

synth_fraud['amount'] = qt_fraud.inverse_transform(
    pd.DataFrame(synth_fraud['amount_qt'].values, columns=['amount'])
).flatten()

synth_legit['amount'] = qt_legit.inverse_transform(
    pd.DataFrame(synth_legit['amount_qt'].values, columns=['amount'])
).flatten()

synth_fraud['amount'] = synth_fraud['amount'].clip(lower=0).round(2)
synth_legit['amount'] = synth_legit['amount'].clip(lower=0).round(2)

synth_fraud = synth_fraud.drop(columns=['amount_qt'])
synth_legit = synth_legit.drop(columns=['amount_qt'])

synthetic_final = pd.concat(
    [synth_fraud, synth_legit], ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Synthetic avg amount: £{synthetic_final['amount'].mean():.2f}")
print(f"Real avg amount:      £{data['amount'].mean():.2f}")

# ── 8. Score ──────────────────────────────────────────────────────────────────
amount_ks, _ = stats.ks_2samp(data['amount'], synthetic_final['amount'])
hour_ks,   _ = stats.ks_2samp(data['transaction_hour'], synthetic_final['transaction_hour'])
synth_fraud_rate = synthetic_final['is_fraud'].mean()
fraud_error      = abs(synth_fraud_rate - real_fraud_rate) / real_fraud_rate * 100
real_set  = set(data.apply(lambda x: tuple(x), axis=1))
synth_set = set(synthetic_final.apply(lambda x: tuple(x), axis=1))
leaks     = len(real_set.intersection(synth_set))

print("\n── FINAL SCORECARD ───────────────────────────────")
print(f"{'Metric':<25} {'Target':>8} {'Exp 6':>8} {'Exp 7':>8} {'Status':>12}")
print("─" * 65)

results = [
    ('Fraud rate error', '< 10%',  '0.0%',  f'{fraud_error:.1f}%',
     '✓ SOLVED' if fraud_error < 10   else '✗ OPEN'),
    ('Amount KS',        '< 0.15', '0.488', f'{amount_ks:.4f}',
     '✓ SOLVED' if amount_ks  < 0.15  else '✗ OPEN'),
    ('Hour KS',          '< 0.10', '0.228', f'{hour_ks:.4f}',
     '✓ SOLVED' if hour_ks    < 0.10  else '✗ OPEN'),
    ('Privacy leaks',    '0',      '0',     str(leaks),
     '✓ SOLVED' if leaks == 0        else '✗ RISK'),
]

for metric, target, exp6, exp7, status in results:
    print(f"{metric:<25} {target:>8} {exp6:>8} {exp7:>8} {status:>12}")

solved = sum(1 for *_, s in results if '✓' in s)
print(f"\n{solved}/4 metrics solved")

if solved == 4:
    print("ALL METRICS SOLVED — pipeline ready to productionise")
elif solved == 3:
    print("3/4 solved — good enough to build on, one open problem remains")
else:
    print("Pipeline needs more work before productionising")

# ── 9. Final chart ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Experiment 7: Separate QT Per Stratum — Final Research Result',
             fontsize=12)

axes[0].hist(data['amount'].clip(0,500), bins=40,
             alpha=0.6, label='Real', color='steelblue', edgecolor='white')
axes[0].hist(synthetic_final['amount'].clip(0,500), bins=40,
             alpha=0.6, label='Synthetic', color='mediumpurple', edgecolor='white')
axes[0].set_title(f'Amount Distribution\nKS: {amount_ks:.4f} (target < 0.15)')
axes[0].set_xlabel('Amount £')
axes[0].legend()

axes[1].hist(data['transaction_hour'], bins=range(0,25),
             alpha=0.6, label='Real', color='steelblue', edgecolor='white')
axes[1].hist(synthetic_final['transaction_hour'], bins=range(0,25),
             alpha=0.6, label='Synthetic', color='mediumpurple', edgecolor='white')
axes[1].set_title(f'Transaction Hour\nKS: {hour_ks:.4f} (target < 0.10)')
axes[1].set_xlabel('Hour of day')
axes[1].legend()

experiments  = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7']
amount_prog  = [0.570,   0.570,  0.570,  0.570,  0.450,  0.488,  amount_ks]
axes[2].plot(experiments, amount_prog, 'o-',
             color='mediumpurple', linewidth=2, markersize=6, label='Amount KS')
axes[2].axhline(y=0.15, color='mediumpurple', linestyle='--',
                alpha=0.5, label='Amount target')
axes[2].set_title('Amount KS Progress\nAcross All Experiments')
axes[2].set_ylabel('KS Score (lower = better)')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('exp7_final.png', dpi=150, bbox_inches='tight')
print("\n── Saved: exp7_final.png ─────────────────────────")
print("Research phase complete. Time to build the product.")