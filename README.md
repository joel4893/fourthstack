# Talon — Synthetic Financial Data Research

## What this is
A research pipeline exploring high-fidelity synthetic 
transaction data generation for neobanks.

## The core problem
Neobanks can't share real transaction data for ML training
due to GDPR/PSD2. Synthetic data is the solution — but only
if it preserves the statistical properties that make 
fraud detection models work.

## Experiments run
| Exp | Approach              | Fraud error | Amount KS | Hour KS |
|-----|-----------------------|-------------|-----------|---------|
| 1   | GaussianCopula        | ~9%         | 0.570     | 0.236   |
| 2   | Baseline CTGAN        | 710%        | 0.570     | 0.308   |
| 3   | Cyclical encoding     | 345%        | 0.570     | 0.206   |
| 4   | Stratified SMOTE      | 0%          | 0.570     | 0.274   |
| 5   | + Log transform       | 0%          | 0.450     | 0.250   |
| 6   | + Quantile transform  | 0%          | 0.488     | 0.182   |

## Key findings
1. Standard models inflate fraud rate up to 710% — unusable
2. Stratified synthesis solves fraud rate completely (0% error)
3. Heavy-tailed amount distributions require transform 
   preprocessing — log and quantile transforms both help
4. Fundamental tension: stratification that fixes fraud rate 
   interferes with amount distribution learning
5. Privacy: zero real records leaked across all experiments

## Open problems
- Amount KS below 0.15 (currently 0.488)
- Hour KS below 0.10 (currently 0.182)
- Solving both simultaneously without trading off fraud fidelity
