
# Detailed Report

## Objective Function
We minimize CIFAR‑10 validation loss averaged over the last 3 epochs.

## Acquisition Function Justification
Expected Improvement (EI) is used due to strong exploration–exploitation balance and stability for small budgets.

## Search Space
- lr: [1e‑5, 1e‑2]
- batch_size: 32–256
- dropout: 0–0.6
- base_filters: 16–64
- epochs: 3–12

## BO vs Random Search Summary
Bayesian Optimization converges faster with fewer wasted trials. It consistently finds lower validation loss under identical budgets.

(Generated comparative summary is placed in results/comparative_summary.txt)
