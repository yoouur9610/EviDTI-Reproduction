"""
Table 2: recompute baseline uncertainty for correct vs incorrect predictions
as MEDIANS (to match the evidential row and the rest of the manuscript).

Means are printed alongside as a cross-check: they should reproduce the
values currently in Table 2 (Softmax 0.2346 / 0.3093, MC Dropout
0.0719 / 0.0962, Evidential Ensemble 0.1121 / 0.1826).

Run from ~/EviDTI (the three CSVs from the earlier baseline runs must be
in the working directory):
    python table2_uncertainty_medians.py
"""
import numpy as np
import pandas as pd

SOFTMAX_RESULT_CSV = '/home/users/nus/e1561459/EviDTI/runs/softmax_seed1/result.csv'


def report(name, unc, correct):
    unc = np.asarray(unc, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    print(f"\n===== {name} =====")
    print(f"  n correct = {correct.sum()}, n incorrect = {(~correct).sum()}, "
          f"accuracy = {correct.mean()*100:.2f}%")
    print(f"  MEDIAN uncertainty  Correct: {np.median(unc[correct]):.4f}   "
          f"Incorrect: {np.median(unc[~correct]):.4f}")
    print(f"  mean uncertainty    Correct: {unc[correct].mean():.4f}   "
          f"Incorrect: {unc[~correct].mean():.4f}   (cross-check vs current Table 2)")
    print(f"  overall MEDIAN = {np.median(unc):.4f}   overall mean = {unc.mean():.4f}")


# ---------------- MC Dropout ----------------
df_mc = pd.read_csv('mc_dropout_results.csv')
report('MC Dropout (std over 30 passes)',
       df_mc['MCDropout_Std'],
       df_mc['Pred_Label'] == df_mc['True_Label'])

# ---------------- Evidential Ensemble ----------------
df_ens = pd.read_csv('evidential_ensemble_results.csv')
print("\n[Evidential Ensemble] columns in CSV:", list(df_ens.columns))
cand = [c for c in df_ens.columns
        if any(k in c.lower() for k in ('disagree', 'std', 'var', 'uncert'))]
if len(cand) == 1:
    report(f'Evidential Ensemble (column: {cand[0]})',
           df_ens[cand[0]],
           df_ens['Ensemble_Pred'] == df_ens['True_Label'])
else:
    print(f"[Evidential Ensemble] could not pick a single disagreement column "
          f"automatically (candidates: {cand}). Tell Claude which column is the "
          f"disagreement measure.")

# ---------------- Softmax confidence ----------------
def parse_bracketed(s):
    if isinstance(s, str):
        return [float(x) for x in s.strip('[]').split()]
    return [float(s)]


df_sm = pd.read_csv(SOFTMAX_RESULT_CSV, index_col=0)
sm_probs = pd.to_numeric(df_sm.iloc[2], errors='coerce').values
pl = np.array(df_sm.iloc[3].apply(parse_bracketed).tolist())
sm_preds, sm_labels = pl[:, 0], pl[:, 1]
sm_conf = np.array([parse_bracketed(x)[0] for x in df_sm.iloc[4]])

# Check the assumption that row 4 is the max softmax probability
ok = np.allclose(sm_conf, np.maximum(sm_probs, 1 - sm_probs), atol=1e-4)
print(f"\n[Softmax] row 4 equals max(p, 1-p): {ok}")
print(f"[Softmax] col0 matches (prob >= 0.5): "
      f"{((sm_probs >= 0.5).astype(int) == sm_preds).mean():.4f}  (should be 1.0)")
if not ok:
    print("[Softmax] WARNING: row 4 is not max(p, 1-p); uncertainty below uses "
          "1 - max(p, 1-p) computed from row 2 instead.")
    sm_conf = np.maximum(sm_probs, 1 - sm_probs)

report('Softmax confidence (uncertainty = 1 - max softmax prob)',
       1 - sm_conf,
       sm_preds == sm_labels)
