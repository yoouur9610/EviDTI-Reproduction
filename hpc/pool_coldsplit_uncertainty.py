"""
Pool result.csv files for the cold-drug and cold-target 5-seed runs and
compute uncertainty diagnostics comparable across splits.

Two DIFFERENT questions, both needed, computed separately per seed/run:
  (a) MAGNITUDE -- does u run higher on average for cold-split (unfamiliar)
      samples than for the random split? -> per-seed/run median u, then
      mean +/- SD across seeds/runs. This is what R2#7 is actually asking.
      Caveat: absolute u scale is still confounded by each run being a
      separately trained model -- report it, don't hide it.
  (b) RANKING -- within a given model, does higher u actually pick out the
      samples it gets wrong? -> per-seed/run AUROC of u vs (1-correct),
      then mean +/- SD across seeds/runs. Computed per-seed/run (NOT on
      the pooled/concatenated set) to avoid between-seed heterogeneity
      (different accuracy + different u scale per seed) inflating or
      deflating a single pooled AUROC number.

result.csv layout (confirm with `head -c 300 result.csv` before trusting
this): index_col=0, columns = samples, rows (after index) =
  row0 = t_id, row1 = d_id, row2 = prob_list,
  row3 = [pred, label] as numpy-array-string e.g. "[1 0]",
  row4 = confidence/uncertainty, row5 = var, row6 = ev, row7 = bk_list

EDIT the path dicts below, then run: python pool_coldsplit_uncertainty.py
"""
import numpy as np
import pandas as pd
import re
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

# ---- EDIT THESE ----
COLD_DRUG_PATHS = {
    1:  "~/EviDTI/runs/coldsplit_seed1/result.csv",
    6:  "~/EviDTI/runs/coldsplit_seed6/result.csv",
    42: "~/EviDTI/runs/coldsplit_seed42/result.csv",
    7:  "~/EviDTI/runs/coldsplit_seed7/result.csv",
    17: "~/EviDTI/runs/coldsplit_seed17/result.csv",
}

COLD_TARGET_PATHS = {
    1:  "~/EviDTI/runs/coldtarget_seed1/result.csv",
    6:  "~/EviDTI/runs/coldtarget_seed6/result.csv",
    42: "~/EviDTI/runs/coldtarget_seed42/result.csv",
    7:  "~/EviDTI/runs/coldtarget_seed7/result.csv",
    17: "~/EviDTI/runs/coldtarget_seed17/result.csv",
}

# All 15 Full-model (random split) run result.csv paths -- same folders
# as pool_fullmodel_predictions.py (which reproduced 42,285 rows / 75.63%).
FULLMODEL_RUN_PATHS = {
    "seed1_run1":  "~/EviDTI/runs/DB_seed1_e100/result.csv",
    "seed1_run2":  "~/EviDTI/runs/DB_seed1_run2/result.csv",
    "seed1_run3":  "~/EviDTI/runs/DB_seed1_run3/result.csv",
    "seed6_run1":  "~/EviDTI/runs/DB_seed6_run1/result.csv",
    "seed6_run2":  "~/EviDTI/runs/DB_seed6_run2/result.csv",
    "seed6_run3":  "~/EviDTI/runs/DB_seed6_run3/result.csv",
    "seed42_run1": "~/EviDTI/runs/DB_seed42_e100/result.csv",
    "seed42_run2": "~/EviDTI/runs/DB_seed42_run2/result.csv",
    "seed42_run3": "~/EviDTI/runs/DB_seed42_run3/result.csv",
    "seed7_run1":  "~/EviDTI/runs/DB_seed7/result.csv",
    "seed7_run2":  "~/EviDTI/runs/DB_seed7_run2/result.csv",
    "seed7_run3":  "~/EviDTI/runs/DB_seed7_run3/result.csv",
    "seed17_run1": "~/EviDTI/runs/DB_seed17/result.csv",
    "seed17_run2": "~/EviDTI/runs/DB_seed17_run2/result.csv",
    "seed17_run3": "~/EviDTI/runs/DB_seed17_run3/result.csv",
}
FULLMODEL_EXPECTED_N = 42285   # total rows verified by pool_fullmodel_predictions.py

KNOWN_ACC_COLD_DRUG = {1: 64.5655, 6: 74.3880, 42: 73.6599, 7: 72.6281, 17: 62.9590}
KNOWN_ACC_COLD_TARGET = {1: 67.8239, 6: 67.6882, 42: 66.2894, 7: 64.3846, 17: 65.5880}
# ------------------------------------------------------------------


def parse_numpy_array_str(s):
    s = s.strip()
    inner = re.sub(r'^\[|\]$', '', s).strip()
    if inner == '':
        return np.array([])
    return np.array([float(x) for x in inner.split()])


def load_result_csv(path):
    raw = pd.read_csv(path, index_col=0)
    n = raw.shape[1]
    preds = np.zeros(n, dtype=int)
    labels = np.zeros(n, dtype=int)
    uncertainty = np.zeros(n, dtype=float)
    for i in range(n):
        pl = parse_numpy_array_str(str(raw.iloc[3, i]))
        if pl.size < 2:
            raise ValueError(f"{path}: row3 col{i} did not parse into [pred label] (got {pl}).")
        preds[i], labels[i] = int(pl[0]), int(pl[1])
        u = parse_numpy_array_str(str(raw.iloc[4, i]))
        if u.size == 0:
            raise ValueError(f"{path}: row4 col{i} is empty/unparseable.")
        uncertainty[i] = float(u[0])
    return pd.DataFrame({'pred': preds, 'label': labels, 'uncertainty': uncertainty})


def per_run_stats(paths_dict, known_acc, split_name):
    """Compute per-seed/run: accuracy (checked), median u, AUROC(u, error).
    Returns the pooled df (for TP/FP breakdowns) plus arrays of per-run
    median_u and per-run auroc for mean+/-SD reporting."""
    dfs = []
    medians, aurocs = [], []
    # per-run direction checks, to support (or drop) the word "consistently"
    n_fp_gt_tp = n_fn_gt_tn = n_inc_gt_cor = n_valid_fp_tp = n_valid_fn_tn = 0
    print(f"\n===== {split_name}: per-seed/run stats =====")
    for key, path in paths_dict.items():
        df = load_result_csv(path)
        df['correct'] = (df['pred'] == df['label']).astype(int)
        acc = 100 * df['correct'].mean()
        expected = known_acc.get(key) if known_acc else None
        flag = "OK" if expected is not None and abs(acc - expected) < 0.01 else \
               ("n/a" if expected is None else "MISMATCH")
        med_u = df['uncertainty'].median()
        if df['correct'].nunique() == 2:
            auroc = roc_auc_score(1 - df['correct'], df['uncertainty'])
        else:
            auroc = float('nan')
        u_of = lambda p, l: df.loc[(df['pred'] == p) & (df['label'] == l), 'uncertainty']
        tp, tn, fp, fn = u_of(1, 1), u_of(0, 0), u_of(1, 0), u_of(0, 1)
        fp_tp = fn_tn = "n/a"
        if len(fp) > 0 and len(tp) > 0:
            n_valid_fp_tp += 1
            fp_tp = fp.median() > tp.median()
            n_fp_gt_tp += int(fp_tp)
        if len(fn) > 0 and len(tn) > 0:
            n_valid_fn_tn += 1
            fn_tn = fn.median() > tn.median()
            n_fn_gt_tn += int(fn_tn)
        n_inc_gt_cor += int(auroc > 0.5)
        print(f"  {key}: acc={acc:.4f}% (expected={expected}, {flag})  "
              f"median_u={med_u:.4f}  AUROC={auroc:.4f}  n={len(df)}  "
              f"FP>TP={fp_tp}  FN>TN={fn_tn}")
        df['run'] = str(key)
        dfs.append(df)
        medians.append(med_u)
        aurocs.append(auroc)

    pooled = pd.concat(dfs, ignore_index=True)
    medians = np.array(medians)
    aurocs = np.array(aurocs)

    print(f"\n-- {split_name}: summary across {len(paths_dict)} runs --")
    print(f"  total n = {len(pooled)}")
    print(f"  direction consistency (median u): FP>TP in {n_fp_gt_tp}/{n_valid_fp_tp} runs, "
          f"FN>TN in {n_fn_gt_tn}/{n_valid_fn_tn} runs, "
          f"Incorrect>Correct (AUROC>0.5) in {n_inc_gt_cor}/{len(paths_dict)} runs")
    print(f"  median_u:  mean={medians.mean():.4f}  SD={medians.std(ddof=1):.4f}")
    print(f"  AUROC:     mean={aurocs.mean():.4f}  SD={aurocs.std(ddof=1):.4f}")
    print(f"  [pooled median_u across all runs, reference only: {pooled['uncertainty'].median():.4f}]")

    pooled_auroc = roc_auc_score(1 - pooled['correct'], pooled['uncertainty'])
    print(f"  [pooled AUROC (all runs concatenated), reference only -- confounded by "
          f"between-run accuracy/u-scale differences, NOT the primary number: {pooled_auroc:.4f}]")

    return pooled, medians, aurocs


def outcome_breakdown(pooled, split_name):
    pooled = pooled.copy()
    pooled['outcome'] = np.select(
        [
            (pooled['pred'] == 1) & (pooled['label'] == 1),
            (pooled['pred'] == 0) & (pooled['label'] == 0),
            (pooled['pred'] == 1) & (pooled['label'] == 0),
            (pooled['pred'] == 0) & (pooled['label'] == 1),
        ],
        ['TP', 'TN', 'FP', 'FN'], default='?'
    )
    print(f"\n-- {split_name}: pooled Correct/Incorrect and TP/FP/TN/FN uncertainty (reference) --")
    for grp, label in [(1, 'Correct'), (0, 'Incorrect')]:
        sub = pooled.loc[pooled['correct'] == grp, 'uncertainty']
        print(f"  {label}: median={sub.median():.4f}  mean={sub.mean():.4f}  n={len(sub)}")
    u_stat, p_val = mannwhitneyu(
        pooled.loc[pooled['correct'] == 0, 'uncertainty'],
        pooled.loc[pooled['correct'] == 1, 'uncertainty'], alternative='greater')
    print(f"  Mann-Whitney U (Incorrect > Correct), pooled: p={p_val:.4g}")
    for label in ['TP', 'FP', 'TN', 'FN']:
        sub = pooled.loc[pooled['outcome'] == label, 'uncertainty']
        if len(sub) > 0:
            print(f"  {label}: median={sub.median():.4f}  mean={sub.mean():.4f}  n={len(sub)}")
    for a, b in [('FP', 'TP'), ('FN', 'TN')]:
        sa = pooled.loc[pooled['outcome'] == a, 'uncertainty']
        sb = pooled.loc[pooled['outcome'] == b, 'uncertainty']
        if len(sa) > 1 and len(sb) > 1:
            _, p = mannwhitneyu(sa, sb, alternative='greater')
            print(f"  Mann-Whitney U ({a} > {b}): p={p:.4g}")


if __name__ == '__main__':
    pooled_drug, med_drug, auroc_drug = per_run_stats(COLD_DRUG_PATHS, KNOWN_ACC_COLD_DRUG, 'Cold-drug')
    outcome_breakdown(pooled_drug, 'Cold-drug')

    pooled_target, med_target, auroc_target = per_run_stats(COLD_TARGET_PATHS, KNOWN_ACC_COLD_TARGET, 'Cold-target')
    outcome_breakdown(pooled_target, 'Cold-target')

    print("\n===== Cross-split comparison (primary numbers: per-run mean +/- SD) =====")
    if FULLMODEL_RUN_PATHS:
        pooled_full, med_full, auroc_full = per_run_stats(FULLMODEL_RUN_PATHS, None, 'Random split (Full model)')
        if len(pooled_full) != FULLMODEL_EXPECTED_N:
            print(f"  [WARNING] Full-model total n={len(pooled_full)}, expected {FULLMODEL_EXPECTED_N} "
                  f"-- check FULLMODEL_RUN_PATHS before using these numbers.")
        else:
            print(f"  Full-model total n={len(pooled_full)} matches expected {FULLMODEL_EXPECTED_N} [OK]")
        outcome_breakdown(pooled_full, 'Random split (Full model)')
        print(f"\n  median_u   -- Random: {med_full.mean():.4f}+/-{med_full.std(ddof=1):.4f}  "
              f"Cold-drug: {med_drug.mean():.4f}+/-{med_drug.std(ddof=1):.4f}  "
              f"Cold-target: {med_target.mean():.4f}+/-{med_target.std(ddof=1):.4f}")
        print(f"  AUROC      -- Random: {auroc_full.mean():.4f}+/-{auroc_full.std(ddof=1):.4f}  "
              f"Cold-drug: {auroc_drug.mean():.4f}+/-{auroc_drug.std(ddof=1):.4f}  "
              f"Cold-target: {auroc_target.mean():.4f}+/-{auroc_target.std(ddof=1):.4f}")
    else:
        print("  [FULLMODEL_RUN_PATHS is empty -- fill in all 15 Full-model run result.csv paths "
              "to get a like-for-like per-run comparison. A single pooled-42285-row AUROC is NOT "
              "an apples-to-apples comparison to the 5-seed cold-split numbers above.]")
        print(f"  Cold-drug:   median_u={med_drug.mean():.4f}+/-{med_drug.std(ddof=1):.4f}  "
              f"AUROC={auroc_drug.mean():.4f}+/-{auroc_drug.std(ddof=1):.4f}")
        print(f"  Cold-target: median_u={med_target.mean():.4f}+/-{med_target.std(ddof=1):.4f}  "
              f"AUROC={auroc_target.mean():.4f}+/-{auroc_target.std(ddof=1):.4f}")

    print("\n  NOTE: median_u is confounded across splits by each run being an independently\n"
          "  trained model with its own evidence scale -- report as descriptive, not a claim of\n"
          "  causal effect. AUROC is the metric whose value doesn't depend on that scale, since\n"
          "  it only measures within-run rank correlation between u and misclassification.")
