"""
Temperature scaling baseline (R1-3 / R2-5).

Fits a single scalar temperature T on the VALIDATION set (never the test
set), then applies it to the TEST set's predicted probabilities.

Usage:
    python temperature_scaling.py <val_result_csv> <test_result_csv> <output_csv>
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

VAL_CSV = sys.argv[1]
TEST_CSV = sys.argv[2]
OUTPUT_CSV = sys.argv[3] if len(sys.argv) > 3 else 'temperature_scaling_results.csv'

EPS = 1e-7


def parse_numpy_array_str(s):
    """Parse strings like '[1 1]' (numpy array str format, space-separated,
    not comma-separated) into a list of floats."""
    s = s.strip('[]')
    return [float(x) for x in s.split()]


def load_test_result_csv(path):
    """
    result.csv from drugbank_solver_hpc.py's evaluation() is stored
    row-wise (transposed relative to a normal table). Row 2 = prob_list,
    row 3 = [pred, label] pairs, stored as numpy-array-style strings
    (space-separated, e.g. "[1 0]") -- NOT valid Python list literals,
    so ast.literal_eval fails on them; use parse_numpy_array_str instead.
    """
    df = pd.read_csv(path, index_col=0)
    probs = pd.to_numeric(df.iloc[2], errors='coerce').values
    p_rows = df.iloc[3].apply(parse_numpy_array_str).tolist()
    p_arr = np.array(p_rows)
    labels = p_arr[:, 1]
    return probs, labels


def load_val_csv(path):
    """Simple two-column CSV from export_val_predictions.py."""
    df = pd.read_csv(path)
    return df['Predicted_Probability'].values, df['True_Label'].values


def prob_to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logit_to_prob(z):
    return 1 / (1 + np.exp(-z))


def nll_loss(T, logits, labels):
    scaled_probs = logit_to_prob(logits / T)
    scaled_probs = np.clip(scaled_probs, EPS, 1 - EPS)
    return -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))


print("Loading validation set predictions...")
val_probs, val_labels = load_val_csv(VAL_CSV)
val_logits = prob_to_logit(val_probs)

print("Fitting temperature on validation set...")
res = minimize_scalar(nll_loss, bounds=(0.05, 10.0), method='bounded',
                       args=(val_logits, val_labels))
T_optimal = res.x
print(f"Optimal temperature T = {T_optimal:.4f}")
print(f"Val NLL before scaling (T=1): {nll_loss(1.0, val_logits, val_labels):.4f}")
print(f"Val NLL after scaling (T={T_optimal:.4f}): {nll_loss(T_optimal, val_logits, val_labels):.4f}")

print("\nApplying fitted temperature to test set...")
test_probs, test_labels = load_test_result_csv(TEST_CSV)
test_logits = prob_to_logit(test_probs)
test_probs_scaled = logit_to_prob(test_logits / T_optimal)

df = pd.DataFrame({
    'True_Label': test_labels,
    'Raw_Prob': test_probs,
    'Temperature_Scaled_Prob': test_probs_scaled,
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} test samples with scaled probabilities to {OUTPUT_CSV}")
print(f"Test NLL before scaling: {nll_loss(1.0, test_logits, test_labels):.4f}")
print(f"Test NLL after scaling: {nll_loss(T_optimal, test_logits, test_labels):.4f}")
