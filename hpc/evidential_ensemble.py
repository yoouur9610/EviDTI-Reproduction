"""
Evidential ensemble baseline (a simplified approximation to a standard
deep ensemble, given the current computational budget -- see Methods
2.4). Uses the three independent repeats already trained under seed=6
(DB_seed6_run1/2/3), which share an identical train/val/test split
(since np.random.seed(6) + permutation is deterministic), so their
result.csv files' rows correspond to the SAME test samples in the SAME
order and can be aligned by position.

The variance across the 3 models' predicted probabilities for each
sample is used as the ensemble disagreement / uncertainty estimate.

Usage:
    python evidential_ensemble.py
"""
import pandas as pd
import numpy as np

RUNS_DIR = '/home/users/nus/e1561459/EviDTI/runs'
RUN_FOLDERS = ['DB_seed6_run1', 'DB_seed6_run2', 'DB_seed6_run3']
OUTPUT_CSV = 'evidential_ensemble_results.csv'


def parse_numpy_array_str(s):
    s = s.strip('[]')
    return [float(x) for x in s.split()]


def load_result_csv(path):
    df = pd.read_csv(path, index_col=0)
    probs = pd.to_numeric(df.iloc[2], errors='coerce').values
    p_rows = df.iloc[3].apply(parse_numpy_array_str).tolist()
    p_arr = np.array(p_rows)
    labels = p_arr[:, 1]
    return probs, labels


all_probs = []
all_labels = []

for folder in RUN_FOLDERS:
    path = f"{RUNS_DIR}/{folder}/result.csv"
    probs, labels = load_result_csv(path)
    all_probs.append(probs)
    all_labels.append(labels)
    print(f"{folder}: {len(probs)} samples loaded")

# Sanity check: confirm all 3 runs share the same test set (same labels
# in the same order), since this ensemble approach REQUIRES that.
n_samples = len(all_labels[0])
labels_match = all(np.array_equal(all_labels[0], all_labels[i]) for i in range(1, 3))
print(f"\n三次run的true label是否完全一致（确认共用同一份测试集划分）: {labels_match}")
if not labels_match:
    print("⚠️ 警告：三次run的测试集labels不一致，说明测试集划分不同，"
          "这个ensemble方法的前提假设不成立，结果不可信！")

all_probs = np.array(all_probs)  # shape: [3, n_samples]
labels = np.array(all_labels[0])  # use run1's labels as reference

ensemble_mean_prob = all_probs.mean(axis=0)
ensemble_disagreement = all_probs.std(axis=0, ddof=1)  # uncertainty estimate
ensemble_pred = (ensemble_mean_prob >= 0.5).astype(int)

accuracy = (ensemble_pred == labels).mean() * 100
mean_disagreement = ensemble_disagreement.mean()

print(f"\n===== Evidential Ensemble (n=3 models, seed=6) =====")
print(f"样本数: {n_samples}")
print(f"Ensemble mean-prediction accuracy: {accuracy:.2f}%")
print(f"Mean disagreement (std across 3 models): {mean_disagreement:.4f}")

# Also report accuracy for correct vs incorrect predictions' disagreement
correct = (ensemble_pred == labels)
print(f"\n正确预测样本的平均分歧度: {ensemble_disagreement[correct].mean():.4f}")
print(f"错误预测样本的平均分歧度: {ensemble_disagreement[~correct].mean():.4f}")

df = pd.DataFrame({
    'True_Label': labels,
    'Ensemble_Mean_Prob': ensemble_mean_prob,
    'Ensemble_Disagreement': ensemble_disagreement,
    'Ensemble_Pred': ensemble_pred,
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n已保存: {OUTPUT_CSV}")
