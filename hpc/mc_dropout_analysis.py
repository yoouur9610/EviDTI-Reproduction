"""
MC Dropout uncertainty baseline (R1-3 / R2-5).

Loads an existing Full-model checkpoint (already trained with the
evidential loss -- no separate training needed for this baseline) and
runs N stochastic forward passes per test sample by keeping Dropout
layers active at inference time, while BatchNorm layers remain in eval
mode (using their stored running statistics, NOT the current batch's
statistics -- mixing the two would corrupt the predictions). The
variance across the N passes' predicted probabilities is used as the
MC Dropout uncertainty estimate.

BUG FIX (found after an initial run produced near-random ~51% accuracy):
the original version used `SubsetRandomSampler(test_idx)` to build
test_loader. SubsetRandomSampler reshuffles sample order EVERY time the
DataLoader is iterated -- this is fine for a single pass, but is wrong
here, since we iterate the SAME loader 30 times and assume the sample
at a given array position is the same sample across all 30 passes
(labels/ids are only recorded once, on pass 0). Because the order
silently changed on every pass, `all_probs[i]` at a given column index
corresponded to a DIFFERENT, effectively random sample each time,
so averaging across passes mixed together unrelated samples' predictions
-- producing meaningless, near-random accuracy. Fixed by using a
`Subset` + a plain DataLoader with a fixed (non-shuffled) sample order,
so the same sample lands in the same position on every pass.

Usage:
    python mc_dropout_analysis.py <checkpoint_path> <yaml_path> <n_passes> <output_csv>
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from drugbank_configs_hpc import get_cfg_defaults
from drugbank_model_hpc import LightAttention
from function_hpc import bond_angle_graph_data

CHECKPOINT_PATH = sys.argv[1]
YAML_PATH = sys.argv[2]
N_PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUTPUT_CSV = sys.argv[4] if len(sys.argv) > 4 else 'mc_dropout_results.csv'

BASE = '/home/users/nus/e1561459/EviDTI/dataset/drugbank'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = get_cfg_defaults()
cfg.merge_from_file(YAML_PATH)
seed = cfg.SOLVER.SEED

print(f"Loading dataset...")
data_list = bond_angle_graph_data(
    root=f'{BASE}/',
    t_1D_path=f'{BASE}/protein_local/EviDTI_Protein_Features_Unpooled',
    d_2D_path=f'{BASE}/feature_backup/drug_embeddings_pooled_compat.npy',
    d_3D_path=f'{BASE}/feature_backup/bisai_HY_QS_3D_emb_64_compat.npy',
    label_file=f'{BASE}/total_cid_unid_csv.csv',
    metadata_path=f'{BASE}/metadata.csv'
)

# Reproduce the SAME test split used during training for this seed
np.random.seed(seed)
shuffled_indices = np.random.permutation(len(data_list))
test_idx = shuffled_indices[int(0.9 * len(data_list)):]

# FIXED: use Subset + shuffle=False so sample order is IDENTICAL across
# all N_PASSES iterations of this loader (only Dropout should introduce
# randomness here, not the data order).
test_subset = Subset(data_list, test_idx.tolist())
test_loader = DataLoader(test_subset, batch_size=cfg.SOLVER.BATCH_SIZE,
                          drop_last=False, shuffle=False)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model = LightAttention(cfg).to(device)
if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))
elif isinstance(checkpoint, nn.Module):
    model.load_state_dict(checkpoint.state_dict())


def set_dropout_train_bn_eval(m):
    """Keep the whole model in eval mode (so BatchNorm uses stored running
    stats), but flip only Dropout layers back to train mode so they
    actually drop units during this forward pass."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


set_dropout_train_bn_eval(model)

print(f"Running {N_PASSES} stochastic forward passes over the test set...")
all_probs = []  # shape: [N_PASSES, n_test_samples]
sample_ids = []
labels_list = []

with torch.no_grad():
    for pass_idx in range(N_PASSES):
        pass_probs = []
        pass_labels = [] if pass_idx == 0 else None
        pass_ids = [] if pass_idx == 0 else None
        for batch in test_loader:
            metadata = batch.metadata
            sequence_lengths = metadata['length'][:, None].to(device)
            mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:, None]
            mask = mask.to(device)

            outputs = model(batch, mask=mask, sequence_lengths=sequence_lengths)
            S = torch.sum(outputs, dim=1, keepdim=True)
            probs = (outputs[:, 1:2] / S).cpu().numpy().flatten()
            pass_probs.extend(probs.tolist())

            if pass_idx == 0:
                pass_labels.extend(batch.l.cpu().numpy().tolist())
                pass_ids.extend([f"{metadata['d_id'][i]}_{metadata['id'][i]}" for i in range(len(batch.l))])

        all_probs.append(pass_probs)
        if pass_idx == 0:
            labels_list = pass_labels
            sample_ids = pass_ids
        print(f"  pass {pass_idx + 1}/{N_PASSES} done")

all_probs = np.array(all_probs)  # [N_PASSES, n_samples]
mean_probs = all_probs.mean(axis=0)
std_probs = all_probs.std(axis=0, ddof=1)  # this is the MC Dropout uncertainty estimate
pred_labels = (mean_probs >= 0.5).astype(int)

df = pd.DataFrame({
    'sample_id': sample_ids,
    'True_Label': labels_list,
    'MCDropout_Mean_Prob': mean_probs,
    'MCDropout_Std': std_probs,
    'Pred_Label': pred_labels,
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} samples to {OUTPUT_CSV}")
print(f"Mean uncertainty (std across passes): {std_probs.mean():.4f}")
print(f"Accuracy (mean-prob thresholded at 0.5): "
      f"{(df['Pred_Label'] == df['True_Label']).mean() * 100:.2f}%")
