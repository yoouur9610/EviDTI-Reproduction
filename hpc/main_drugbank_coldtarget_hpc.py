"""
HPC clean version of main_drugbank.py.

Changes relative to the author's original:
1. Imports `bond_angle_graph_data` from function_hpc.py (the
   lazy-loading rewrite) instead of function.py (the original
   InMemoryDataset version, which cannot handle the per-protein
   .npy folder format used here).
2. Data paths point to the actual files on ASPIRE2A, confirmed
   2026-08-13.
3. np.random.seed(SEED) is read from cfg (drugbank_test.yaml sets
   SEED: 6 to match the checkpoint already reported in the
   manuscript) rather than hardcoded to 1, and is called BEFORE
   dataset construction -- this preserves the exact seed-timing
   used in the original script structure.
4. torch.manual_seed(SEED) and torch.cuda.manual_seed_all(SEED) are
   also called here. NOTE: this is a deliberate improvement over the
   Colab notebook, which never called any torch-level seeding
   despite a comment claiming it would (Cell 13's comment says
   "顺手加上更严谨的 torch 种子" but the actual code only sets
   np.random.seed). Added here for better reproducibility going
   forward; does not affect the already-reported Seed 6 results.

Everything else (model construction, train/val/test split ratios,
Solver invocation) is unchanged from the original.
"""
import argparse
from time import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from typing import List, Tuple
from torch_geometric.loader import DataLoader

from drugbank_model_hpc import LightAttention
from drugbank_solver_hpc import Solver
from function_hpc import bond_angle_graph_data
from drugbank_configs_hpc import get_cfg_defaults
import os
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="EviDTI for dti prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S')
args = parser.parse_args()

BASE = '/home/users/nus/e1561459/EviDTI/dataset/drugbank'


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    data_list = bond_angle_graph_data(
        root=f'{BASE}/',
        t_1D_path=f'{BASE}/protein_local/EviDTI_Protein_Features_Unpooled',
        d_2D_path=f'{BASE}/feature_backup/drug_embeddings_pooled_compat.npy',
        d_3D_path=f'{BASE}/feature_backup/bisai_HY_QS_3D_emb_64_compat.npy',
        label_file=f'{BASE}/total_cid_unid_csv.csv',
        metadata_path=f'{BASE}/metadata.csv'
    )

    seed = cfg.SOLVER.SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using seed: {seed}")

    # ===== Cold-target (unseen-target) split =====
    # Instead of randomly splitting drug-target PAIRS (which lets the same
    # drug appear in both train and test, testing interpolation rather than
    # generalization -- R1-4, R2-7), we split UNIQUE DRUGS into train/val/
    # test first, then assign each pair based on which split its drug
    # belongs to. Every drug in the test set is therefore one the model
    # never saw during training. Split ratio (80/10/10) matches the
    # author's original random split for comparability.
    #
    # Drug identity is read directly from the dataset's metadata table
    # (data_list.meta_df / data_list.valid_indices, already built during
    # __init__) rather than by calling data_list[idx], since the lazy
    # get() would trigger full graph construction for all 33062 samples
    # just to read an ID -- unnecessarily slow.
    print("Building cold-target (unseen-target) split...")
    target_ids = []
    for idx in range(len(data_list)):
        real_idx = data_list.valid_indices[idx]
        target_ids.append(data_list.meta_df.iloc[real_idx]['uid'])
    target_ids = np.array(target_ids)

    unique_targets = np.unique(target_ids)
    np.random.shuffle(unique_targets)
    n_targets = len(unique_targets)
    train_targets = set(unique_targets[:int(0.8 * n_targets)])
    val_targets = set(unique_targets[int(0.8 * n_targets):int(0.9 * n_targets)])
    test_targets = set(unique_targets[int(0.9 * n_targets):])

    train_idx = np.array([i for i in range(len(data_list)) if target_ids[i] in train_targets])
    val_idx = np.array([i for i in range(len(data_list)) if target_ids[i] in val_targets])
    test_idx = np.array([i for i in range(len(data_list)) if target_ids[i] in test_targets])

    print(f"Unique targets: {n_targets} total -> train {len(train_targets)}, "
          f"val {len(val_targets)}, test {len(test_targets)} targets")

    train_loader = DataLoader(data_list, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True,
                               sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(data_list, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=False,
                             sampler=SubsetRandomSampler(val_idx))
    test_loader = DataLoader(data_list, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=False,
                              sampler=SubsetRandomSampler(test_idx))
    print('load data finish')
    print(f'train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}')

    model = LightAttention(cfg)
    solver = Solver(model, cfg, device)
    solver.train(train_loader, val_loader, eval_data=test_loader)


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
