"""
HPC clean version of main_davis.py.

Changes relative to the author's original:
1. Imports from function_hpc.py / davis_model_hpc.py / davis_solver_hpc.py
   (the fixed versions) instead of the originals.
2. Data paths point to the actual files on ASPIRE2A (author-provided
   Davis features -- not self-extracted, so no numpy-version
   compatibility issue is expected here, unlike DrugBank's 2D/3D
   features).
3. np.random.seed(SEED) is read from cfg instead of hardcoded to 3
   (the original hardcoded value), and torch.manual_seed/
   cuda.manual_seed_all are added (the original never called these,
   same gap as main_drugbank.py -- see main_drugbank_hpc.py's notes).
4. MAX_EPOCH is controlled via the yaml (configs/davis_hpc.yaml sets
   500 + early stopping, NOT the fixed 10 used to produce the
   originally reported Davis results -- that 10-epoch run is what
   caused the evidence collapse discussed separately).
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

from davis_configs_hpc import get_cfg_defaults
from davis_model_hpc import LightAttention
from davis_solver_hpc import Solver
from function_hpc import davis_graph_data
import os
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="EviDTI for dti prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S')
args = parser.parse_args()

BASE = '/home/users/nus/e1561459/EviDTI/dataset/davis'


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    data_list = davis_graph_data(
        root=f'{BASE}/',
        t_1D_path=f'{BASE}/davis_t_feature.npy',
        d_2D_path=f'{BASE}/davis_d_2D_features.npy',
        d_3D_path=f'{BASE}/davis_d_3d_feature.npy',
        label_file=f'{BASE}/davis_total_cid_unid.csv',
        metadata_path=f'{BASE}/davis_total_cid_unid.csv'
    )

    seed = cfg.SOLVER.SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using seed: {seed}")

    shuffled_indices = np.random.permutation(len(data_list))
    train_idx = shuffled_indices[:int(0.8 * len(data_list))]
    val_idx = shuffled_indices[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
    test_idx = shuffled_indices[int(0.9 * len(data_list)):]

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
