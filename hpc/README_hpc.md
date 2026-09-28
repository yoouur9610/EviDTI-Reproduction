# HPC experiments

This folder contains the code used for the main experiments on the NSCC ASPIRE2A cluster.

## Environment

- Python 3.8.20
- PyTorch 1.12.0, CUDA 11.3
- PyTorch Geometric 2.2.0
- RDKit 2020.09.1

```bash
conda env create -f environment.yml
conda activate evidti
```

## Files

All files are kept in this folder because the scripts import each other.

**Training entry points**

| Script | Experiment | Manuscript |
|---|---|---|
| `main_drugbank_hpc.py` | DrugBank full model, random split | Tables 1, 2, 3, 4, 6, 7; Figs 2(d–f), 3(f–h), 4, 5; Table S1 |
| `main_drugbank_ablation2d_hpc.py` | DrugBank w/o 2D | Table 4 |
| `main_drugbank_ablation3d_hpc.py` | DrugBank w/o 3D | Table 4 |
| `main_drugbank_coldsplit_hpc.py` | DrugBank cold-drug split | Tables 6a, 6b; Table S3 |
| `main_drugbank_coldtarget_hpc.py` | DrugBank cold-target split | Tables 6a, 6b; Table S3 |
| `main_davis_hpc.py` | Davis, random split | Table 5 |

**Models and training loops**

- `drugbank_model_hpc.py`: full EviDTI model
- `drugbank_model_ablation2d_hpc.py`, `drugbank_model_ablation3d_hpc.py`: identical to the full model, except that the output of the 2D or 3D branch is replaced with zeros before feature concatenation
- `davis_model_hpc.py`: EviDTI model for Davis
- `drugbank_solver_hpc.py`, `davis_solver_hpc.py`: training, early stopping, and evaluation

**Shared utilities**

- `function_hpc.py`: dataset classes (DrugBank lazy loading; Davis in-memory)
- `utils.py`: evidential Dirichlet loss (Sensoy et al., 2018)
- `compound_encode.py`: atom, bond, and bond-angle encoders (from GeoGNN)
- `drugbank_configs_hpc.py`, `davis_configs_hpc.py`: default hyperparameters

**Uncertainty baselines** (Table 2)

- `main_drugbank_softmax_hpc.py`, `drugbank_model_softmax_hpc.py`, `drugbank_solver_softmax_hpc.py`: deterministic softmax classifier trained with cross-entropy (entry point, model, and training loop)
- `mc_dropout_analysis.py`: MC Dropout, 30 stochastic forward passes with dropout kept active
- `evidential_ensemble.py`: disagreement among three evidential models trained under the same seed
- `temperature_scaling.py`: single temperature fitted on the evidential model's validation outputs

**Analysis**

- `pool_coldsplit_uncertainty.py`: per-run and pooled uncertainty statistics, error-detection AUROC, and FP>TP / FN>TN counts (Table 6b, Table S3)
- `table2_uncertainty_medians.py`: median uncertainty of correct and incorrect predictions for the baselines (Table 2)

**Example config and job script**

- `drugbank_hpc.yaml`, `davis_hpc.yaml`: example configs
- `drugbank_hpc.pbs`: example PBS job script

## Before running

The data paths are hard-coded in each `main_*.py` file (the `BASE` variable) and in the `SOFTMAX_RESULT_CSV` path in `table2_uncertainty_medians.py`. Change them to your own locations.

## Running

```bash
python main_drugbank_hpc.py --cfg drugbank_hpc.yaml --data drugbank
python main_davis_hpc.py --cfg davis_hpc.yaml --data davis
```

On ASPIRE2A, submit with `qsub drugbank_hpc.pbs`.

To run another experiment, replace the script name with the corresponding `main_*.py` file. To reproduce each run, set `SEED` in the yaml file.

| Setting | DrugBank | Davis |
|---|---|---|
| Seeds | 1, 6, 7, 17, 42 | 1, 6, 7, 17, 42 |
| Repeats per seed | 3 (full model); 1 (ablations, cold splits) | 1 |
| Max epochs | 100 | 500 |
| Early stopping patience | 30 | 50 |
| Batch size | 32 | 32 |
| Learning rate | 3 × 10⁻⁴, ReduceLROnPlateau (factor 0.5, patience 10, min 1.5 × 10⁻⁴) | Same |
| KL coefficient | min(0.3, (epoch + 1) / 50) | Fixed 0.1 |

Each run creates `runs/la_test1_<PBS job ID>/`, which contains:

- `checkpoint.pth`: the best model, selected by validation accuracy
- `result.csv`: test-set predictions

Test metrics are also appended to `matrix_result.csv` in the order: accuracy, recall, precision, MCC, F1, AUC, AUPR.

### `result.csv` layout

Read it with `pd.read_csv(path, index_col=0)`. Each column is one test sample, and each row is:

| Row | Content |
|---|---|
| 0 | target ID |
| 1 | drug ID (SMILES) |
| 2 | predicted probability of interaction, p̂ |
| 3 | `[predicted label, true label]` |
| 4 | uncertainty u = K / S |
| 5 | duplicate of row 4 |
| 6 | evidence |
| 7 | belief mass |

## Changes relative to the original EviDTI code

1. **Bond-angle GNN layers.** In the bond-angle branch, layers 5 and 6 reused the weights of layer 3. Each layer now has its own weights.
2. **Bond-angle graph batching.** Node indices were not offset when sub-graphs were concatenated into a batch. The offset is now applied.
3. **Loss reduction.** `loss.sum()` was changed to `loss.mean()`.
4. **Label precision.** One-hot labels are cast to float32 instead of float16.
5. **KL coefficient (DrugBank).** A dynamic KL annealing schedule is used; the value is given in the table above.
6. **Learning-rate scheduler.** The original scheduler was created but never stepped. It was replaced with ReduceLROnPlateau.
7. **Seeding.** `torch.manual_seed` and `torch.cuda.manual_seed_all` were added.
8. **Run directories.** Run directories are named by PBS job ID, so that concurrent jobs do not overwrite each other.

**Known inherited behavior.** In the Davis model (`davis_model_hpc.py`), the output of the bond-angle branch is computed but not passed to the atom-level layers. This follows the original Davis code and was left unchanged. The Davis results therefore do not use bond-angle information.
