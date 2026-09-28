# EviDTI-Reproduction

Code accompanying the manuscript **"Uncertainty-aware deep learning for drug-target interaction prediction"** (Ruojia You, Nguyen Quoc Khanh Le, Matthew Chin Heng Chua).

This study uses the evidential multimodal DTI framework **EviDTI** (Zhao et al., *Nat Commun* 2025, 16:6915) as a test platform to characterize the behavior of evidential uncertainty in drug–target interaction prediction. It does not propose a new architecture.

## Repository structure

```
EviDTI-Reproduction/
├── colab/   Initial experiments on Google Colab (NVIDIA T4)
└── hpc/     Main experiments on the NSCC ASPIRE2A HPC cluster
```

| Folder | What it contains | Results in the manuscript |
|---|---|---|
| `colab/` | Feature extraction (drug 2D/3D, protein), 30-epoch DrugBank training, 10-epoch Davis training, visualization | Table 1 and Table 5 "Colab" rows; Figure 2(a–c); Figure 3(a–e) |
| `hpc/` | Full model (15 runs), ablations (w/o 2D, w/o 3D), cold-drug and cold-target splits, Davis (500 epochs), uncertainty baselines, and analysis scripts | All other results, including Tables 1–7 (HPC rows), Figures 2(d–f), 3(f–h), 4, 5, and Supplementary Tables S1 and S3 |

See the README inside each folder for details.

## Data

The DrugBank and Davis drug–target pairs and labels are the preprocessed files released with the original EviDTI study. Feature files are not included in this repository because of their size. The `colab/` notebooks regenerate the DrugBank drug and protein features.

## Reference

Zhao Y, Xing Y, Zhang Y, et al. Evidential deep learning-based drug-target interaction prediction. *Nat Commun*. 2025;16(1):6915. doi:10.1038/s41467-025-62235-6
