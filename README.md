# EviDTI Reproduction

This is a reproduction of EviDTI: Evidential Deep Learning for Guided Drug-Target Interaction Prediction.

Original paper: https://doi.org/10.1038/s41467-025-62235-6  
Original source code: https://github.com/zhaoyanpeng208/EviDTI

## Catalogue

- [Environment](#Environment)
- [Dataset](#Dataset)
- [Feature Extraction](#Feature-Extraction)
- [Train and Evaluate](#Train-and-Evaluate)

## Environment

All experiments were run on Google Colab (T4 GPU). The original EviDTI source code must be downloaded first and placed in Google Drive before running any notebooks:

```bash
git clone https://github.com/zhaoyanpeng208/EviDTI.git
```

Or download the zip file from the repository and upload it to Google Drive.

The following dependencies are required and installed within each notebook.

For Drug 2D feature extraction:
```
rdkit, torch_geometric, yacs, openbabel-wheel
```

For Drug 3D feature extraction, PaddlePaddle and PaddleHelix are required.
Install Miniconda (Python 3.8) and run:
```bash
pip install scikit-learn networkx pandas rdkit paddlepaddle-gpu==2.5.2
pip install paddlehelix --no-deps
pip install pgl
```
Then clone the full PaddleHelix source:
```bash
git clone https://github.com/PaddlePaddle/PaddleHelix.git
cp -r PaddleHelix/pahelix ./pahelix
```

For protein feature extraction:
```
transformers, sentencepiece, accelerate, torch
```

For model training and evaluation:
```
rdkit, yacs, torch_geometric, tensorboard
```

## Dataset

The DrugBank and Davis datasets used in this study are available at:  
https://zenodo.org/records/14056305

## Feature Extraction

Run the following notebooks in order:
```
01_Extract_Drug_2D_Features.ipynb
02_Extract_Drug_3D_Features.ipynb
03_Extract_Protein_Features.ipynb
```

## Train and Evaluate

DrugBank dataset:
```
04_Train_and_Evaluate_DrugBank.ipynb
```

Davis dataset:
```
06_Train_and_Evaluate_Davis.ipynb
```
