import torch
import numpy as np
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


device = 'cuda'

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

# evidential classification
def dirichlet_loss(y, alphas, lam=1):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al
    :y: labels to predict
    :alphas: predicted parameters for Dirichlet
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    def KL(alpha):
        """
        Compute KL for Dirichlet defined by alpha to uniform dirichlet
        :alpha: parameters for Dirichlet

        :return: KL
        """
        beta = torch.ones_like(alpha)
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)

        ln_alpha = torch.lgamma(S_alpha)-torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)

        # digamma terms
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)

        # KL
        kl = ln_alpha + ln_beta + torch.sum((alpha - beta)*(dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
        return kl


    # Hard code to 2 classes per task, since this assumption is already made
    # for the existing chemprop classification tasks
    num_classes = 2
    num_tasks = 1

    # y_one_hot = torch.eye(num_classes)[y.long()]
    y_one_hot = y
    if y.is_cuda:
        y_one_hot = y_one_hot.to(device)

    # alphas = torch.reshape(alphas, (alphas.shape[0], num_tasks, num_classes))
    alphas = torch.reshape(alphas, (alphas.shape[0], num_classes))
    # SOS term
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum(torch.pow((y_one_hot - p), 2), dim=-1, keepdim=True)
    B = torch.sum((p*(1 - p)) / (S+1), dim=-1, keepdim=True)
    SOS = A + B

    # KL
    alpha_hat = y_one_hot + (1-y_one_hot)*alphas
    KL = lam * KL(alpha_hat)

    #loss = torch.mean(SOS + KL)
    loss = SOS + KL
    loss = torch.mean(loss, dim=-1)
    return loss

import time
# drugbank proprocess
def proprocess(d_2D_path, t_1D_path, d_3D_path,label_file_path,metadata_path):
    d_2D_feature = np.load(d_2D_path, allow_pickle=True)
    t_1D_feature = np.load(t_1D_path, allow_pickle=True)
    # t_1D_feature = []
    d_3D_feature = np.load(d_3D_path, allow_pickle=True)
    # start = time.time()
    # start_end = start
    # t_3D_feature = np.load(t_3D_path, allow_pickle=True)
    t_3D_feature = []
    # end = time.time() - start_end
    # print(end)
    label_file = pd.read_csv(label_file_path)
    label = label_file['label'].tolist()

    t_metadata_path = metadata_path
    t_metadata = pd.read_csv(t_metadata_path)
    t_id = t_metadata['uid'].tolist()
    t_seq = t_metadata['seq'].tolist()
    d_id = t_metadata['cid']
    d_smile = t_metadata['smile']
    #molecule GCN
    # d_2D_feature = d_smile
    #integer
    # for seq in t_seq:
    #     integer_feature = integer_label_protein(seq)
    #     t_1D_feature.append(integer_feature)
    metadata_list = []
    for i in range(len(t_id)):
        metadata = {'id': t_id[i],
                    'sequence': str(t_seq[i]),
                    'length': len(t_seq[i]),
                    'd_id': d_id[i],
                    'smiles': d_smile[i]
                    }
        metadata_list.append(metadata)
    return d_2D_feature,t_1D_feature,d_3D_feature, label, metadata_list

from tqdm import tqdm


def KIBA_proprocess(d_2D_path, t_1D_path, d_3D_path,label_file_path,metadata_path):
    d_2D_feature = np.load(d_2D_path, allow_pickle=True)
    t_1D_feature = np.load(t_1D_path, allow_pickle=True)
    d_3D_feature = np.load(d_3D_path, allow_pickle=True)
    start = time.time()
    start_end = start
    # t_3D_feature = np.load(t_3D_path, allow_pickle=True)
    end = time.time() - start_end
    print(end)
    label_file = pd.read_csv(label_file_path)
    label = label_file['label'].tolist()

    t_metadata_path = metadata_path
    t_metadata = pd.read_csv(t_metadata_path)

    t_id = t_metadata['uid'].tolist()
    t_seq = t_metadata['seq'].tolist()
    d_id = t_metadata['cid']
    d_smile = t_metadata['smiles']
    metadata_list = []
    t_1D_feature_list = []
    d_3D_feature_list = []
    label_list = []
    for i in tqdm(range(len(t_id))):
        if len(d_smile[i])<300:
            metadata = {'id': t_id[i],
                        'sequence': str(t_seq[i]),
                        'length': len(t_seq[i]),
                        'd_id': d_id[i],
                        'smiles': d_smile[i]
                        }
            metadata_list.append(metadata)
            t_1D_feature_list.append(t_1D_feature[i])
            d_3D_feature_list.append(d_3D_feature[i])
            label_list.append(label[i])

    return d_2D_feature,t_1D_feature_list,d_3D_feature_list, label_list, metadata_list

def davis_proprocess(d_2D_path, t_1D_path, d_3D_path,label_file,metadata_path):
    d_2D_feature = np.load(d_2D_path, allow_pickle=True)
    t_1D_feature = np.load(t_1D_path, allow_pickle=True)
    d_3D_feature = np.load(d_3D_path, allow_pickle=True)
    label_file = pd.read_csv(label_file)
    label = label_file['label'].tolist()

    t_metadata_path = metadata_path
    t_metadata = pd.read_csv(t_metadata_path)

    t_id = t_metadata['uid'].tolist()
    t_seq = t_metadata['seq'].tolist()
    d_id = t_metadata['cid']
    d_smile = t_metadata['smiles']
    metadata_list = []
    t_1D_feature_list = []
    d_3D_feature_list = []
    label_list = []
    for i in tqdm(range(len(t_id))):
        if len(d_smile[i])<300:
            metadata = {'id': t_id[i],
                        'sequence': str(t_seq[i]),
                        'length': len(t_seq[i]),
                        'd_id': d_id[i],
                        'smiles': d_smile[i]
                        }
            metadata_list.append(metadata)
            t_1D_feature_list.append(t_1D_feature[i])
            d_3D_feature_list.append(d_3D_feature[i])
            label_list.append(label[i])

    return d_2D_feature,t_1D_feature_list,d_3D_feature_list, label_list, metadata_list

def case_proprocess(d_2D_path, t_1D_path, d_3D_path,label_file,metadata_path):
    d_2D_feature = np.load(d_2D_path, allow_pickle=True)
    t_1D_feature = np.load(t_1D_path, allow_pickle=True)
    d_3D_feature = np.load(d_3D_path, allow_pickle=True)
    label_file = pd.read_csv(label_file)
    label = label_file['label'].tolist()

    t_metadata_path = metadata_path
    t_metadata = pd.read_csv(t_metadata_path)

    t_id = t_metadata['uid'].tolist()
    t_seq = t_metadata['seq'].tolist()
    d_id = t_metadata['cid']
    d_smile = t_metadata['SMILES']
    metadata_list = []
    t_1D_feature_list = []
    d_3D_feature_list = []
    label_list = []
    for i in tqdm(range(len(t_id))):
        # if len(d_smile[i])<300:
        metadata = {'id': t_id[i],
                    'sequence': str(t_seq[i]),
                    'length': len(t_seq[i]),
                    'd_id': d_id[i],
                    'smile': d_smile[i]
                    }
        metadata_list.append(metadata)
        t_1D_feature_list.append(t_1D_feature[i])
        d_3D_feature_list.append(d_3D_feature[i])
        label_list.append(label[i])

    return d_2D_feature,t_1D_feature_list,d_3D_feature_list, label_list, metadata_list

def cal_top_hit_ratio(p_list,c_list):
    positive_list = []
    for i,p in enumerate(p_list):
        if p[0] == 1 and p[1] == 1:
            positive_list.append([int(1),c_list[i]])
        elif p[0] == 1 and p[1] == 0:
            positive_list.append([int(0), c_list[i]])
    try:
        sort_list = sorted(positive_list, key=lambda x: x[1])[0:6]
        if all(element == 1 for element in sort_list):
            print("列表中的所有元素都为1")
            return 1
        else:
            print("列表中至少有一个元素不为1")
            return 0
    except:
        return 0

