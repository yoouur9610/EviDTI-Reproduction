"""
HPC clean version of the DrugBank lazy-loading dataset class,
plus the Davis dataset class (InMemoryDataset style, unchanged
from the author's original -- Davis protein features are a single
combined .npy file, not per-protein files, so the lazy-loading
rewrite used for DrugBank is not needed here).
Derived from Yuki's Colab notebooks, with re.sub patch mechanisms
and debug prints removed. Logic is otherwise unchanged from what
was actually used to produce the reported checkpoints.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from tqdm import tqdm

PROTEIN_MAX_LEN = 1000  # fixed padding/truncation length used during training


class bond_angle_graph_data(PyGDataset):
    def __init__(self, root, d_2D_path, t_1D_path, d_3D_path, label_file, metadata_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.d_2D_path = d_2D_path
        self.t_1D_path = t_1D_path
        self.d_3D_path = d_3D_path
        self.label_file = label_file
        self.metadata_path = metadata_path

        super(bond_angle_graph_data, self).__init__(root, transform, pre_transform, pre_filter)

        self.meta_df = pd.read_csv(metadata_path)
        self.labels = pd.read_csv(label_file).iloc[:, 2].values

        d_2D_raw = np.load(d_2D_path, allow_pickle=True)
        self.d_2D_dict = d_2D_raw.item() if d_2D_raw.ndim == 0 else d_2D_raw

        d_3D_raw = np.load(d_3D_path, allow_pickle=True)
        if d_3D_raw.ndim == 1:
            self.d_3D_dict = {item['smiles']: item for item in d_3D_raw if 'smiles' in item}
        else:
            self.d_3D_dict = d_3D_raw.item()

        self.valid_indices = []
        for idx in range(len(self.meta_df)):
            smile = self.meta_df.iloc[idx]['smile']
            uid = self.meta_df.iloc[idx]['uid']
            if smile in self.d_3D_dict and smile in self.d_2D_dict:
                if os.path.exists(os.path.join(self.t_1D_path, f"{uid}.npy")):
                    self.valid_indices.append(idx)

        print(f"[bond_angle_graph_data] valid samples: {len(self.valid_indices)} / {len(self.meta_df)}")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.meta_df.iloc[real_idx]
        current_uid = row['uid']
        current_smile = row['smile']
        label = self.labels[real_idx]

        # protein feature: fixed-length pad/truncate
        prot_path = os.path.join(self.t_1D_path, f"{current_uid}.npy")
        t_1D_fea = np.load(prot_path, allow_pickle=True)
        if hasattr(t_1D_fea, 'is_sparse') and t_1D_fea.is_sparse:
            t_1D_fea = t_1D_fea.to_dense()
        if t_1D_fea.ndim == 1:
            t_1D_fea = t_1D_fea[np.newaxis, :]

        L = t_1D_fea.shape[0]
        actual_len = min(L, PROTEIN_MAX_LEN)
        if L >= PROTEIN_MAX_LEN:
            t_1D_fea = t_1D_fea[:PROTEIN_MAX_LEN, :]
        else:
            t_1D_fea = np.pad(t_1D_fea, ((0, PROTEIN_MAX_LEN - L), (0, 0)), mode='constant')
        # Cell 6 fix: the extra leading dim is required so that PyG's
        # default per-batch concatenation (cat along dim=0, not stack)
        # produces a correctly-shaped [batch, seq_len, feat_dim] tensor
        # instead of flattening seq_len into the batch dimension.
        t_tensor = torch.tensor(t_1D_fea, dtype=torch.float).unsqueeze(0)

        # drug 3D features
        d_3D_info = self.d_3D_dict[current_smile]
        atom_feature = np.stack([
            d_3D_info['atomic_num'], d_3D_info['chiral_tag'], d_3D_info['degree'],
            d_3D_info['explicit_valence'], d_3D_info['formal_charge'],
            d_3D_info['hybridization'], d_3D_info['implicit_valence']
        ], axis=1)
        bond_feature = np.stack([
            d_3D_info['bond_dir'], d_3D_info['bond_type'],
            d_3D_info['is_in_ring'], d_3D_info['bond_length']
        ], axis=1)
        angle_feature = np.array(d_3D_info['bond_angle'])
        if angle_feature.ndim == 1:
            angle_feature = np.expand_dims(angle_feature, axis=1)

        atom_bond_graph = Data(
            x=torch.tensor(atom_feature, dtype=torch.float),
            edge_index=torch.tensor(d_3D_info['edges'], dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(bond_feature, dtype=torch.float)
        )
        bag_graph = Data(
            x=torch.tensor(bond_feature, dtype=torch.float),
            edge_index=torch.tensor(d_3D_info['BondAngleGraph_edges'], dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(angle_feature, dtype=torch.float)
        )
        atom_bond_graph.bag = bag_graph

        d_2D_fea = np.array(self.d_2D_dict[current_smile]).flatten()
        # Cell 7 fix: same PyG batching issue as Cell 6 (t_1D_feature) --
        # an extra leading dim is required so cat(dim=0) across a batch
        # stacks samples correctly instead of flattening the 256-dim
        # feature into the batch dimension.
        atom_bond_graph.d_2D_feature = torch.tensor(d_2D_fea, dtype=torch.float).view(1, -1)
        atom_bond_graph.t_1D_feature = t_tensor
        atom_bond_graph.l = torch.tensor(label, dtype=torch.long)

        atom_bond_graph.cid = str(row.get('cid', current_smile))
        atom_bond_graph.unid = str(current_uid)

        atom_bond_graph.metadata = {
            'id': f"{current_uid}_{current_smile}",
            'd_id': current_smile,
            't_id': current_uid,
            'smile': current_smile,
            'length': torch.tensor(actual_len, dtype=torch.long)
        }

        return atom_bond_graph


def davis_proprocess(d_2D_path, t_1D_path, d_3D_path, label_file_path, metadata_path):
    """
    Unchanged from utils.py's original davis_proprocess, EXCEPT for the
    diagnostic prints added below. NOTE: this function has a latent
    indexing bug inherited from the author's original code -- d_2D_feature
    is returned WITHOUT applying the `len(d_smile[i]) < 300` filter that
    all four other outputs (t_1D_feature_list, d_3D_feature_list,
    label_list, metadata_list) go through. If any sample actually gets
    filtered out, d_2D_feature[i] in davis_graph_data.process() will be
    misaligned against the other four lists. The diagnostic print below
    reports whether this filter removes anything for the real Davis data
    -- if `filtered == total`, the bug is dormant and harmless; if not,
    it needs a real fix (filtering d_2D_feature the same way) before any
    Davis result can be trusted.
    """
    d_2D_feature = np.load(d_2D_path, allow_pickle=True)
    t_1D_feature = np.load(t_1D_path, allow_pickle=True)
    d_3D_feature = np.load(d_3D_path, allow_pickle=True)
    label_file = pd.read_csv(label_file_path)
    label = label_file['label'].tolist()

    t_metadata = pd.read_csv(metadata_path)
    t_id = t_metadata['uid'].tolist()
    t_seq = t_metadata['seq'].tolist()
    d_id = t_metadata['cid']
    d_smile = t_metadata['smiles']

    metadata_list = []
    t_1D_feature_list = []
    d_3D_feature_list = []
    label_list = []
    for i in tqdm(range(len(t_id))):
        if len(d_smile[i]) < 300:
            metadata = {'id': t_id[i], 'sequence': str(t_seq[i]), 'length': len(t_seq[i]),
                        'd_id': d_id[i], 'smiles': d_smile[i]}
            metadata_list.append(metadata)
            t_1D_feature_list.append(t_1D_feature[i])
            d_3D_feature_list.append(d_3D_feature[i])
            label_list.append(label[i])

    print(f"[davis_proprocess] total samples: {len(t_id)}, "
          f"kept after SMILES<300 filter: {len(label_list)} "
          f"(d_2D_feature array length: {len(d_2D_feature)} -- "
          f"{'OK, filter removed nothing, indices stay aligned' if len(label_list) == len(t_id) else 'WARNING: filter removed samples, d_2D indices will be MISALIGNED'})")

    return d_2D_feature, t_1D_feature_list, d_3D_feature_list, label_list, metadata_list


class davis_graph_data(InMemoryDataset):
    """
    Unchanged from the author's original InMemoryDataset implementation
    (function.py). Davis is small enough, and protein features arrive as
    a single combined .npy file (not per-protein files like DrugBank),
    so the one-time process()+cache approach works fine here -- no
    lazy-loading rewrite needed.
    """
    def __init__(self, root, d_2D_path, t_1D_path, d_3D_path, label_file, metadata_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.d_2D_path = d_2D_path
        self.t_1D_path = t_1D_path
        self.d_3D_path = d_3D_path
        self.label_file = label_file
        self.metadata_path = metadata_path
        super(davis_graph_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['davis.pt']

    def download(self):
        pass

    def process(self):
        d_2D_feature, t_1D_feature, d_3D_feature, label, metadata_list = davis_proprocess(
            self.d_2D_path, self.t_1D_path, self.d_3D_path, self.label_file, self.metadata_path)
        print(len(d_2D_feature))
        print(len(label))
        data_len = len(label)
        atom_bond_graph_list = []
        for i in range(data_len):
            d_2D_fea = d_2D_feature[i]
            t_1D_fea = t_1D_feature[i]
            atom_feature = np.stack([d_3D_feature[i]['atomic_num'], d_3D_feature[i]['chiral_tag'],
                                      d_3D_feature[i]['degree'], d_3D_feature[i]['explicit_valence'],
                                      d_3D_feature[i]['formal_charge'], d_3D_feature[i]['hybridization'],
                                      d_3D_feature[i]['implicit_valence']])
            bond_feature = np.stack([d_3D_feature[i]['bond_dir'], d_3D_feature[i]['bond_type'],
                                      d_3D_feature[i]['is_in_ring'], d_3D_feature[i]['bond_length']])
            angle_feature = d_3D_feature[i]['bond_angle']

            # Matches the author's original function.py EXACTLY (verified
            # 2026-08-18 against the raw uploaded file, and against the
            # Davis notebook's Cell 16/17 which import the unmodified
            # `function.py` and work correctly). d_2D_feature, t_1D_feature,
            # edge_attr, and label are all left as raw numpy/python objects
            # -- NOT wrapped in torch.tensor(), NOT unsqueezed. PyG's
            # InMemoryDataset.collate() only tries to torch.cat attributes
            # that are already torch.Tensor; non-tensor attributes are
            # stored per-sample in a plain list instead, which is exactly
            # why the varying real protein sequence lengths never caused a
            # problem in the original code -- no shape-matching is required
            # for non-tensor attributes. An earlier version of this file
            # incorrectly added torch.tensor()/.unsqueeze(0)/.view(1,-1)
            # here (copying a fix that was only valid for DrugBank's
            # different lazy-loading Dataset), which broke this and caused
            # a "sizes must match" error during construction. Reverted.
            atom_bond_graph = Data(x=torch.tensor(atom_feature).transpose(1, 0),
                                    edge_index=torch.LongTensor(d_3D_feature[i]['edges'].transpose(1, 0)),
                                    edge_attr=bond_feature.T)
            atom_bond_graph.bag = Data(x=torch.tensor(bond_feature).transpose(1, 0),
                                        edge_index=torch.LongTensor(
                                            d_3D_feature[i]['BondAngleGraph_edges'].transpose(1, 0)),
                                        edge_attr=angle_feature)

            atom_bond_graph.d_2D_feature = d_2D_fea
            atom_bond_graph.t_1D_feature = t_1D_fea
            atom_bond_graph.l = label[i]
            atom_bond_graph.metadata = metadata_list[i]

            atom_bond_graph_list.append(atom_bond_graph)

        if self.pre_filter is not None:
            atom_bond_graph_list = [d for d in atom_bond_graph_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            atom_bond_graph_list = [self.pre_transform(d) for d in atom_bond_graph_list]

        data_1, slices_1 = self.collate(atom_bond_graph_list)
        torch.save((data_1, slices_1), self.processed_paths[0])
