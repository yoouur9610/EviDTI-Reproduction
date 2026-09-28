"""
Deterministic softmax confidence baseline (drugbank_solver_softmax_hpc.py).

This is NOT the evidential model. It is an independently trained classifier
used as one of the four uncertainty-quantification baselines requested by
R1-3 / R2-5 (softmax confidence, temperature scaling, MC Dropout, deep
ensembles). Feature extraction (protein/drug/GeoGNN branches) is identical
to the Full evidential model (see drugbank_model_softmax_hpc.py); only the
output layer and loss function differ:

  - Output layer: standard softmax (drugbank_model_softmax_hpc.py) instead
    of softplus(x)+1 -> Dirichlet alpha parameters.
  - Loss: cross-entropy (implemented here as NLLLoss on log-softmax
    probabilities, numerically equivalent to CrossEntropyLoss on logits)
    instead of dirichlet_loss. No KL annealing term -- that concept is
    specific to evidential learning and does not apply here.
  - "Confidence" (predict_val's `c` output) = max softmax probability,
    the standard textbook definition of softmax confidence. There is no
    evidence/belief-mass decomposition, since this model does not produce
    Dirichlet parameters.

Shared with drugbank_solver_hpc.py (the evidential Full model solver):
loss.mean() aggregation, float32 labels, ReduceLROnPlateau scheduler
(patience=10, min_lr=1.5e-4), val_loss tracking, unique run-directory
naming via PBS_JOBID. Early-stopping patience (30) and MAX_EPOCH (from
yaml) are also kept identical to the Full model for a fair comparison
under matched training budgets.
"""
import os
from typing import Tuple

import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, \
            accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from torch.utils.tensorboard import SummaryWriter


def _unique_run_dir():
    """
    Generate a run directory name that cannot collide across
    concurrently-starting jobs on different compute nodes. The original
    code used only a per-second timestamp (datetime.now().strftime(...)),
    which was CONFIRMED to collide in practice: three separate PBS jobs,
    launched by the queue scheduler within the same second on three
    different nodes, all wrote into the same 'runs/la_test1_...' folder,
    silently overwriting each other's checkpoint.pth and result.csv.

    Fix: prefer PBS_JOBID, which the scheduler guarantees is globally
    unique per job -- far more reliable than reconstructing uniqueness
    from a timestamp. Falls back to hostname+PID+microsecond precision
    for interactive/non-PBS runs (e.g. quick tests on the login node),
    which is not scheduler-guaranteed but is unique in practice for any
    two processes that aren't started in the exact same microsecond on
    the exact same host.
    """
    import socket
    pbs_jobid = os.environ.get('PBS_JOBID')
    if pbs_jobid:
        tag = pbs_jobid.split('.')[0]  # keep just the numeric job id part
    else:
        tag = f"{socket.gethostname()}_{os.getpid()}_{datetime.now().strftime('%d-%m_%H-%M-%S-%f')}"
    return 'runs/la_test1_{}'.format(tag)
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib
matplotlib.use('Agg')  # no display available on HPC compute nodes
import matplotlib.pyplot as plt

from utils import cal_top_hit_ratio


class Solver():
    def __init__(self, model, cfg, device, optim=torch.optim.Adam, loss_func=None, eval=False):
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = cfg.SOLVER['BATCH_SIZE']
        self.epoch = cfg.SOLVER['MAX_EPOCH']
        self.lr = cfg.SOLVER['LR']
        self.weight_decay = cfg.SOLVER['WEIGHT_DECAY']
        self.loss_func_name = cfg.SOLVER['LOSS_FUNCTION']
        self.optim = optim(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # ReduceLROnPlateau: halves LR after 10 epochs without val-acc
        # improvement, floors at 1.5e-4 (matches manuscript 2.3.2).
        # Chosen over CosineAnnealingLR because MAX_EPOCH is governed
        # by early stopping (patience=30), not a fixed horizon, so a
        # plateau-based schedule is the natural match.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='max', factor=0.5, patience=10, min_lr=1.5e-4
        )

        if eval:
            checkpoint = torch.load(os.path.join(''), map_location=self.device)
            self.writer = SummaryWriter(_unique_run_dir())
            self.max_val_acc = checkpoint['maximum_accuracy']

        if not eval:
            self.start_epoch = 0
            self.max_val_acc = 0
            self.writer = SummaryWriter(_unique_run_dir())

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_data=None):
        epochs_no_improve = 0
        min_train_acc = 95
        max_train_acc = 5
        for epoch in range(self.start_epoch, self.epoch):
            self.model.train()
            train_loss, train_results = self.predict(train_loader, epoch + 1, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                val_loss, _ = self.predict(val_loader, epoch + 1, optim=None)
                _, _, val_results, c, prob_list = self.predict_val(val_loader, epoch + 1)

            val_results = np.array(val_results)
            val_results = np.squeeze(val_results)
            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                train_mcc = matthews_corrcoef(train_results[:, 1], train_results[:, 0])
                val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
                val_auc = roc_auc_score(val_results[:, 1], prob_list)

            print('[Epoch %d] Train Loss: %.4f | Val Loss: %.4f | Train Acc: %.2f%% | Val Acc: %.2f%% | '
                  'Val MCC: %.4f | lr: %.6f' % (
                epoch + 1, train_loss, val_loss, train_acc, val_acc, val_mcc,
                self.optim.param_groups[0]['lr']))

            self.writer.add_scalars('Acc', {'train': train_acc, 'val': val_acc}, epoch + 1)
            self.writer.add_scalars('MCC', {'train': train_mcc, 'val': val_mcc}, epoch + 1)
            self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch + 1)

            # Cell 12 fix: separate train/val loss history saved to their
            # own CSV files (in addition to the TensorBoard scalars above).
            save_path = getattr(self, 'result_dir', getattr(self, 'save_dir', self.writer.log_dir))
            train_dir = os.path.join(save_path, 'train_loss')
            val_dir = os.path.join(save_path, 'val_loss')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            if not hasattr(self, 'train_history'):
                self.train_history = []
            if not hasattr(self, 'val_history'):
                self.val_history = []
            self.train_history.append({'Epoch': epoch + 1, 'Train_Loss': train_loss,
                                        'Train_Acc': train_acc, 'Train_MCC': train_mcc})
            self.val_history.append({'Epoch': epoch + 1, 'Val_Loss': val_loss,
                                      'Val_Acc': val_acc, 'Val_MCC': val_mcc})
            pd.DataFrame(self.train_history).to_csv(os.path.join(train_dir, 'train_loss.csv'), index=False)
            pd.DataFrame(self.val_history).to_csv(os.path.join(val_dir, 'val_loss.csv'), index=False)

            # step the scheduler on validation accuracy for this epoch
            self.scheduler.step(val_acc)

            if val_acc >= self.max_val_acc:
                epochs_no_improve = 0
                self.max_val_acc = val_acc
                self.save_checkpoint(epoch + 1)
            else:
                epochs_no_improve += 1

            with open(os.path.join(self.writer.log_dir, 'epoch.txt'), 'w') as file:
                file.write(str(epoch))

            if train_acc >= max_train_acc:
                max_train_acc = train_acc
            if epochs_no_improve >= 30 and max_train_acc >= min_train_acc:
                break

        if eval_data:
            checkpoint = torch.load(os.path.join(self.writer.log_dir, 'checkpoint.pth'))
            self.model = checkpoint
            self.evaluation(eval_data, filename='val_data_after_training')

    def predict(self, data_loader: DataLoader, epoch: int = None, optim: torch.optim.Optimizer = None) -> \
            Tuple[float, np.ndarray]:

        results = []
        running_loss = 0
        for i, batch in enumerate(data_loader):
            metadata = batch.metadata
            label_org = batch.l

            sequence_lengths = metadata['length'][:, None].to(self.device)
            mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:, None]
            label_org = label_org.to(self.device)
            # CrossEntropyLoss expects raw integer class labels (not one-hot)
            # and raw logits (not softmax-ed probabilities). Since the model
            # (drugbank_model_softmax_hpc.py) already applies softmax
            # internally for consistency with how predictions/probabilities
            # are read elsewhere in this codebase, we recover an equivalent
            # loss via NLLLoss on log-probabilities instead of re-deriving
            # logits, which is numerically equivalent to CrossEntropyLoss on
            # logits: CE(logits, y) == NLLLoss(log_softmax(logits), y), and
            # log(softmax(x)) == log_softmax(x) up to floating point.
            prediction = self.model(batch, mask=mask.to(self.device), sequence_lengths=sequence_lengths)
            log_probs = torch.log(prediction.clamp(min=1e-12))
            loss = torch.nn.functional.nll_loss(log_probs, label_org)
            if optim:
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            pred = torch.max(prediction[..., -2:], dim=1)[1]

            results.append(torch.stack((pred, label_org), dim=1).detach().cpu().numpy())
            loss_item = loss.item()
            running_loss += loss_item

            if i % 100 == 99:
                if epoch:
                    print('Epoch %d ' % (epoch), end=' ')
                print('[Iter %5d/%5d] %s: loss: %.7f, accuracy: %.4f%%' % (
                    i + 1, len(data_loader), 'Train' if optim else 'Val', loss_item,
                    100 * (pred == label_org).sum().item() / self.batch_size))

        running_loss /= len(data_loader)
        return running_loss, np.concatenate(results)

    def predict_val(self, data_loader: DataLoader, epoch: int = None):
        preds = []
        labels = []
        t_id = []
        d_id = []
        for i, batch in enumerate(data_loader):
            metadata = batch.metadata
            label_org = batch.l
            label_org = label_org.to(self.device)

            sequence_lengths = metadata['length'][:, None].to(self.device)
            mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:, None]
            batch_preds = self.model(batch, mask=mask.to(self.device), sequence_lengths=sequence_lengths)
            batch_preds = batch_preds.tolist()
            label_org = label_org.tolist()
            preds.extend(batch_preds)
            labels.extend(label_org)
            id1 = metadata['id']
            id2 = metadata['d_id']
            t_id.extend(id1)
            d_id.extend(id2)

        p = []
        c = []
        prob_list = []

        for i in range(len(preds)):
            # preds[i] is a standard softmax output (already sums to 1),
            # NOT Dirichlet alphas. "Confidence" for this baseline is simply
            # the max softmax probability -- the quantity R1-3/R2-5 refer to
            # as "deterministic softmax confidence". No evidence/belief-mass
            # decomposition applies here (that is evidential-specific).
            probs = np.array(preds[i])
            probs = np.squeeze(probs)
            if probs[0] >= probs[1]:
                pred = 0
            else:
                pred = 1

            p.append(np.stack([pred, labels[i]]))
            prob_list.append(probs[1])

            # Confidence = max softmax probability (the standard, textbook
            # definition of "softmax confidence" used as this baseline).
            c.append(np.max(probs))

        return t_id, d_id, p, c, prob_list

    def evaluation(self, eval_dataset, filename: str = '', lookup_dataset: Dataset = None):
        self.model.eval()
        with torch.no_grad():
            t_id, d_id, p, c, prob_list = self.predict_val(eval_dataset)
        val_results = np.squeeze(p)
        val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
        val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
        val_f1 = f1_score(val_results[:, 1], val_results[:, 0])
        val_auc = roc_auc_score(val_results[:, 1], prob_list)
        val_recall = recall_score(val_results[:, 1], val_results[:, 0])
        val_pre = precision_score(val_results[:, 1], val_results[:, 0])
        fpr, tpr, tresholds = roc_curve(val_results[:, 1], prob_list, pos_label=1)
        precision, recall, _thresholds = precision_recall_curve(val_results[:, 1], prob_list)
        val_prauc = auc(recall, precision)
        print(val_acc, val_recall, val_pre, val_mcc, val_f1, val_auc, val_prauc)

        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.5f)' % val_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.writer.log_dir, 'roc_curve.png'), dpi=200, bbox_inches='tight')
        plt.close()

        result_id = []
        result_id.append(t_id)
        result_id.append(d_id)
        result_id.append(prob_list)
        result_id.append(p)
        result_id.append(c)  # confidence = max softmax probability
        result_csv = pd.DataFrame(result_id)
        result_csv.to_csv(os.path.join(self.writer.log_dir, 'result.csv'))
        matrixs = [val_acc, val_recall, val_pre, val_mcc, val_f1, val_auc, val_prauc]
        with open('matrix_result.csv', 'a') as f:
            f.write('\t'.join(map(str, matrixs)) + '\n')

    def save_checkpoint(self, epoch: int):
        run_dir = self.writer.log_dir
        torch.save(self.model, os.path.join(run_dir, 'checkpoint.pth'))
