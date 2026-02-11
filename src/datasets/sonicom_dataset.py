import numpy as np
import torch
from torch.utils.data import Dataset


class SonicomHLogDataset(Dataset):
    """
    Dataset for SONICOM H_log stored in npz.

    Returns:
        x_sparse: (D,2,106)  float32
        y_dense : (793,2,106) float32
    """
    def __init__(self, H_log: np.ndarray, input_didxs: np.ndarray, subject_indices: np.ndarray):
        """
        H_log: (N,793,2,106) in dB domain
        input_didxs: (D,)
        subject_indices: list of subject ids to use
        """
        self.H = H_log.astype(np.float32)
        self.input_didxs = input_didxs.astype(np.int64)
        self.subject_indices = subject_indices.astype(np.int64)

    def __len__(self):
        return len(self.subject_indices)

    def __getitem__(self, i):
        sid = self.subject_indices[i]
        y = self.H[sid]                           # (793,2,106)
        x = y[self.input_didxs]                   # (D,2,106)
        return torch.from_numpy(x), torch.from_numpy(y)
