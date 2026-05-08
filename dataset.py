import glob
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            return (torch.from_numpy(data["x_static"]).float(),
                    torch.from_numpy(data["x_dynamic"]).float(),
                    torch.from_numpy(data["y"]).float())