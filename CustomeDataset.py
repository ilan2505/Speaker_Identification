import torch
from torch.utils.data import Dataset


class CustomeDataset(Dataset):
    """
        Custome Dataset class:
        X type => list of torch.tensor(float64), representing the emission on the wave.
        Y type => list of Integers representing the binary labels.
    """
    def __init__(self, X, Y):
        self.x = X
        self.y = torch.tensor(Y)
        self.len = len(Y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
