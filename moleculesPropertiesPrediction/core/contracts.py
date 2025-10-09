import torch
from torch.utils.data import DataLoader


class DataModule:
    def setup(self): pass
    def train_loader(self) -> DataLoader: pass
    def val_loader(self) -> DataLoader: pass
    def test_loader(self) -> DataLoader: pass
    @property
    def input_dim(self) -> int: pass
    @property
    def output_dim(self) -> int: pass


class Model(torch.nn.Module):
    def forward(self, batch) -> torch.Tensor: pass
