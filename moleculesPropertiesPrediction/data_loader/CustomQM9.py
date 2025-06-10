import random

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import QM9


class CustomQM9(InMemoryDataset):
    def __init__(self, root, train_r_targets):
        super().__init__(root)
        self.original_data = QM9(root)
        self.data, self.slices = self.process_data(train_r_targets)

    def process_data(self, train_r_targets):
        data_list = []
        for data in self.original_data:
            r_target = random.choice(train_r_targets)
            data.r_target = torch.tensor(r_target)
            data_list.append(data)
        return self.collate(data_list)

    def get(self, idx):
        return super().get(idx)
