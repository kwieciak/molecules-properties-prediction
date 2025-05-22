import random

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import QM9


class CustomQM9(InMemoryDataset):
    def __init__(self, root, target_indices):
        super().__init__(root)
        self.original_data = QM9(root)
        self.data, self.slices = self.process_data(target_indices)

    def process_data(self, target_indices):
        data_list = []
        for data in self.original_data:
            target_index = random.choice(target_indices)
            data.r_target = torch.tensor(target_index)
            data_list.append(data)
        return self.collate(data_list)

    def get(self, idx):
        return super().get(idx)
