import random

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import QM9


class CustomQM9(InMemoryDataset):
    def __init__(self, root, train_r_targets, assign_loaded_targets):
        super().__init__(root)
        self.original_data = QM9(root)
        if not assign_loaded_targets:
            self.data, self.slices = self.process_data_randomly(train_r_targets)
        elif assign_loaded_targets:
            self.data, self.slices = self.process_data_assign_loaded_targets(train_r_targets)


    def process_data_randomly(self, train_r_targets):
        data_list = []
        for data in self.original_data:
            r_target = random.choice(train_r_targets)
            data.r_target = torch.tensor(r_target)
            data_list.append(data)
        return self.collate(data_list)

    def process_data_assign_loaded_targets(self, train_r_targets):
        data_list = []
        for r_target, data in zip(train_r_targets, self.original_data):
            data.r_target = torch.tensor(r_target)
            data_list.append(data)
        return self.collate(data_list)

    def get(self, idx):
        return super().get(idx)
