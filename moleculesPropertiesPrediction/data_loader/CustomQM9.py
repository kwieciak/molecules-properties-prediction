import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import QM9


class CustomQM9(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.original_data = QM9(root)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []
        for data in self.original_data:
            data.zebra = torch.tensor([3], dtype=torch.float)
            data_list.append(data)
        return self.collate(data_list)

    def get(self, idx):
        return super().get(idx)
