import random

import pandas as pd
import torch
import torch_geometric.data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split


def load_dataset(batch_size, train_ratio, val_ratio, test_ratio, target_indices, device, dataset_usage_ratio=1.0,
                 shuffling=False):
    dataset_path = "./data"
    dataset = QM9(root=dataset_path)
    dataset.transform = lambda data: add_new_attribute(data, target_indices)

    # choosing regression targets
    y_target = pd.DataFrame(dataset.data.y.cpu().numpy())
    dataset.data.y = torch.Tensor(y_target[target_indices].values).to(device)

    # shuffling data
    if shuffling:
        dataset = dataset.shuffle()

    # splitting the data
    num_samples = int(len(dataset) * dataset_usage_ratio)
    indices = list(range(num_samples))

    train_index, temp_index = train_test_split(indices, test_size=(1.0 - train_ratio), random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    # normalizing the data
    data_mean = dataset.data.y[train_index].mean(dim=0, keepdim=True)
    data_std = dataset.data.y[train_index].std(dim=0, keepdim=True)
    dataset.data.y = ((dataset.data.y - data_mean) / data_std).to(device)

    # putting datasets into dataloaders
    train_loader = DataLoader([dataset[i] for i in train_index], batch_size=batch_size, shuffle=shuffling)
    val_loader = DataLoader([dataset[i] for i in val_index], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_index], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def add_new_attribute(data, target_indices):
    data.task_index = torch.tensor([random.choice(target_indices)], dtype=torch.int)
    return data

def prepare_task_indices(target_indices, size):
    num_tasks = len(target_indices)
    samples_per_task = size // num_tasks
    task_indices = []

    for i in range(size):
        task_index = (i // samples_per_task) % num_tasks
        task_indices.append(task_index)

    return task_indices
    #return torch.tensor(task_indices, dtype=torch.long)

