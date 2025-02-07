import pandas as pd
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


def load_dataset(batch_size, train_ratio, val_ratio, test_ratio, device, dataset_usage_ratio=1, shuffling=False):
    dataset_path = "./data"
    qm9 = QM9(root=dataset_path)

    # choosing one regression target
    y_target = pd.DataFrame(qm9.data.y.cpu().numpy())
    qm9.data.y = torch.Tensor(y_target[0]).to(device)

    # shuffling data
    if shuffling:
        qm9 = qm9.shuffle()

    # splitting the data
    num_samples = int(len(qm9) * dataset_usage_ratio)
    indices = list(range(num_samples))

    train_index, temp_index = train_test_split(indices, test_size=(1 - train_ratio), random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    # normalizing the data
    data_mean = qm9.data.y[train_index].mean()
    data_std = qm9.data.y[train_index].std()
    qm9.data.y = ((qm9.data.y - data_mean) / data_std).to(device)

    # putting datasets into dataloaders
    train_loader = DataLoader([qm9[i] for i in train_index], batch_size=batch_size, shuffle=shuffling)
    val_loader = DataLoader([qm9[i] for i in val_index], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([qm9[i] for i in test_index], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
