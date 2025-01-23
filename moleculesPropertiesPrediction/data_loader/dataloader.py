import pandas as pd
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


def load_dataset(batch_size, train_ratio, val_ratio, test_ratio):
    dataset_path = "./data"
    qm9 = QM9(root=dataset_path)


    #choosing one regression target
    y_target = pd.DataFrame(qm9.data.y.numpy())
    qm9.data.y = torch.Tensor(y_target[0])

    # shuffling data
    qm9 = qm9.shuffle()

    # splitting the data
    train_index = int(int(0.3 * len(qm9)) * train_ratio )
    val_index = train_index + int(int(0.3 * len(qm9)) * val_ratio)
    test_index = val_index + int(int(0.3 * len(qm9)) * test_ratio)

    # normalizing the data
    data_mean = qm9.data.y[0:train_index].mean()
    data_std = qm9.data.y[0:train_index].std()
    qm9.data.y = (qm9.data.y - data_mean) / data_std

    # putting datasets into dataloaders
    train_loader = DataLoader(qm9[0:train_index], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(qm9[train_index:val_index], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(qm9[val_index:test_index], batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
