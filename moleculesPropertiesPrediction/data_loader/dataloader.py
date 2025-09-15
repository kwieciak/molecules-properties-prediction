import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data_loader.CustomQM9 import CustomQM9
from utils.utils import save_r_targets_to_csv


def load_dataset(batch_size, train_ratio, val_ratio, test_ratio, train_r_targets, device, dataset_usage_ratio=1.0,
                 start_index=0, assign_loaded_targets = False,
                 shuffling=False, remove_outliers=True, iqr_k=2):
    dataset_path = "./data"
    dataset = CustomQM9(dataset_path, train_r_targets, assign_loaded_targets)
    filename = "r_targets_" + str(train_r_targets)[:20]
    save_r_targets_to_csv(dataset.r_target, filename)

    # splitting the data
    N = len(dataset)
    num_samples = int(N * dataset_usage_ratio)
    indices = list(range(start_index, min(start_index + num_samples, N)))
    full_index = [i for i in range(N)]

    # removing outliers
    if remove_outliers:
        mask = calculate_outliers_iqr(dataset, iqr_k)
        indices = [i for i in indices if mask[i]==True]
        full_index = [i for i in full_index if mask[i]==True]

    train_index, temp_index = train_test_split(indices, test_size=(1.0 - train_ratio), random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    # normalizing (standardization) the data
    data_mean = dataset.data.y[full_index].mean(dim=0, keepdim=True)
    data_std = dataset.data.y[full_index].std(dim=0, keepdim=True)
    dataset.data.y = ((dataset.data.y - data_mean) / data_std).to(device)

    # putting datasets into dataloaders
    train_loader = DataLoader([dataset[i] for i in train_index], batch_size=batch_size, shuffle=shuffling)
    val_loader = DataLoader([dataset[i] for i in val_index], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_index], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def calculate_outliers_iqr(dataset, iqr_k):
    y_all = dataset.data.y[:]
    Q1 = torch.quantile(y_all, 0.25, dim=0)
    Q3 = torch.quantile(y_all, 0.75, dim=0)
    IQR = Q3 - Q1

    lower = Q1 - iqr_k * IQR
    upper = Q3 + iqr_k * IQR

    mask = ((y_all >= lower) & (y_all <= upper)).all(dim=1)
    return mask

def calculate_outliers_zscore(dataset):
    y_all = dataset.data.y[:]
    y_mean = y_all.mean(dim=0, keepdim=True)
    y_std = y_all.std(dim=0, keepdim=True)

    z_score = (y_all - y_mean) / y_std

    mask = (z_score.abs() < 3).all(dim=1)

    return mask