from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data_loader.CustomQM9 import CustomQM9
from utils.utils import save_r_targets_to_csv


def load_dataset(batch_size, train_ratio, val_ratio, test_ratio, train_r_targets, device, dataset_usage_ratio=1.0,
                 start_index=0, assign_loaded_targets = False,
                 shuffling=False):
    dataset_path = "./data"
    dataset = CustomQM9(dataset_path, train_r_targets, assign_loaded_targets)

    filename = "r_targets_" + str(train_r_targets)[:20]
    save_r_targets_to_csv(dataset.r_target, filename)

    # splitting the data
    num_samples = int(len(dataset) * dataset_usage_ratio)
    indices = list(range(start_index, start_index + num_samples))

    train_index, temp_index = train_test_split(indices, test_size=(1.0 - train_ratio), random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    # normalizing (standardization) the data
    data_mean = dataset.data.y[train_index].mean(dim=0, keepdim=True)
    data_std = dataset.data.y[train_index].std(dim=0, keepdim=True)
    dataset.data.y = ((dataset.data.y - data_mean) / data_std).to(device)

    # putting datasets into dataloaders
    train_loader = DataLoader([dataset[i] for i in train_index], batch_size=batch_size, shuffle=shuffling)
    val_loader = DataLoader([dataset[i] for i in val_index], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_index], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
