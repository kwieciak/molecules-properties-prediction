import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as GraphDataLoader

from enums.enums import Normalization
from utils.utils import save_r_targets_to_csv, get_target_attr, ensure_2d_tensor


def load_dataset(
        dataset,
        batch_size: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        is_graph: bool,
        normalization_range: str,
        target_attr: str = "y",
        dataset_usage_ratio: float = 1.0,
        start_index: int = 0,
        normalization: Normalization | None = Normalization.STANDARD,
        shuffling: bool = False,
        remove_outliers: bool = True,
        iqr_k: int = 2
):
    if hasattr(dataset, "r_target"):
        filename = "r_targets_" + str(getattr(dataset, "r_target"))[:20]
        save_r_targets_to_csv(getattr(dataset, "r_target"), filename)

    if not hasattr(dataset, "data"):
        raise AttributeError("The dataset must have the ‘data’ attribute.")

    y_all = get_target_attr(dataset, target_attr)
    y_all = ensure_2d_tensor(y_all)

    # splitting the data
    N = len(dataset)
    num_samples = int(N * dataset_usage_ratio)
    indices = list(range(start_index, min(start_index + num_samples, N)))
    full_index = list(range(N))

    # removing outliers
    if remove_outliers:
        mask = calculate_outliers_iqr(y_all, iqr_k)
        indices = [i for i in indices if mask[i] == True]
        full_index = [i for i in full_index if mask[i] == True]

    train_index, temp_index = train_test_split(indices, test_size=(1.0 - train_ratio), random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)
    if normalization_range == "train":
        normalisation_index = train_index
    elif normalization_range == "full":
        normalisation_index = full_index
    else:
        raise Exception("normalization_range must be 'train' or 'full'")

    # normalizing (standardization) the data
    if normalization == Normalization.STANDARD:
        data_mean = y_all[normalisation_index].mean(dim=0, keepdim=True)
        data_std = y_all[normalisation_index].std(dim=0, keepdim=True)
        normalized = ((y_all - data_mean) / data_std)
        setattr(dataset.data, target_attr, normalized)

    # normalizing (min/max) the data
    elif normalization == Normalization.MINMAX:
        data_min, _ = y_all[normalisation_index].min(dim=0, keepdim=True)
        data_max, _ = y_all[normalisation_index].max(dim=0, keepdim=True)
        normalized = ((y_all - data_min) / (data_max - data_min))
        setattr(dataset.data, target_attr, normalized)

    elif normalization is None:
        pass
    else:
        raise ValueError(f"Unknown normalization {normalization}")

    # putting datasets into dataloaders
    if is_graph:
        train_loader = GraphDataLoader([dataset[i] for i in train_index], batch_size=batch_size, shuffle=shuffling)
        val_loader = GraphDataLoader([dataset[i] for i in val_index], batch_size=batch_size, shuffle=False)
        test_loader = GraphDataLoader([dataset[i] for i in test_index], batch_size=batch_size, shuffle=False)
    elif not is_graph:
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=batch_size, shuffle=shuffling)
        val_loader = DataLoader(Subset(dataset, val_index), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=batch_size, shuffle=False)
    else:
        raise NotImplementedError

    return train_loader, val_loader, test_loader


def calculate_outliers_iqr(y_all: torch.Tensor, iqr_k: float) -> torch.Tensor:
    Q1 = torch.quantile(y_all, 0.25, dim=0)
    Q3 = torch.quantile(y_all, 0.75, dim=0)
    IQR = Q3 - Q1

    lower = Q1 - iqr_k * IQR
    upper = Q3 + iqr_k * IQR

    mask = ((y_all >= lower) & (y_all <= upper)).all(dim=1)
    return mask


def calculate_outliers_zscore(y_all: torch.Tensor) -> torch.Tensor:
    y_mean = y_all.mean(dim=0, keepdim=True)
    y_std = y_all.std(dim=0, keepdim=True)

    z_score = (y_all - y_mean) / y_std

    mask = (z_score.abs() < 3).all(dim=1)

    return mask
