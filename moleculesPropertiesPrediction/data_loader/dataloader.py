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

    #Przypisanie nowego atrybutu bezposrednio do pojedynczych obiektow - AttributeError: 'GlobalStorage' object has no attribute 'task_idx'
    #dataset[0].task_index = torch.tensor([0.5], dtype=torch.float)

    #Przypisanie nowego atrybutu do całego datasetu
    num_samples = dataset.data.idx.shape[0]
    dataset.data.task_idx = torch.zeros(num_samples, dtype=torch.long)
    #print(dataset.data)
    #Data(x=[2359210, 11], edge_index=[2, 4883516], edge_attr=[4883516, 4], y=[130831, 19], pos=[2359210, 3],idx=[130831], name=[130831], z=[2359210], task_idx=[130831])
    #print(dataset[0])
    #Data(x=[5, 11], edge_index=[2, 8], edge_attr=[8, 4], y=[1, 19], pos=[5, 3], idx=[1], name='gdb_1', z=[5]) - brak task_idx
    #print(dataset[0].task_idx)
    #AttributeError: 'GlobalStorage' object has no attribute 'task_idx'


    # choosing regression targets
    y_target = pd.DataFrame(dataset.data.y.cpu().numpy())
    dataset.data.y = torch.Tensor(y_target[target_indices].values).to(device)

    # shuffling data
    if shuffling:
        dataset = dataset.shuffle()

    # splitting the data
    num_samples = int(len(dataset) * dataset_usage_ratio)
    indices = list(range(num_samples))

    # attaching task_index
    # num_tasks = len(target_indices)
    # samples_per_task = num_samples // num_tasks
    # new_dataset = []
    #
    # for i in range(num_samples):
    #     task_index = (i // samples_per_task) % num_tasks
    #
    #     data = dataset[i]
    #     data.task_index = torch.tensor([task_index + 1], dtype=torch.long)
    #
    #     new_dataset.append(data)
    #
    # dataset=new_dataset

    num_tasks = len(target_indices)
    samples_per_task = num_samples // num_tasks
    task_indices = []

    for i in range(num_samples):
        task_index = (i//samples_per_task) % num_tasks
        task_indices.append(task_index)

    train_index, temp_index = train_test_split(indices, test_size=(1.0 - train_ratio), random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    # normalizing the data
    data_mean = dataset.data.y[train_index].mean(dim=0, keepdim=True)
    data_std = dataset.data.y[train_index].std(dim=0, keepdim=True)
    dataset.data.y = ((dataset.data.y - data_mean) / data_std).to(device)

    # putting datasets into dataloaders
    #train_loader = DataLoader([dataset[i] for i in train_index], batch_size=batch_size, shuffle=shuffling, collate_fn=collate_fn)
    train_loader = DataLoader([dataset[i] for i in train_index], batch_size=batch_size, shuffle=shuffling)
    val_loader = DataLoader([dataset[i] for i in val_index], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_index], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, torch.tensor(task_indices, dtype=torch.long)

# def collate_fn(batch):
#     batch_data = torch_geometric.data.Batch.from_data_list(batch)
#     batch_task_index = torch.stack([data.task_idx for data in batch])
#
#     return batch_data, batch_task_index
