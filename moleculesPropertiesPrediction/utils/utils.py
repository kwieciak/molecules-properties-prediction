import csv
import os

import pandas as pd
import psutil
import torch
from matplotlib import pyplot as plt

from config import timestamp


def get_target_attr(dataset, name):
    if hasattr(dataset.data, name):
        return getattr(dataset.data, name)
    if hasattr(dataset, name):
        return getattr(dataset, name)
    raise AttributeError(f"Attribute '{name}' not found on dataset or dataset.data.")


def ensure_2d_tensor(y_all, dtype=None, device=None):
    if isinstance(y_all, torch.Tensor):
        t = y_all.to(device=device, dtype=dtype) if (dtype or device) else y_all

    # list/tuple of tensors -> stack
    elif isinstance(y_all, (list, tuple)) and y_all and isinstance(y_all[0], torch.Tensor):
        t = torch.stack(y_all, dim=0).to(device=device, dtype=dtype) if (dtype or device) else torch.stack(y_all, dim=0)

    # list/tuple of numbers -> tensor
    elif isinstance(y_all, (list, tuple)):
        t = torch.tensor(y_all, dtype=dtype, device=device) if (dtype or device) else torch.tensor(y_all)

    # NumPy array
    else:
        try:
            import numpy as np
            if isinstance(y_all, np.ndarray):
                t = torch.from_numpy(y_all).to(device=device, dtype=dtype) if (dtype or device) else torch.from_numpy(
                    y_all)
            else:
                t = torch.tensor([y_all], dtype=dtype, device=device) if (dtype or device) else torch.tensor([y_all])
        except Exception:
            t = torch.tensor([y_all], dtype=dtype, device=device) if (dtype or device) else torch.tensor([y_all])

    if t.dim() == 1:
        t = t.unsqueeze(1)
    return t


def print_memory_usage():
    ram_memory = psutil.virtual_memory()
    free_ram = ram_memory.available / (1024 ** 3)
    total_ram = ram_memory.total / (1024 ** 3)
    print(f"Free RAM: {free_ram:.2f} GB / {total_ram:.2f} GB")


def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"VRAM usage: {gpu_memory:.2f} GB")


def ensure_folder(folder):
    os.makedirs(folder, exist_ok=True)


def save_loss_to_csv(loss_list, filename):
    ensure_folder(f"results/{timestamp}/csv/loss")
    filename = f"results/{timestamp}/csv/loss/{filename}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])
        for epoch, loss in enumerate(loss_list, start=1):
            writer.writerow([epoch, loss])


def save_metrics_to_csv(metrics_dict, filename):
    ensure_folder(f"results/{timestamp}/csv/metrics")
    filename = f"results/{timestamp}/csv/metrics/{filename}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for metric, value in metrics_dict.items():
            writer.writerow([metric, value])


def save_preds_targets_to_csv(preds, targets, filename):
    ensure_folder(f"results/{timestamp}/csv/predstargets")
    filename = f"results/{timestamp}/csv/predstargets/{filename}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prediction", "Target"])
        for pred, target in zip(preds, targets):
            writer.writerow([pred, target])


def save_r_targets_to_csv(r_targets, filename):
    ensure_folder(f"results/{timestamp}/csv/r_targets")
    filename = f"results/{timestamp}/csv/r_targets/{filename}.csv"
    r_target_list = r_targets.tolist()
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["r_target"])
        for value in r_target_list:
            writer.writerow([value])


def load_csv(filepath):
    return pd.read_csv(filepath)


def load_loss_csv(filepath):
    loss_list = []
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            loss_list.append(float(row['Loss']))
    return loss_list


def load_metrics_csv(filepath):
    metrics_dict = {}
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            metrics_dict[row['Metric']] = float(row['Value'])
    return metrics_dict


def load_preds_targets_csv(filepath):
    preds_list = []
    targets_list = []
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            preds_list.append(float(row['Prediction']))
            targets_list.append(float(row['Target']))
    return preds_list, targets_list


def load_r_targets_csv(filepath):
    r_targets_list = []
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            r_targets_list.append(int(row['r_target']))
    return r_targets_list


def plot_metric_comparison(metrics1, metrics2, metric_name, label1, label2):
    ensure_folder(f"results/{timestamp}/plots/metrics")

    values = [metrics1[metric_name], metrics2[metric_name]]
    labels = [label1, label2]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, width=0.5)
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} comparison")
    plt.ylim(0, max(values) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

    filename = f"results/{timestamp}/plots/metrics/{metric_name}_comparison.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_learning_curve(train_loss, val_loss, label):
    ensure_folder(f"results/{timestamp}/plots/lc")
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning curve – {label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    filename = f"results/{timestamp}/plots/lc/learning_curve_{label.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_parity_plot(preds, targets, label):
    ensure_folder(f"results/{timestamp}/plots/parity")

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.6, label="Values")
    min_val = min(min(preds), min(targets))
    max_val = max(max(preds), max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Diagonal line")

    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title(f"Parity Plot – {label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results/{timestamp}/plots/parity/parity_plot_{label.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def print_model_details(model):
    for name, param in model.named_parameters():
        print(f"{name:40} | shape: {tuple(param.size())} | grad: {param.requires_grad}")
