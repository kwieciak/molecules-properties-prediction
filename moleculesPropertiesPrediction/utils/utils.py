import csv
import os
from datetime import datetime

import pandas as pd
import psutil
import torch
from matplotlib import pyplot as plt


def print_memory_usage():
    ram_memory = psutil.virtual_memory()
    free_ram = ram_memory.available / (1024 ** 3)
    total_ram = ram_memory.total / (1024 ** 3)
    print(f"Free RAM: {free_ram:.2f} GB / {total_ram:.2f} GB")


def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"VRAM usage: {gpu_memory:.2f} GB")


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_results_folder():
    os.makedirs("results", exist_ok=True)


def save_loss_to_csv(loss_list, filename):
    ensure_results_folder()
    timestamp = get_timestamp()
    filename = f"results/{filename}_{timestamp}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])
        for epoch, loss in enumerate(loss_list, start=1):
            writer.writerow([epoch, loss])


def save_metrics_to_csv(metrics_dict, filename):
    ensure_results_folder()
    timestamp = get_timestamp()
    filename = f"results/{filename}_{timestamp}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for metric, value in metrics_dict.items():
            writer.writerow([metric, value])

def save_preds_targets_to_csv(preds,targets,filename):
    ensure_results_folder()
    timestamp = get_timestamp()
    filename = f"results/{filename}_{timestamp}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prediction", "Target"])
        for pred, target in zip(preds, targets):
            writer.writerow([pred, target])

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




# TODO: funkcja do tworzenia plotu, parity plot
